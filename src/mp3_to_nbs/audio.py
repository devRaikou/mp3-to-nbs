"""
Audio analysis pipeline.

Handles MP3 loading, pitch tracking, onset detection, and amplitude
extraction via librosa.  Supports two analysis strategies:

- **Frame-based** (default): scans every frame for voiced pitch and groups
  consecutive voiced frames into note events.  Produces the most complete
  representation of the audio.
- **Onset-based**: only emits notes at detected onset (attack) times.
  Better for percussive / rhythmic material but can miss sustained tones.

The two strategies can also be combined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from mp3_to_nbs.config import ConversionConfig, PitchAlgorithm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedNote:
    """A single note event extracted from the audio signal.

    Attributes:
        time: Onset time in seconds.
        frequency: Fundamental frequency in Hz.
        amplitude: Normalised amplitude in [0.0, 1.0].
        duration: Estimated duration in seconds (may be ``0.0`` if
            unavailable).
    """

    time: float
    frequency: float
    amplitude: float
    duration: float = 0.0


@dataclass
class AudioData:
    """Container for loaded audio and its sample rate."""

    samples: np.ndarray
    sample_rate: int
    duration: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_audio(path: str | Path, sample_rate: int = 22050) -> AudioData:
    """Load an audio file and return mono samples at the given rate.

    Args:
        path: Path to the audio file (MP3, WAV, FLAC, OGG, …).
        sample_rate: Target sample rate in Hz.

    Returns:
        An :class:`AudioData` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        RuntimeError: If librosa cannot decode the file.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    logger.info("Loading audio: %s", filepath.name)

    try:
        samples, sr = librosa.load(str(filepath), sr=sample_rate, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to decode audio file: {exc}") from exc

    duration = float(librosa.get_duration(y=samples, sr=sr))
    logger.info(
        "Loaded %.1fs of audio (%d samples @ %d Hz)",
        duration,
        len(samples),
        sr,
    )
    return AudioData(samples=samples, sample_rate=sr, duration=duration)


# ---------------------------------------------------------------------------
# Pitch tracking
# ---------------------------------------------------------------------------

def _track_pitch_pyin(
    audio: AudioData,
    config: ConversionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pitch tracking using the pYIN algorithm.

    Returns:
        Tuple of (times, frequencies, voiced_flags).  Unvoiced frames
        have ``frequency == 0``.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio.samples,
        fmin=config.min_frequency,
        fmax=config.max_frequency,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    times = librosa.times_like(
        f0,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    # Replace NaN with 0
    f0 = np.nan_to_num(f0, nan=0.0)

    return times, f0, voiced_flag


def _track_pitch_piptrack(
    audio: AudioData,
    config: ConversionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pitch tracking using the piptrack algorithm.

    Returns:
        Tuple of (times, frequencies, magnitude_flags).
    """
    pitches, magnitudes = librosa.piptrack(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
        fmin=config.min_frequency,
        fmax=config.max_frequency,
    )

    # Pick the pitch with the highest magnitude per frame
    n_frames = pitches.shape[1]
    f0 = np.zeros(n_frames)
    mag = np.zeros(n_frames)

    for frame_idx in range(n_frames):
        magnitudes_frame = magnitudes[:, frame_idx]
        if magnitudes_frame.max() > 0:
            best_bin = magnitudes_frame.argmax()
            f0[frame_idx] = pitches[best_bin, frame_idx]
            mag[frame_idx] = magnitudes_frame[best_bin]

    times = librosa.times_like(
        f0,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    voiced = f0 > 0

    return times, f0, voiced


# ---------------------------------------------------------------------------
# RMS amplitude
# ---------------------------------------------------------------------------

def _compute_rms(
    audio: AudioData,
    config: ConversionConfig,
) -> np.ndarray:
    """Compute per-frame RMS energy, normalised to [0, 1]."""
    rms = librosa.feature.rms(
        y=audio.samples,
        hop_length=config.hop_length,
    )[0]

    rms_max = rms.max()
    if rms_max > 0:
        rms = rms / rms_max

    return rms


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def _detect_onsets(
    audio: AudioData,
    config: ConversionConfig,
) -> np.ndarray:
    """Find onset frames and convert to sample times.

    Returns an array of onset times in seconds.
    """
    onset_env = librosa.onset.onset_strength(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    # Adaptive threshold: map sensitivity 0–1 to delta 0.2–0.02
    delta = 0.2 - 0.18 * config.onset_sensitivity

    onset_frames = librosa.onset.onset_detect(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
        onset_envelope=onset_env,
        delta=delta,
        backtrack=True,
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )
    logger.info("Detected %d onsets", len(onset_times))
    return onset_times


# ---------------------------------------------------------------------------
# Frame-based note extraction (primary strategy)
# ---------------------------------------------------------------------------

def _extract_notes_from_frames(
    pitch_times: np.ndarray,
    frequencies: np.ndarray,
    rms: np.ndarray,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Group consecutive voiced frames into note events.

    Walks through every analysis frame.  When a voiced pitch is found,
    a note region starts and continues as long as the pitch stays within
    ±1 semitone (6%).  When the pitch jumps or drops to zero, the region
    ends and a :class:`DetectedNote` is emitted.

    This approach captures sustained notes, vibrato, and pitch bends
    much better than onset-only detection.
    """
    notes: list[DetectedNote] = []

    if len(frequencies) == 0:
        return notes

    n_frames = len(frequencies)
    min_freq = config.min_frequency
    max_freq = config.max_frequency

    # State for the current note region
    region_start: int | None = None
    region_freqs: list[float] = []
    region_amps: list[float] = []

    def _emit_region():
        """Finalise the current region into a DetectedNote."""
        if region_start is None or not region_freqs:
            return

        start_time = float(pitch_times[region_start])
        end_idx = region_start + len(region_freqs) - 1
        end_time = float(pitch_times[min(end_idx, n_frames - 1)])
        duration = max(end_time - start_time, 0.02)

        # Use the median frequency and mean amplitude for the region
        freq = float(np.median(region_freqs))
        amp = float(np.mean(region_amps))

        # Apply velocity sensitivity curve
        amp = amp ** config.velocity_sensitivity
        amp = max(0.0, min(1.0, amp))

        if amp > 0.01:  # Skip near-silent regions
            notes.append(DetectedNote(
                time=start_time,
                frequency=freq,
                amplitude=amp,
                duration=duration,
            ))

    for i in range(n_frames):
        freq = float(frequencies[i])
        amp = float(rms[min(i, len(rms) - 1)])

        is_voiced = min_freq <= freq <= max_freq

        if is_voiced:
            if region_start is None:
                # Start new region
                region_start = i
                region_freqs = [freq]
                region_amps = [amp]
            else:
                # Check if pitch is close to the current region
                median_freq = np.median(region_freqs)

                # Allow ±1 semitone (ratio of ~1.06)
                ratio = freq / median_freq if median_freq > 0 else 999
                if 0.94 <= ratio <= 1.06:
                    # Continue region
                    region_freqs.append(freq)
                    region_amps.append(amp)
                else:
                    # Pitch jump — end current region, start new one
                    _emit_region()
                    region_start = i
                    region_freqs = [freq]
                    region_amps = [amp]
        else:
            # Unvoiced frame — end current region
            if region_start is not None:
                _emit_region()
                region_start = None
                region_freqs = []
                region_amps = []

    # Emit final region
    _emit_region()

    return notes


# ---------------------------------------------------------------------------
# Chroma-based note extraction (fallback for complex audio)
# ---------------------------------------------------------------------------

def _extract_notes_from_chroma(
    audio: AudioData,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Extract notes using chromagram analysis.

    This is a fallback strategy for complex, polyphonic audio (e.g.
    full songs with drums, bass, and melody).  It uses the chromagram
    to detect which pitch classes are active at each time frame and
    emits a note for each active chroma bin.

    Produces many more notes than pYIN but sacrifices octave accuracy
    (chroma folds all octaves together).
    """
    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(
        y=audio.samples,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
        n_chroma=12,
    )

    times = librosa.times_like(
        chroma[0],
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    rms = _compute_rms(audio, config)

    notes: list[DetectedNote] = []

    # Threshold: only consider chroma bins above this energy
    chroma_threshold = 0.4

    # Note names to approximate frequency (octave 4)
    chroma_to_freq = [
        261.63,  # C4
        277.18,  # C#4
        293.66,  # D4
        311.13,  # D#4
        329.63,  # E4
        349.23,  # F4
        369.99,  # F#4
        392.00,  # G4
        415.30,  # G#4
        440.00,  # A4
        466.16,  # A#4
        493.88,  # B4
    ]

    # Track active notes per chroma bin to avoid duplicates
    active: dict[int, float] = {}  # bin -> start_time

    for frame_idx in range(len(times)):
        t = float(times[frame_idx])
        rms_idx = min(frame_idx, len(rms) - 1)
        amp = float(rms[rms_idx])

        for bin_idx in range(12):
            energy = float(chroma[bin_idx, frame_idx])

            if energy > chroma_threshold:
                if bin_idx not in active:
                    active[bin_idx] = t
            else:
                if bin_idx in active:
                    start_t = active.pop(bin_idx)
                    duration = t - start_t

                    if duration >= 0.05:  # Minimum 50ms
                        note_amp = amp ** config.velocity_sensitivity
                        note_amp = max(0.0, min(1.0, note_amp))

                        if note_amp > 0.01:
                            notes.append(DetectedNote(
                                time=start_t,
                                frequency=chroma_to_freq[bin_idx],
                                amplitude=note_amp,
                                duration=duration,
                            ))

    # Emit remaining active notes
    end_time = float(times[-1]) if len(times) > 0 else 0
    for bin_idx, start_t in active.items():
        duration = end_time - start_t
        if duration >= 0.05:
            notes.append(DetectedNote(
                time=start_t,
                frequency=chroma_to_freq[bin_idx],
                amplitude=0.5,
                duration=duration,
            ))

    return notes


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze(
    audio: AudioData,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Run the full analysis pipeline on loaded audio.

    Uses a multi-strategy approach:
        1. Run pitch tracking (pYIN or piptrack) and extract notes from
           consecutive voiced frames.
        2. If frame-based extraction finds very few notes (< 5), fall
           back to chromagram analysis which handles complex polyphonic
           audio better.
        3. Compute RMS amplitude for velocity mapping.

    Args:
        audio: Loaded audio data.
        config: Conversion configuration.

    Returns:
        A list of :class:`DetectedNote` instances sorted by time.
    """
    # --- Pitch tracking ---
    if config.pitch_algorithm == PitchAlgorithm.PYIN:
        pitch_times, frequencies, voiced = _track_pitch_pyin(audio, config)
    else:
        pitch_times, frequencies, voiced = _track_pitch_piptrack(audio, config)

    # --- RMS amplitude ---
    rms = _compute_rms(audio, config)

    # --- Primary strategy: frame-based note extraction ---
    notes = _extract_notes_from_frames(pitch_times, frequencies, rms, config)
    logger.info("Frame-based extraction: %d notes", len(notes))

    # --- Fallback: chromagram for complex audio ---
    if len(notes) < 5:
        logger.info(
            "Few notes from pitch tracking (%d) — trying chromagram analysis",
            len(notes),
        )
        chroma_notes = _extract_notes_from_chroma(audio, config)
        logger.info("Chromagram extraction: %d notes", len(chroma_notes))

        if len(chroma_notes) > len(notes):
            notes = chroma_notes

    # --- Sort by time ---
    notes.sort(key=lambda n: n.time)

    logger.info("Analysis complete: %d notes detected", len(notes))
    return notes
