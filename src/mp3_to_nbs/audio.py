"""
Audio analysis pipeline.

Handles MP3 loading, onset detection, pitch tracking, and amplitude
extraction via librosa.  Returns a list of detected note events that
the converter can map to NBS notes.
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
# Analysis
# ---------------------------------------------------------------------------

def _detect_onsets(
    audio: AudioData,
    config: ConversionConfig,
) -> np.ndarray:
    """Find onset frames and convert to sample times.

    Returns an array of onset times in seconds.
    """
    # Onset strength envelope
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


def analyze(
    audio: AudioData,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Run the full analysis pipeline on loaded audio.

    Steps:
        1. Detect onset times.
        2. Track pitch across all frames.
        3. Compute per-frame RMS amplitude.
        4. For each onset, sample the pitch and amplitude at that time
           and emit a :class:`DetectedNote`.

    Args:
        audio: Loaded audio data.
        config: Conversion configuration.

    Returns:
        A list of :class:`DetectedNote` instances sorted by time.
    """
    onset_times = _detect_onsets(audio, config)

    # Pitch tracking
    if config.pitch_algorithm == PitchAlgorithm.PYIN:
        pitch_times, frequencies, voiced = _track_pitch_pyin(audio, config)
    else:
        pitch_times, frequencies, voiced = _track_pitch_piptrack(audio, config)

    # RMS amplitude envelope
    rms = _compute_rms(audio, config)
    rms_times = librosa.times_like(
        rms,
        sr=audio.sample_rate,
        hop_length=config.hop_length,
    )

    notes: list[DetectedNote] = []

    for onset_t in onset_times:
        # Find nearest pitch frame
        pitch_idx = int(np.argmin(np.abs(pitch_times - onset_t)))
        freq = float(frequencies[pitch_idx])

        # Skip unvoiced / silent frames
        if freq < config.min_frequency or freq > config.max_frequency:
            continue

        # Sample amplitude at onset
        rms_idx = int(np.argmin(np.abs(rms_times - onset_t)))
        amp = float(rms[rms_idx])

        # Apply velocity sensitivity curve
        amp = amp ** config.velocity_sensitivity
        amp = max(0.0, min(1.0, amp))

        # Estimate duration (time until next onset or end)
        note_idx = len(notes)
        duration = 0.0

        notes.append(DetectedNote(
            time=float(onset_t),
            frequency=freq,
            amplitude=amp,
            duration=duration,
        ))

    # Fill in durations (gap to next note, capped at 2 seconds)
    for i in range(len(notes) - 1):
        gap = notes[i + 1].time - notes[i].time
        notes[i] = DetectedNote(
            time=notes[i].time,
            frequency=notes[i].frequency,
            amplitude=notes[i].amplitude,
            duration=min(gap, 2.0),
        )
    if notes:
        notes[-1] = DetectedNote(
            time=notes[-1].time,
            frequency=notes[-1].frequency,
            amplitude=notes[-1].amplitude,
            duration=0.5,
        )

    logger.info("Analysis complete: %d notes detected", len(notes))
    return notes
