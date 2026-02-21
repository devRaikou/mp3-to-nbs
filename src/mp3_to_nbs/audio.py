"""
Audio analysis pipeline.

Handles audio loading, pitch tracking, onset detection, and amplitude
extraction via librosa. Uses multiple strategies to capture both
melody and bass from complex, polyphonic audio:

1. **Harmonic separation**: splits audio into harmonic and percussive
   components so melody can be tracked independently from bass.
2. **Multi-octave chromagram**: detects which pitch classes are active
   at each frame, producing a wider note range than single-voice pYIN.
3. **pYIN pitch tracking**: used on the harmonic component for precise
   single-voice pitch detection.

The strategies are combined and deduplicated to produce the final
note list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from mp3_to_nbs.config import ConversionConfig

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
        duration: Estimated duration in seconds.
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
    """Load an audio file and return mono samples at the given rate."""
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
# RMS amplitude
# ---------------------------------------------------------------------------


def _compute_rms(
    y: np.ndarray,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """Compute per-frame RMS energy, normalised to [0, 1]."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = rms.max()
    if rms_max > 0:
        rms = rms / rms_max
    return rms


# ---------------------------------------------------------------------------
# Strategy 1: Chromagram-based multi-octave note extraction
# ---------------------------------------------------------------------------


def _extract_notes_chroma(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Extract notes using chromagram analysis across multiple octaves.

    The CQT chromagram detects which pitch classes (C, C#, D, ..., B)
    are active at each frame. We then assign octaves based on the
    spectral centroid and energy distribution across frequency bands.

    This produces a much wider variety of notes than single-voice
    pitch tracking.
    """
    hop = config.hop_length

    # Compute CQT chromagram (12 pitch classes)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12, threshold=0.1)

    # Compute spectral centroid to estimate octave at each frame
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    # Compute RMS for amplitude
    rms = _compute_rms(y, sr, hop)

    times = librosa.times_like(chroma[0], sr=sr, hop_length=hop)

    # Base frequencies for each chroma bin in octave 3, 4, and 5
    chroma_base_freqs = {
        3: [
            130.81,
            138.59,
            146.83,
            155.56,
            164.81,
            174.61,
            185.00,
            196.00,
            207.65,
            220.00,
            233.08,
            246.94,
        ],
        4: [
            261.63,
            277.18,
            293.66,
            311.13,
            329.63,
            349.23,
            369.99,
            392.00,
            415.30,
            440.00,
            466.16,
            493.88,
        ],
        5: [
            523.25,
            554.37,
            587.33,
            622.25,
            659.26,
            698.46,
            739.99,
            783.99,
            830.61,
            880.00,
            932.33,
            987.77,
        ],
    }

    # Threshold for chroma energy activation
    chroma_threshold = 0.35

    notes: list[DetectedNote] = []
    # Track active regions per chroma bin
    active: dict[int, tuple[float, list[float]]] = {}  # bin -> (start_time, amps)

    for frame_idx in range(len(times)):
        t = float(times[frame_idx])
        rms_idx = min(frame_idx, len(rms) - 1)
        amp = float(rms[rms_idx])

        # Determine octave from spectral centroid
        cent = float(centroid[min(frame_idx, len(centroid) - 1)])
        if cent < 200:
            octave = 3
        elif cent < 500:
            octave = 4
        else:
            octave = 5

        for bin_idx in range(12):
            energy = float(chroma[bin_idx, frame_idx])

            if energy > chroma_threshold:
                if bin_idx not in active:
                    active[bin_idx] = (t, [amp])
                else:
                    active[bin_idx][1].append(amp)
            else:
                if bin_idx in active:
                    start_t, amps = active.pop(bin_idx)
                    duration = t - start_t

                    if duration >= 0.04:  # Minimum 40ms
                        mean_amp = float(np.mean(amps))
                        mean_amp = mean_amp**config.velocity_sensitivity
                        mean_amp = max(0.0, min(1.0, mean_amp))

                        if mean_amp > 0.01:
                            freq = chroma_base_freqs[octave][bin_idx]
                            notes.append(
                                DetectedNote(
                                    time=start_t,
                                    frequency=freq,
                                    amplitude=mean_amp,
                                    duration=duration,
                                )
                            )

    # Emit remaining active notes
    end_time = float(times[-1]) if len(times) > 0 else 0
    for bin_idx, (start_t, amps) in active.items():
        duration = end_time - start_t
        if duration >= 0.04:
            mean_amp = float(np.mean(amps))
            mean_amp = mean_amp**config.velocity_sensitivity
            mean_amp = max(0.0, min(1.0, mean_amp))
            if mean_amp > 0.01:
                freq = chroma_base_freqs[4][bin_idx]
                notes.append(
                    DetectedNote(
                        time=start_t,
                        frequency=freq,
                        amplitude=mean_amp,
                        duration=duration,
                    )
                )

    return notes


# ---------------------------------------------------------------------------
# Strategy 2: pYIN frame-based pitch tracking on harmonic component
# ---------------------------------------------------------------------------


def _extract_notes_pyin(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Track pitch on the harmonic component using pYIN.

    Groups consecutive voiced frames with similar pitch into note
    events.
    """
    hop = config.hop_length
    min_freq = config.min_frequency
    max_freq = config.max_frequency

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=min_freq, fmax=max_freq, sr=sr, hop_length=hop
    )
    f0 = np.nan_to_num(f0, nan=0.0)

    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    rms = _compute_rms(y, sr, hop)

    notes: list[DetectedNote] = []

    region_start = None
    region_freqs: list[float] = []
    region_amps: list[float] = []

    def _emit():
        if region_start is None or not region_freqs:
            return
        start_time = float(times[region_start])
        end_idx = region_start + len(region_freqs) - 1
        end_time = float(times[min(end_idx, len(times) - 1)])
        dur = max(end_time - start_time, 0.02)

        freq = float(np.median(region_freqs))
        amp = float(np.mean(region_amps))
        amp = amp**config.velocity_sensitivity
        amp = max(0.0, min(1.0, amp))

        if amp > 0.01:
            notes.append(
                DetectedNote(
                    time=start_time,
                    frequency=freq,
                    amplitude=amp,
                    duration=dur,
                )
            )

    for i in range(len(f0)):
        freq = float(f0[i])
        amp = float(rms[min(i, len(rms) - 1)])
        is_voiced = min_freq <= freq <= max_freq

        if is_voiced:
            if region_start is None:
                region_start = i
                region_freqs = [freq]
                region_amps = [amp]
            else:
                median_freq = np.median(region_freqs)
                ratio = freq / median_freq if median_freq > 0 else 999
                if 0.94 <= ratio <= 1.06:
                    region_freqs.append(freq)
                    region_amps.append(amp)
                else:
                    _emit()
                    region_start = i
                    region_freqs = [freq]
                    region_amps = [amp]
        else:
            if region_start is not None:
                _emit()
                region_start = None
                region_freqs = []
                region_amps = []

    _emit()
    return notes


# ---------------------------------------------------------------------------
# Strategy 3: Onset + multi-pitch detection
# ---------------------------------------------------------------------------


def _extract_notes_onset(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Detect onsets and sample multiple pitches at each onset.

    Uses piptrack to find multiple simultaneous pitches at each
    onset, producing polyphonic note detection.
    """
    hop = config.hop_length

    # Detect onsets
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    delta = 0.2 - 0.18 * config.onset_sensitivity
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop,
        onset_envelope=onset_env,
        delta=delta,
        backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    # Multi-pitch via piptrack
    pitches, magnitudes = librosa.piptrack(
        y=y,
        sr=sr,
        hop_length=hop,
        fmin=config.min_frequency,
        fmax=config.max_frequency,
    )

    rms = _compute_rms(y, sr, hop)
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop)

    notes: list[DetectedNote] = []
    min_freq = config.min_frequency
    max_freq = config.max_frequency

    for onset_t in onset_times:
        frame = int(
            np.argmin(np.abs(librosa.times_like(pitches[0], sr=sr, hop_length=hop) - onset_t))
        )

        # Find top 3 pitches by magnitude at this frame
        frame_mags = magnitudes[:, frame]
        frame_pitches = pitches[:, frame]

        top_bins = np.argsort(frame_mags)[::-1][:3]

        rms_idx = int(np.argmin(np.abs(rms_times - onset_t)))
        amp = float(rms[rms_idx])

        for bin_idx in top_bins:
            freq = float(frame_pitches[bin_idx])
            mag = float(frame_mags[bin_idx])

            if mag <= 0 or freq < min_freq or freq > max_freq:
                continue

            note_amp = amp**config.velocity_sensitivity
            note_amp = max(0.0, min(1.0, note_amp))

            if note_amp > 0.01:
                notes.append(
                    DetectedNote(
                        time=float(onset_t),
                        frequency=freq,
                        amplitude=note_amp,
                        duration=0.1,
                    )
                )

    return notes


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def _deduplicate_notes(
    notes: list[DetectedNote],
    time_tolerance: float = 0.05,
    freq_tolerance: float = 0.06,
) -> list[DetectedNote]:
    """Remove duplicate notes that are very close in time and frequency.

    Two notes are considered duplicates if they are within
    ``time_tolerance`` seconds and within ``freq_tolerance`` ratio
    of each other's frequency.
    """
    if not notes:
        return notes

    sorted_notes = sorted(notes, key=lambda n: (n.time, n.frequency))
    result: list[DetectedNote] = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        prev = result[-1]

        time_close = abs(note.time - prev.time) < time_tolerance
        if time_close and prev.frequency > 0:
            freq_ratio = note.frequency / prev.frequency
            freq_close = (1 - freq_tolerance) <= freq_ratio <= (1 + freq_tolerance)
        else:
            freq_close = False

        if time_close and freq_close:
            # Keep the louder one
            if note.amplitude > prev.amplitude:
                result[-1] = note
        else:
            result.append(note)

    return result


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------


def analyze(
    audio: AudioData,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Run the full analysis pipeline on loaded audio.

    Combines three strategies for maximum coverage:
        1. Chromagram analysis — captures active pitch classes with
           octave estimation from spectral centroid.
        2. pYIN on harmonic component — precise single-voice tracking.
        3. Onset + multi-pitch — detects polyphonic attacks.

    Results are merged, deduplicated, and sorted by time.
    """
    y = audio.samples
    sr = audio.sample_rate

    # Separate harmonic and percussive components
    logger.info("Separating harmonic/percussive components")
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Strategy 1: Chromagram on full audio
    logger.info("Running chromagram analysis")
    chroma_notes = _extract_notes_chroma(y, sr, config)
    logger.info("Chromagram: %d notes", len(chroma_notes))

    # Strategy 2: pYIN on harmonic component
    logger.info("Running pYIN pitch tracking on harmonic component")
    pyin_notes = _extract_notes_pyin(y_harmonic, sr, config)
    logger.info("pYIN (harmonic): %d notes", len(pyin_notes))

    # Strategy 3: Onset + multi-pitch
    logger.info("Running onset + multi-pitch detection")
    onset_notes = _extract_notes_onset(y, sr, config)
    logger.info("Onset multi-pitch: %d notes", len(onset_notes))

    # Merge all notes
    all_notes = chroma_notes + pyin_notes + onset_notes

    # Deduplicate
    all_notes = _deduplicate_notes(all_notes)

    # Sort by time
    all_notes.sort(key=lambda n: n.time)

    logger.info(
        "Analysis complete: %d notes detected (chroma=%d, pyin=%d, onset=%d)",
        len(all_notes),
        len(chroma_notes),
        len(pyin_notes),
        len(onset_notes),
    )
    return all_notes
