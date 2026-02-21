"""
Audio analysis pipeline.

Handles audio loading, pitch tracking, onset detection, and amplitude
extraction via librosa. Uses multiple strategies to capture accurate
notes from complex, polyphonic audio:

1. **CQT peak detection**: uses the full Constant-Q Transform spectrum
   to find active notes with precise frequency and octave information.
   Unlike chromagram, CQT preserves octave — no guessing needed.
2. **pYIN pitch tracking**: runs on the harmonic component for precise
   single-voice melody tracking.
3. **Onset + multi-pitch**: detects polyphonic attacks using piptrack.

The strategies are combined and deduplicated.
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


def _compute_rms(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Compute per-frame RMS energy, normalised to [0, 1]."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_max = rms.max()
    if rms_max > 0:
        rms = rms / rms_max
    return rms


# ---------------------------------------------------------------------------
# Strategy 1: CQT-based note detection (primary — preserves octave)
# ---------------------------------------------------------------------------


def _extract_notes_cqt(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Extract notes using Constant-Q Transform peak picking.

    Unlike chromagram which folds all octaves together, the CQT
    preserves the full frequency spectrum. Each CQT bin maps to a
    specific MIDI note, giving us both the pitch class AND the
    correct octave directly.

    We pick the top active peaks per frame and group consecutive
    activations of the same note into note events.
    """
    hop = config.hop_length
    fmin = max(config.min_frequency, librosa.note_to_hz("C2"))
    n_bins = 48  # 4 octaves * 12 bins/octave (C2 to B5)

    # Compute CQT magnitude
    cqt = np.abs(
        librosa.cqt(
            y=y,
            sr=sr,
            hop_length=hop,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )

    # Convert to dB and normalise
    cqt_db = librosa.amplitude_to_db(cqt, ref=cqt.max())
    cqt_norm = (cqt_db - cqt_db.min()) / (cqt_db.max() - cqt_db.min() + 1e-8)

    # Get the actual frequency for each CQT bin
    cqt_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=12)

    times = librosa.times_like(cqt[0], sr=sr, hop_length=hop)
    rms = _compute_rms(y, sr, hop)

    # Activation threshold (notes below this are considered inactive)
    threshold = 0.25

    # Maximum simultaneous notes per frame
    max_simultaneous = 6

    # Track active note regions: bin_idx -> (start_frame, amplitudes)
    active: dict[int, tuple[int, list[float]]] = {}
    notes: list[DetectedNote] = []

    n_frames = cqt_norm.shape[1]

    for frame_idx in range(n_frames):
        rms_idx = min(frame_idx, len(rms) - 1)
        frame_rms = float(rms[rms_idx])

        # Find active bins this frame (above threshold)
        frame_values = cqt_norm[:, frame_idx]

        # Pick top peaks
        active_bins = set()
        top_indices = np.argsort(frame_values)[::-1][:max_simultaneous]
        for bin_idx in top_indices:
            if frame_values[bin_idx] >= threshold:
                active_bins.add(int(bin_idx))

        # End regions for bins that are no longer active
        ended_bins = [b for b in active if b not in active_bins]
        for bin_idx in ended_bins:
            start_frame, amps = active.pop(bin_idx)
            _emit_cqt_note(
                notes,
                bin_idx,
                start_frame,
                frame_idx,
                amps,
                times,
                cqt_freqs,
                config,
            )

        # Start or continue regions for active bins
        for bin_idx in active_bins:
            if bin_idx in active:
                active[bin_idx][1].append(frame_rms)
            else:
                active[bin_idx] = (frame_idx, [frame_rms])

    # Emit remaining active notes
    for bin_idx, (start_frame, amps) in active.items():
        _emit_cqt_note(
            notes,
            bin_idx,
            start_frame,
            n_frames - 1,
            amps,
            times,
            cqt_freqs,
            config,
        )

    return notes


def _emit_cqt_note(
    notes: list[DetectedNote],
    bin_idx: int,
    start_frame: int,
    end_frame: int,
    amps: list[float],
    times: np.ndarray,
    cqt_freqs: np.ndarray,
    config: ConversionConfig,
) -> None:
    """Emit a note from a CQT region."""
    start_time = float(times[min(start_frame, len(times) - 1)])
    end_time = float(times[min(end_frame, len(times) - 1)])
    duration = end_time - start_time

    if duration < 0.03:  # Skip very short activations (< 30ms)
        return

    freq = float(cqt_freqs[bin_idx])
    amp = float(np.mean(amps))
    amp = amp**config.velocity_sensitivity
    amp = max(0.0, min(1.0, amp))

    if amp > 0.01:
        notes.append(
            DetectedNote(
                time=start_time,
                frequency=freq,
                amplitude=amp,
                duration=duration,
            )
        )


# ---------------------------------------------------------------------------
# Strategy 2: pYIN on harmonic component
# ---------------------------------------------------------------------------


def _extract_notes_pyin(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Track pitch on the harmonic component using pYIN.

    Groups consecutive voiced frames with similar pitch into note
    events. pYIN is very accurate for single-voice melody but only
    returns one pitch per frame.
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
# Strategy 3: Onset + multi-pitch
# ---------------------------------------------------------------------------


def _extract_notes_onset(
    y: np.ndarray,
    sr: int,
    config: ConversionConfig,
) -> list[DetectedNote]:
    """Detect onsets and sample multiple pitches at each onset.

    Uses piptrack to find multiple simultaneous pitches at each
    attack point.
    """
    hop = config.hop_length

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
    """Remove duplicate notes that are very close in time and frequency."""
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

    Combines three strategies for maximum coverage and accuracy:
        1. CQT peak detection — captures notes with correct octave
           directly from the frequency spectrum.
        2. pYIN on harmonic component — precise monophonic melody.
        3. Onset + multi-pitch — polyphonic attack detection.

    Results are merged, deduplicated, and sorted by time.
    """
    y = audio.samples
    sr = audio.sample_rate

    # Separate harmonic and percussive components
    logger.info("Separating harmonic/percussive components")
    y_harmonic, _ = librosa.effects.hpss(y)

    # Strategy 1: CQT on full audio (most accurate octave info)
    logger.info("Running CQT peak detection")
    cqt_notes = _extract_notes_cqt(y, sr, config)
    logger.info("CQT: %d notes", len(cqt_notes))

    # Strategy 2: pYIN on harmonic component (precise melody)
    logger.info("Running pYIN on harmonic component")
    pyin_notes = _extract_notes_pyin(y_harmonic, sr, config)
    logger.info("pYIN (harmonic): %d notes", len(pyin_notes))

    # Strategy 3: Onset + multi-pitch
    logger.info("Running onset + multi-pitch detection")
    onset_notes = _extract_notes_onset(y, sr, config)
    logger.info("Onset multi-pitch: %d notes", len(onset_notes))

    # Merge all notes
    all_notes = cqt_notes + pyin_notes + onset_notes

    # Deduplicate
    all_notes = _deduplicate_notes(all_notes)

    # Sort by time
    all_notes.sort(key=lambda n: n.time)

    logger.info(
        "Analysis complete: %d notes (cqt=%d, pyin=%d, onset=%d)",
        len(all_notes),
        len(cqt_notes),
        len(pyin_notes),
        len(onset_notes),
    )
    return all_notes
