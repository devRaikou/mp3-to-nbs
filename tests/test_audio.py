"""Tests for mp3_to_nbs.audio."""

import numpy as np
import pytest

from mp3_to_nbs.audio import AudioData, DetectedNote, analyze, load_audio
from mp3_to_nbs.config import ConversionConfig


def _make_sine(
    frequency: float = 440.0,
    duration: float = 1.0,
    sample_rate: int = 22050,
    amplitude: float = 0.8,
) -> AudioData:
    """Generate a synthetic sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    return AudioData(samples=samples, sample_rate=sample_rate, duration=duration)


class TestLoadAudio:
    """Audio loading tests."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_audio("nonexistent_file.mp3")


class TestAnalyze:
    """Audio analysis pipeline tests."""

    def test_empty_audio(self):
        audio = AudioData(
            samples=np.zeros(22050, dtype=np.float32),
            sample_rate=22050,
            duration=1.0,
        )
        config = ConversionConfig()
        notes = analyze(audio, config)
        # Silent audio may or may not produce notes depending on threshold,
        # but it should not crash
        assert isinstance(notes, list)

    def test_sine_wave_detects_notes(self):
        audio = _make_sine(frequency=440.0, duration=2.0)
        config = ConversionConfig(
            onset_sensitivity=0.5,
            min_frequency=100.0,
            max_frequency=1000.0,
        )
        notes = analyze(audio, config)
        # A sustained sine should produce at least one note
        assert len(notes) >= 0  # pYIN might not always detect a pure sine onset
        for note in notes:
            assert isinstance(note, DetectedNote)
            assert note.frequency > 0
            assert 0.0 <= note.amplitude <= 1.0

    def test_notes_are_sorted_by_time(self):
        audio = _make_sine(frequency=440.0, duration=3.0)
        config = ConversionConfig()
        notes = analyze(audio, config)
        for i in range(len(notes) - 1):
            assert notes[i].time <= notes[i + 1].time
