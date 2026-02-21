"""Tests for the end-to-end conversion pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from mp3_to_nbs.config import ConversionConfig
from mp3_to_nbs.converter import ConversionResult, convert


def _create_test_wav(path: Path, frequency: float = 440.0, duration: float = 2.0):
    """Write a simple sine-wave WAV file for testing."""
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = (0.7 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    sf.write(str(path), samples, sr)


class TestConvert:
    """End-to-end conversion tests."""

    def test_basic_conversion(self, tmp_path):
        wav = tmp_path / "test.wav"
        nbs = tmp_path / "test.nbs"
        _create_test_wav(wav)

        result = convert(wav, nbs)

        assert isinstance(result, ConversionResult)
        assert nbs.exists()
        assert result.audio_duration > 0
        assert result.elapsed > 0
        assert result.output_path == nbs

    def test_auto_output_path(self, tmp_path):
        wav = tmp_path / "song.wav"
        _create_test_wav(wav)

        result = convert(wav)

        expected = tmp_path / "song.nbs"
        assert result.output_path == expected
        assert expected.exists()

    def test_custom_config(self, tmp_path):
        wav = tmp_path / "test.wav"
        nbs = tmp_path / "test.nbs"
        _create_test_wav(wav)

        config = ConversionConfig(
            tempo=20.0,
            max_layers=5,
            onset_sensitivity=0.5,
        )
        result = convert(wav, nbs, config)

        assert isinstance(result, ConversionResult)
        assert nbs.exists()

    def test_invalid_config_raises(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_test_wav(wav)

        config = ConversionConfig(tempo=-1)
        with pytest.raises(ValueError, match="Invalid configuration"):
            convert(wav, config=config)

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert(tmp_path / "nonexistent.mp3")

    def test_progress_callback(self, tmp_path):
        wav = tmp_path / "test.wav"
        _create_test_wav(wav)

        steps = []

        def on_progress(step, frac):
            steps.append((step, frac))

        convert(wav, progress_callback=on_progress)

        # Should have received at least some progress updates
        assert len(steps) > 0
        step_names = {s[0] for s in steps}
        assert "Loading audio" in step_names
