"""Tests for mp3_to_nbs.config."""

import pytest

from mp3_to_nbs.config import ConversionConfig, PitchAlgorithm, get_preset, PRESETS


class TestConversionConfig:
    """ConversionConfig dataclass tests."""

    def test_default_values(self):
        cfg = ConversionConfig()
        assert cfg.tempo == 10.0
        assert cfg.min_frequency == 65.0
        assert cfg.max_frequency == 2100.0
        assert cfg.onset_sensitivity == 0.35
        assert cfg.max_layers == 20
        assert cfg.instrument == "harp"
        assert cfg.quantize is True
        assert cfg.pitch_algorithm == PitchAlgorithm.PYIN

    def test_validate_valid(self):
        cfg = ConversionConfig()
        assert cfg.validate() == []

    def test_validate_bad_tempo(self):
        cfg = ConversionConfig(tempo=-1)
        errors = cfg.validate()
        assert any("tempo" in e for e in errors)

    def test_validate_bad_frequency_range(self):
        cfg = ConversionConfig(min_frequency=5000, max_frequency=100)
        errors = cfg.validate()
        assert any("max_frequency" in e for e in errors)

    def test_validate_bad_sensitivity(self):
        cfg = ConversionConfig(onset_sensitivity=1.5)
        errors = cfg.validate()
        assert any("onset_sensitivity" in e for e in errors)

    def test_validate_bad_layers(self):
        cfg = ConversionConfig(max_layers=0)
        errors = cfg.validate()
        assert any("max_layers" in e for e in errors)

    def test_validate_bad_hop_length(self):
        cfg = ConversionConfig(hop_length=16)
        errors = cfg.validate()
        assert any("hop_length" in e for e in errors)


class TestPresets:
    """Preset retrieval tests."""

    def test_all_presets_valid(self):
        for name in PRESETS:
            cfg = get_preset(name)
            assert cfg.validate() == [], f"Preset '{name}' has validation errors"

    def test_get_preset_case_insensitive(self):
        cfg = get_preset("FAITHFUL")
        assert cfg.tempo == 20.0

    def test_get_preset_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_preset_returns_copy(self):
        a = get_preset("default")
        b = get_preset("default")
        a.tempo = 99
        assert b.tempo != 99
