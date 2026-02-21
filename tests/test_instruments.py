"""Tests for mp3_to_nbs.instruments."""

import pytest

from mp3_to_nbs.instruments import (
    INSTRUMENTS,
    INSTRUMENT_BY_NAME,
    frequency_to_note,
    hz_to_midi,
    midi_to_hz,
    midi_to_nbs_key,
    pick_instrument,
)


class TestHzMidiConversion:
    """Frequency ↔ MIDI conversion tests."""

    def test_a4_is_midi_69(self):
        assert abs(hz_to_midi(440.0) - 69) < 0.01

    def test_c4_is_midi_60(self):
        # C4 ≈ 261.63 Hz
        assert abs(hz_to_midi(261.63) - 60) < 0.1

    def test_roundtrip(self):
        for midi in range(40, 90):
            hz = midi_to_hz(midi)
            recovered = hz_to_midi(hz)
            assert abs(recovered - midi) < 0.01

    def test_zero_frequency(self):
        assert hz_to_midi(0.0) == 0.0

    def test_negative_frequency(self):
        assert hz_to_midi(-100.0) == 0.0


class TestMidiToNbsKey:
    """MIDI → NBS key mapping tests."""

    def test_in_range(self):
        # MIDI 54 → key 0, MIDI 66 → key 12, MIDI 78 → key 24
        assert midi_to_nbs_key(54) == 0
        assert midi_to_nbs_key(66) == 12
        assert midi_to_nbs_key(78) == 24

    def test_below_range_folds(self):
        # MIDI 42 is 12 below 54, should fold to key 0
        key = midi_to_nbs_key(42)
        assert 0 <= key <= 24

    def test_above_range_folds(self):
        # MIDI 90 is 12 above 78, should fold to key 12
        key = midi_to_nbs_key(90)
        assert 0 <= key <= 24

    def test_all_keys_in_range(self):
        for midi in range(20, 110):
            key = midi_to_nbs_key(midi)
            assert 0 <= key <= 24


class TestPickInstrument:
    """Instrument auto-selection tests."""

    def test_harp_for_mid_range(self):
        inst = pick_instrument(66)  # F#4, center of harp range
        assert inst.name == "harp"

    def test_bass_for_low_range(self):
        inst = pick_instrument(40)
        assert inst.name in ("double_bass", "guitar", "didgeridoo")

    def test_fallback(self):
        inst = pick_instrument(120, default_name="harp")
        # Should still return something valid
        assert inst.id >= 0


class TestFrequencyToNote:
    """End-to-end frequency → NBS note tests."""

    def test_a4(self):
        inst_id, key, pitch = frequency_to_note(440.0)
        assert 0 <= key <= 24
        assert -100 <= pitch <= 100

    def test_low_frequency(self):
        inst_id, key, pitch = frequency_to_note(100.0)
        assert 0 <= key <= 24

    def test_returns_valid_instrument(self):
        inst_id, key, pitch = frequency_to_note(440.0)
        assert inst_id in range(16)
