"""
Minecraft Note Block instrument definitions and frequency mapping.

Maps detected audio frequencies to the closest NBS instrument and key.
Minecraft note blocks have a 2-octave range (F#3–F#5) across 25 keys
(0–24), centered at F#4 (key 12, MIDI 66).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NBS key range: 0–24  →  F#3 (MIDI 54) to F#5 (MIDI 78)
NBS_KEY_MIN = 0
NBS_KEY_MAX = 24
NBS_KEY_CENTER = 12  # F#4

# Corresponding MIDI note numbers
MIDI_MIN = 54  # F#3
MIDI_MAX = 78  # F#5
MIDI_CENTER = 66  # F#4

# Reference tuning
A4_FREQUENCY = 440.0
A4_MIDI = 69


# ---------------------------------------------------------------------------
# Instrument data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instrument:
    """A vanilla Minecraft Note Block instrument.

    Attributes:
        id: NBS instrument index (0–15).
        name: Human-readable identifier.
        minecraft_name: Minecraft sound event suffix.
        low_midi: Suggested lower MIDI bound for auto-assignment.
        high_midi: Suggested upper MIDI bound for auto-assignment.
        priority: Tiebreaker when multiple instruments cover the same
            range — higher values are preferred.
    """

    id: int
    name: str
    minecraft_name: str
    low_midi: int
    high_midi: int
    priority: int = 0


# Vanilla instruments ordered by ID.  The MIDI bounds are rough guides
# for auto-instrument selection; they overlap intentionally so the
# priority field can break ties.
INSTRUMENTS: list[Instrument] = [
    Instrument(0, "harp", "harp", 54, 78, priority=10),
    Instrument(1, "double_bass", "bass", 30, 54, priority=8),
    Instrument(2, "bass_drum", "basedrum", 30, 54, priority=2),
    Instrument(3, "snare", "snare", 54, 78, priority=1),
    Instrument(4, "click", "hat", 66, 90, priority=1),
    Instrument(5, "guitar", "guitar", 42, 66, priority=7),
    Instrument(6, "flute", "flute", 66, 90, priority=6),
    Instrument(7, "bell", "bell", 78, 102, priority=5),
    Instrument(8, "chime", "chime", 78, 102, priority=4),
    Instrument(9, "xylophone", "xylophone", 78, 102, priority=3),
    Instrument(10, "iron_xylophone", "iron_xylophone", 54, 78, priority=3),
    Instrument(11, "cow_bell", "cow_bell", 66, 90, priority=2),
    Instrument(12, "didgeridoo", "didgeridoo", 30, 54, priority=5),
    Instrument(13, "bit", "bit", 54, 78, priority=4),
    Instrument(14, "banjo", "banjo", 54, 78, priority=6),
    Instrument(15, "pling", "pling", 54, 78, priority=5),
]

INSTRUMENT_BY_NAME: dict[str, Instrument] = {inst.name: inst for inst in INSTRUMENTS}
INSTRUMENT_BY_ID: dict[int, Instrument] = {inst.id: inst for inst in INSTRUMENTS}

# Melodic instruments — used for auto-instrument (excludes percussion).
_MELODIC_INSTRUMENTS = [
    inst for inst in INSTRUMENTS if inst.name not in ("bass_drum", "snare", "click")
]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def hz_to_midi(frequency: float) -> float:
    """Convert a frequency in Hz to a (fractional) MIDI note number.

    Uses the standard equal-temperament formula:
        ``midi = 69 + 12 * log2(freq / 440)``
    """
    if frequency <= 0:
        return 0.0
    return A4_MIDI + 12.0 * math.log2(frequency / A4_FREQUENCY)


def midi_to_hz(midi_note: float) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return A4_FREQUENCY * (2.0 ** ((midi_note - A4_MIDI) / 12.0))


def midi_to_nbs_key(midi_note: int) -> int:
    """Clamp a MIDI note to the NBS key range [0, 24].

    If the MIDI note falls outside F#3–F#5 it is octave-folded to the
    nearest valid key.
    """
    key = midi_note - MIDI_MIN
    # Octave-fold into range
    while key < NBS_KEY_MIN:
        key += 12
    while key > NBS_KEY_MAX:
        key -= 12
    return max(NBS_KEY_MIN, min(NBS_KEY_MAX, key))


def pick_instrument(
    midi_note: int,
    default_name: str = "harp",
) -> Instrument:
    """Select the best vanilla instrument for a given MIDI note.

    Iterates over melodic instruments, finds those whose suggested MIDI
    range covers the note, and returns the one with the highest priority.
    Falls back to the default instrument if nothing matches.

    Args:
        midi_note: The MIDI note number to match.
        default_name: Fallback instrument name.

    Returns:
        The best-matching :class:`Instrument`.
    """
    candidates = [
        inst for inst in _MELODIC_INSTRUMENTS if inst.low_midi <= midi_note <= inst.high_midi
    ]
    if not candidates:
        return INSTRUMENT_BY_NAME.get(default_name, INSTRUMENTS[0])
    return max(candidates, key=lambda i: i.priority)


def frequency_to_note(
    frequency: float,
    auto_instrument: bool = True,
    default_instrument: str = "harp",
) -> tuple[int, int, int]:
    """Map a detected frequency to NBS (instrument_id, key, fine_pitch).

    Args:
        frequency: Detected frequency in Hz.
        auto_instrument: Whether to auto-select the instrument.
        default_instrument: Fallback instrument name when auto is off.

    Returns:
        A tuple of ``(instrument_id, nbs_key, fine_pitch)`` where
        *fine_pitch* is in the range [-100, +100] cents.
    """
    midi_float = hz_to_midi(frequency)
    midi_rounded = int(round(midi_float))
    fine_pitch = int(round((midi_float - midi_rounded) * 100))
    fine_pitch = max(-100, min(100, fine_pitch))

    if auto_instrument:
        instrument = pick_instrument(midi_rounded)
    else:
        instrument = INSTRUMENT_BY_NAME.get(default_instrument, INSTRUMENTS[0])

    nbs_key = midi_to_nbs_key(midi_rounded)

    return instrument.id, nbs_key, fine_pitch
