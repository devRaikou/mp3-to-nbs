"""
Conversion configuration and presets.

Provides a dataclass-based configuration system with sensible defaults
and built-in presets for common use cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PitchAlgorithm(Enum):
    """Available pitch detection algorithms."""

    PYIN = "pyin"
    PIPTRACK = "piptrack"


@dataclass
class ConversionConfig:
    """Configuration for the MP3 → NBS conversion pipeline.

    Attributes:
        tempo: Ticks per second in the output NBS file. Higher values give
            finer time resolution but produce larger files. Minecraft's
            default redstone tick rate is 10 TPS.
        min_frequency: Lowest frequency (Hz) to consider during pitch
            detection. Notes below this are discarded.
        max_frequency: Highest frequency (Hz) to consider. Notes above
            this are discarded.
        onset_sensitivity: Controls how aggressively note onsets are
            detected. Range [0.0, 1.0] — lower values catch more notes
            but may introduce false positives.
        velocity_sensitivity: Exponent applied to the amplitude-to-velocity
            curve. Values < 1.0 compress dynamics, > 1.0 expand them.
        max_layers: Maximum number of NBS layers. Polyphonic notes that
            exceed this limit are dropped (quietest first).
        instrument: Default NBS instrument name. One of the vanilla
            Minecraft instruments (e.g. "harp", "bass", "flute").
        auto_instrument: When True, the converter attempts to pick the
            best instrument per note based on frequency range.
        quantize: Snap detected note times to the nearest tick boundary.
        pitch_algorithm: Which pitch detection backend to use.
        song_name: Metadata — song title embedded in the NBS header.
        song_author: Metadata — author name embedded in the NBS header.
        description: Metadata — description embedded in the NBS header.
        hop_length: STFT hop length in samples. Smaller values increase
            time resolution at the cost of compute.
    """

    tempo: float = 10.0
    min_frequency: float = 65.0
    max_frequency: float = 2100.0
    onset_sensitivity: float = 0.35
    velocity_sensitivity: float = 0.8
    max_layers: int = 20
    instrument: str = "harp"
    auto_instrument: bool = True
    quantize: bool = True
    pitch_algorithm: PitchAlgorithm = PitchAlgorithm.PYIN
    song_name: str = ""
    song_author: str = ""
    description: str = ""
    hop_length: int = 512

    def validate(self) -> list[str]:
        """Check configuration for invalid values.

        Returns:
            A list of human-readable error strings. Empty if valid.
        """
        errors: list[str] = []

        if self.tempo <= 0 or self.tempo > 100:
            errors.append(f"tempo must be in (0, 100], got {self.tempo}")
        if self.min_frequency <= 0:
            errors.append(f"min_frequency must be positive, got {self.min_frequency}")
        if self.max_frequency <= self.min_frequency:
            errors.append(
                f"max_frequency ({self.max_frequency}) must be greater "
                f"than min_frequency ({self.min_frequency})"
            )
        if not 0.0 <= self.onset_sensitivity <= 1.0:
            errors.append(
                f"onset_sensitivity must be in [0, 1], got {self.onset_sensitivity}"
            )
        if self.velocity_sensitivity <= 0:
            errors.append(
                f"velocity_sensitivity must be positive, got {self.velocity_sensitivity}"
            )
        if self.max_layers < 1 or self.max_layers > 256:
            errors.append(f"max_layers must be in [1, 256], got {self.max_layers}")
        if self.hop_length < 64 or self.hop_length > 8192:
            errors.append(f"hop_length must be in [64, 8192], got {self.hop_length}")

        return errors


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, ConversionConfig] = {
    "default": ConversionConfig(),
    "faithful": ConversionConfig(
        tempo=20.0,
        onset_sensitivity=0.2,
        velocity_sensitivity=1.0,
        max_layers=30,
        hop_length=256,
    ),
    "dense": ConversionConfig(
        tempo=20.0,
        onset_sensitivity=0.15,
        velocity_sensitivity=0.6,
        max_layers=40,
        hop_length=256,
    ),
    "minimal": ConversionConfig(
        tempo=5.0,
        onset_sensitivity=0.5,
        velocity_sensitivity=1.2,
        max_layers=10,
        hop_length=1024,
    ),
}


def get_preset(name: str) -> ConversionConfig:
    """Retrieve a named configuration preset.

    Args:
        name: Preset identifier (case-insensitive). Available presets:
            ``default``, ``faithful``, ``dense``, ``minimal``.

    Returns:
        A fresh :class:`ConversionConfig` instance with preset values.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    key = name.lower().strip()
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    # Return a copy so callers can mutate without affecting the original.
    import copy

    return copy.deepcopy(PRESETS[key])
