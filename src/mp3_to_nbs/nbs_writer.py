"""
NBS file writer.

Builds an NBS (Note Block Song) file using the ``pynbs`` library.
Handles note placement across layers and sets header metadata.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pynbs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NBSNote:
    """A single note ready to be placed in the NBS file.

    Attributes:
        tick: Horizontal position (time) in the song.
        instrument: Vanilla instrument ID (0-15).
        key: NBS key (0-24).
        velocity: Note velocity (0-100).
        panning: Stereo panning (-100=left, 0=center, 100=right).
            Note: pynbs adds 100 when writing to disk (NBS spec: 0-200).
        pitch: Fine pitch adjustment in cents (-100 to +100).
    """

    tick: int
    instrument: int
    key: int
    velocity: int = 100
    panning: int = 0  # 0 = center (pynbs adds +100 when writing)
    pitch: int = 0


# ---------------------------------------------------------------------------
# Layer allocation
# ---------------------------------------------------------------------------


def _allocate_layers(
    notes: list[NBSNote],
    max_layers: int,
) -> list[tuple[NBSNote, int]]:
    """Assign each note to a layer, avoiding collisions on the same tick.

    Two notes cannot occupy the same (tick, layer) slot.  When all
    layers for a given tick are full, the quietest note is dropped.

    Args:
        notes: Notes sorted by tick.
        max_layers: Maximum number of layers to use.

    Returns:
        List of (note, layer_index) tuples.
    """
    placed: list[tuple[NBSNote, int]] = []

    by_tick: dict[int, list[NBSNote]] = defaultdict(list)
    for note in notes:
        by_tick[note.tick].append(note)

    for tick in sorted(by_tick):
        tick_notes = by_tick[tick]
        # Sort by velocity descending - drop quietest if overflow
        tick_notes.sort(key=lambda n: n.velocity, reverse=True)

        occupied: set[int] = set()
        for note in tick_notes:
            if len(occupied) >= max_layers:
                logger.debug(
                    "Layer overflow at tick %d - dropping note (key=%d, vel=%d)",
                    tick,
                    note.key,
                    note.velocity,
                )
                break

            # Find the first free layer
            layer = 0
            while layer in occupied:
                layer += 1

            occupied.add(layer)
            placed.append((note, layer))

    return placed


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_nbs(
    notes: list[NBSNote],
    output_path: str | Path,
    *,
    max_layers: int = 20,
    tempo: float = 10.0,
    song_name: str = "",
    song_author: str = "",
    original_author: str = "",
    description: str = "",
) -> Path:
    """Build and write an NBS file from a list of notes.

    Args:
        notes: The notes to include in the song.
        output_path: Destination file path.
        max_layers: Maximum number of layers to allocate.
        tempo: Song tempo in ticks per second.
        song_name: Song title for the NBS header.
        song_author: Author name for the NBS header.
        original_author: Original author for the NBS header.
        description: Song description for the NBS header.

    Returns:
        The resolved output :class:`Path`.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not notes:
        logger.warning("No notes to write - creating empty NBS file")

    # Sort notes by tick for consistent layer allocation
    sorted_notes = sorted(notes, key=lambda n: (n.tick, -n.velocity))

    # Allocate layers
    placed = _allocate_layers(sorted_notes, max_layers)

    # Build the pynbs file
    nbs_file = pynbs.new_file(
        song_name=song_name,
        song_author=song_author,
        original_author=original_author,
        description=description,
        tempo=tempo,
    )

    # Add notes
    for note, layer in placed:
        nbs_file.notes.append(
            pynbs.Note(
                tick=note.tick,
                layer=layer,
                instrument=note.instrument,
                key=note.key,
                velocity=note.velocity,
                panning=note.panning,
                pitch=note.pitch,
            )
        )

    # Rebuild ALL layers from scratch with correct settings.
    # pynbs auto-creates layers with pan=0 which can cause issues.
    max_layer_used = max((layer for _, layer in placed), default=0) if placed else 0
    total_layers = max_layer_used + 1

    nbs_file.layers.clear()
    for layer_id in range(total_layers):
        nbs_file.layers.append(
            pynbs.Layer(
                id=layer_id,
                name=f"Layer {layer_id + 1}",
                lock=False,
                volume=100,
                panning=0,  # 0 = center (pynbs adds +100 when writing to disk)
            )
        )

    # Update header to match actual content
    nbs_file.header.song_layers = total_layers
    nbs_file.header.song_length = max((n.tick for n in nbs_file.notes), default=0)

    # Save
    nbs_file.save(str(output))
    logger.info(
        "NBS file saved: %s (%d notes across %d layers)",
        output.name,
        len(placed),
        total_layers,
    )

    return output
