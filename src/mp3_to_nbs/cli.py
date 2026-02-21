"""
Command-line interface for mp3-to-nbs.

Provides a rich, user-friendly CLI with progress bars, preset selection,
and detailed conversion summaries.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from mp3_to_nbs import __version__
from mp3_to_nbs.config import PRESETS, ConversionConfig, PitchAlgorithm, get_preset
from mp3_to_nbs.converter import convert

console = Console()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mp3-to-nbs",
        description=(
            "Convert MP3 audio files to Minecraft Note Block Studio (.nbs) format.\n\n"
            "Analyses the input audio for pitch, onset, and amplitude information, "
            "then maps the detected notes onto Minecraft Note Block instruments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  mp3-to-nbs song.mp3\n"
            "  mp3-to-nbs song.mp3 -o output.nbs --tempo 20\n"
            "  mp3-to-nbs song.mp3 --preset faithful --verbose\n"
            "  mp3-to-nbs audio.wav -i flute --max-layers 30\n"
        ),
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to the input audio file (MP3, WAV, FLAC, OGG).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .nbs file path. Defaults to <input>.nbs.",
    )
    parser.add_argument(
        "-t",
        "--tempo",
        type=float,
        default=None,
        help="Ticks per second (default: 10). Higher = finer resolution.",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default=None,
        help="Default NBS instrument (e.g. harp, bass, flute, guitar).",
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default=None,
        choices=sorted(PRESETS.keys()),
        help="Use a built-in configuration preset.",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Maximum number of NBS layers (default: 20).",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=None,
        help="Onset detection sensitivity 0.0–1.0 (default: 0.35).",
    )
    parser.add_argument(
        "--pitch-algo",
        type=str,
        default=None,
        choices=["pyin", "piptrack"],
        help="Pitch detection algorithm (default: pyin).",
    )
    parser.add_argument(
        "--no-auto-instrument",
        action="store_true",
        help="Disable automatic instrument selection by frequency.",
    )
    parser.add_argument(
        "--song-name",
        type=str,
        default=None,
        help="Song name for the NBS header.",
    )
    parser.add_argument(
        "--author",
        type=str,
        default=None,
        help="Author name for the NBS header.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all output except errors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


# ---------------------------------------------------------------------------
# CLI logic
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    elif not quiet:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=verbose,
                show_path=verbose,
                rich_tracebacks=True,
            )
        ],
    )


def _build_config(args: argparse.Namespace) -> ConversionConfig:
    """Merge CLI arguments into a ConversionConfig."""
    config = get_preset(args.preset) if args.preset else ConversionConfig()

    # Override with explicit CLI flags
    if args.tempo is not None:
        config.tempo = args.tempo
    if args.instrument is not None:
        config.instrument = args.instrument
    if args.max_layers is not None:
        config.max_layers = args.max_layers
    if args.sensitivity is not None:
        config.onset_sensitivity = args.sensitivity
    if args.pitch_algo is not None:
        config.pitch_algorithm = PitchAlgorithm(args.pitch_algo)
    if args.no_auto_instrument:
        config.auto_instrument = False
    if args.song_name is not None:
        config.song_name = args.song_name
    if args.author is not None:
        config.song_author = args.author

    return config


def _print_banner(quiet: bool) -> None:
    """Print the startup banner."""
    if quiet:
        return

    banner = Text()
    banner.append("# ", style="bold magenta")
    banner.append("mp3-to-nbs", style="bold white")
    banner.append(f" v{__version__}", style="dim")

    console.print(
        Panel(
            banner,
            border_style="blue",
            padding=(0, 2),
        )
    )


def _print_summary(result, quiet: bool) -> None:
    """Print conversion results as a rich table."""
    if quiet:
        return

    table = Table(
        title="Conversion Summary",
        title_style="bold green",
        border_style="dim",
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Input", str(result.input_path.name))
    table.add_row("Output", str(result.output_path.name))
    table.add_row("Audio Duration", f"{result.audio_duration:.1f}s")
    table.add_row("Notes Detected", str(result.notes_detected))
    table.add_row("Notes Placed", str(result.notes_placed))
    if result.notes_dropped > 0:
        table.add_row("Notes Dropped", f"[yellow]{result.notes_dropped}[/yellow]")
    table.add_row("Layers Used", str(result.layers_used))
    table.add_row("Song Length", f"{result.song_ticks} ticks")
    table.add_row("Elapsed", f"{result.elapsed:.2f}s")

    console.print()
    console.print(table)
    console.print()
    console.print(f"  [bold green]OK[/bold green] Saved to [bold]{result.output_path}[/bold]")
    console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose, args.quiet)
    _print_banner(args.quiet)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {input_path}")
        return 1

    # Build configuration
    try:
        config = _build_config(args)
        errors = config.validate()
        if errors:
            for err in errors:
                console.print(f"[bold red]Error:[/bold red] {err}")
            return 1
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    # Run conversion with a progress display
    try:
        steps = ["Loading audio", "Analysing audio", "Mapping notes", "Writing NBS"]
        step_progress: dict[str, float] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TimeElapsedColumn(),
            console=console,
            disable=args.quiet,
        ) as progress:
            task_id = progress.add_task("Converting...", total=len(steps))

            def on_progress(step_name: str, fraction: float) -> None:
                if fraction >= 1.0 and step_name not in step_progress:
                    step_progress[step_name] = 1.0
                    progress.update(task_id, advance=1, description=step_name)

            result = convert(
                input_path=input_path,
                output_path=args.output,
                config=config,
                progress_callback=on_progress,
            )

        _print_summary(result, args.quiet)

    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except RuntimeError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 1
    except Exception as exc:
        logging.getLogger(__name__).debug("Unhandled exception", exc_info=True)
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
