<p align="center">
  <h1 align="center">🎵 mp3-to-nbs</h1>
  <p align="center">
    Convert MP3 audio files to Minecraft Note Block Studio (<code>.nbs</code>) format.
  </p>
  <p align="center">
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#how-it-works">How It Works</a> •
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

**mp3-to-nbs** is a command-line tool that analyses audio files and converts them into playable Note Block Studio songs. It uses spectral analysis to detect pitch, onset, and amplitude information from the input audio and maps the results onto Minecraft's vanilla note block instruments.

## Features

- **Automatic pitch detection** — uses the pYIN algorithm for robust fundamental frequency tracking
- **Smart instrument mapping** — assigns notes to the best-fitting vanilla instrument based on frequency range
- **Onset detection** — finds note attacks to produce natural-sounding rhythms
- **Configurable presets** — choose from `default`, `faithful`, `dense`, or `minimal` conversion profiles
- **Layer management** — distributes polyphonic notes across layers with overflow protection
- **Fine pitch support** — preserves pitch accuracy beyond the 25-key note block range
- **Multiple audio formats** — supports MP3, WAV, FLAC, OGG, and more
- **Beautiful CLI** — progress bars, conversion summaries, and coloured output via Rich

## Installation

### Requirements

- Python 3.9 or later
- [FFmpeg](https://ffmpeg.org/download.html) (required by librosa for MP3 decoding)

### From source

```bash
git clone https://github.com/devRaikou/mp3-to-nbs.git
cd mp3-to-nbs
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

## Usage

### Basic conversion

```bash
mp3-to-nbs song.mp3
```

This creates `song.nbs` in the same directory.

### Specify output path

```bash
mp3-to-nbs song.mp3 -o my_song.nbs
```

### Use a preset

```bash
mp3-to-nbs song.mp3 --preset faithful
```

### Custom tempo and instrument

```bash
mp3-to-nbs song.mp3 --tempo 20 --instrument flute
```

### Full options

```bash
mp3-to-nbs song.mp3 \
  -o output.nbs \
  --tempo 20 \
  --preset faithful \
  --instrument guitar \
  --max-layers 30 \
  --sensitivity 0.3 \
  --pitch-algo pyin \
  --song-name "My Song" \
  --author "devra" \
  --verbose
```

### All flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `<input>.nbs` | Output file path |
| `--tempo` | `-t` | `10.0` | Ticks per second |
| `--instrument` | `-i` | `harp` | Default instrument |
| `--preset` | `-p` | — | Configuration preset |
| `--max-layers` | — | `20` | Maximum NBS layers |
| `--sensitivity` | — | `0.35` | Onset sensitivity (0–1) |
| `--pitch-algo` | — | `pyin` | Pitch algorithm (`pyin` / `piptrack`) |
| `--no-auto-instrument` | — | — | Disable frequency-based instrument selection |
| `--song-name` | — | filename | NBS header song name |
| `--author` | — | — | NBS header author |
| `--verbose` | `-v` | — | Debug logging |
| `--quiet` | `-q` | — | Suppress output |

## Configuration

### Presets

| Preset | Tempo | Sensitivity | Max Layers | Best for |
|--------|-------|-------------|------------|----------|
| `default` | 10 TPS | 0.35 | 20 | General use |
| `faithful` | 20 TPS | 0.20 | 30 | High accuracy |
| `dense` | 20 TPS | 0.15 | 40 | Complex tracks |
| `minimal` | 5 TPS | 0.50 | 10 | Simple melodies |

Presets can be overridden with individual CLI flags:

```bash
mp3-to-nbs song.mp3 --preset faithful --tempo 15
```

### Python API

```python
from mp3_to_nbs import convert, ConversionConfig

config = ConversionConfig(
    tempo=20.0,
    max_layers=30,
    instrument="guitar",
    song_name="My Song",
    song_author="devra",
)

result = convert("song.mp3", "output.nbs", config)
print(f"Placed {result.notes_placed} notes in {result.elapsed:.1f}s")
```

## How It Works

The conversion pipeline has four stages:

```
┌─────────┐     ┌───────────┐     ┌─────────────┐     ┌───────────┐
│  Load   │────▶│  Analyse  │────▶│  Map Notes  │────▶│ Write NBS │
│  Audio  │     │  (pYIN)   │     │ (key+inst)  │     │  (pynbs)  │
└─────────┘     └───────────┘     └─────────────┘     └───────────┘
```

1. **Load Audio** — reads the input file via librosa, resamples to 22050 Hz mono.
2. **Analyse** — runs onset detection to find note attacks, pYIN pitch tracking for fundamental frequency estimation, and RMS for amplitude.
3. **Map Notes** — quantises timestamps to the NBS tick grid, converts Hz → MIDI → NBS key (with octave folding), selects the best instrument, and maps amplitude to velocity.
4. **Write NBS** — distributes notes across layers (avoiding collisions), sets header metadata, and saves via pynbs.

### Instruments

Minecraft has 16 vanilla note block instruments, each suited to a different frequency range:

| ID | Instrument | Range |
|----|-----------|-------|
| 0 | Harp | Mid (F#3–F#5) |
| 1 | Double Bass | Low (B1–F#3) |
| 5 | Guitar | Low-Mid (F#2–F#4) |
| 6 | Flute | Mid-High (F#4–F#6) |
| 7 | Bell | High (F#5–F#7) |
| 13 | Bit | Mid (F#3–F#5) |
| 14 | Banjo | Mid (F#3–F#5) |
| 15 | Pling | Mid (F#3–F#5) |

When auto-instrument is enabled (default), the converter picks the instrument whose range best matches each detected note's frequency.

## Project Structure

```
src/mp3_to_nbs/
├── __init__.py        # Package metadata & public API
├── __main__.py        # python -m entry point
├── cli.py             # CLI argument parser & Rich output
├── audio.py           # Audio loading & spectral analysis
├── converter.py       # Conversion pipeline orchestrator
├── nbs_writer.py      # NBS file builder (pynbs)
├── instruments.py     # Instrument definitions & mapping
└── config.py          # Configuration dataclass & presets
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
