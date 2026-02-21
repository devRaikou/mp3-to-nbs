# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-21

### Added

- Initial release
- MP3/WAV/FLAC/OGG to NBS conversion via CLI and Python API
- pYIN and piptrack pitch detection algorithms
- Onset detection with configurable sensitivity
- Smart instrument auto-selection based on frequency range
- Octave folding for out-of-range pitches
- Fine pitch (detune) support
- Layer allocation with collision avoidance
- Built-in configuration presets: `default`, `faithful`, `dense`, `minimal`
- Rich CLI with progress bars and conversion summaries
- Python API via `convert()` function
- Comprehensive test suite
- GitHub Actions CI (Python 3.9–3.12)
