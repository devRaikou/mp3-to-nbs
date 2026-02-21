# Contributing to mp3-to-nbs

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/mp3-to-nbs.git
cd mp3-to-nbs
```

2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

3. Install in development mode:

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=mp3_to_nbs --cov-report=term-missing
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Request Process

1. Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature
```

2. Make your changes and add tests where appropriate.

3. Ensure all tests pass and linting is clean.

4. Write a clear commit message describing your change.

5. Open a pull request against `main`.

## Reporting Issues

Found a bug or have a feature request? Open an issue with:

- A clear title and description
- Steps to reproduce (for bugs)
- Expected vs actual behaviour
- Your Python version and OS

## Code of Conduct

Be respectful and constructive. We're all here to build something cool.
