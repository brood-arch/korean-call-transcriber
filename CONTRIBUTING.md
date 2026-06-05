# Contributing to Korean Call Transcriber

Thank you for your interest! Here's how to contribute.

## Development Setup

1. Fork and clone the repo
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -e ".[dev]"`
4. Copy `.env.example` to `.env` and fill in required values

## Code Style

- Python 3.11+ compatible
- Use `ruff` for linting: `ruff check src/ tests/`
- English docstrings (Google style)
- Type hints on public functions

## Testing

- Write tests for any new functionality
- Run: `python -m pytest tests/ -v`
- All tests must pass before PR submission

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure CI passes (lint + test)
5. Open a PR with a description of changes

## Adding New Integrations

New integrations should go in `src/integrations/`:
- Use environment variables for credentials (never hardcode)
- Add corresponding test in `tests/`
- Update README.md and docs/architecture.md
- Add env vars to `.env.example`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, error message, steps to reproduce
