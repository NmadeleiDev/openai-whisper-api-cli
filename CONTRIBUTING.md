# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Release flow
- Update version in `pyproject.toml` and `whisper_cli/__init__.py`
- Run tests and build checks
- Create and push a `v*` tag to trigger PyPI publish workflow
