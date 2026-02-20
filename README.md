# openai-whisper-api-cli

Simple Python CLI wrapper for OpenAI Speech-to-Text.

## Features
- Transcribe local audio/video files with OpenAI Speech-to-Text
- API key from `--api-key` or `OPENAI_API_KEY`
- Print result to stdout (default)
- Optionally write result to a file
- Supports multiple response formats

## Install (Recommended)

```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install openai-whisper-api-cli
```

Verify:

```bash
owhisper --version
```

Upgrade:

```bash
pipx upgrade openai-whisper-api-cli
```

## Install From Source

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Authentication

Set env var:

```bash
export OPENAI_API_KEY="sk-..."
```

Or pass key directly:

```bash
owhisper transcribe ./audio.mp3 --api-key "sk-..."
```

`--api-key` takes precedence over `OPENAI_API_KEY`.

## Usage

Print transcription to stdout:

```bash
owhisper transcribe ./audio.mp3
```

Write transcription to file:

```bash
owhisper transcribe ./audio.mp3 --output-file ./transcript.txt
```

Use custom model and response format:

```bash
owhisper transcribe ./audio.mp3 \
  --model gpt-4o-transcribe \
  --response-format json \
  --language en
```

### Supported models
- `gpt-4o-transcribe`
- `gpt-4o-mini-transcribe`
- `gpt-4o-transcribe-diarize`
- `whisper-1`

Model list source:
- https://developers.openai.com/api/docs/guides/speech-to-text/

## CI and Publishing

GitHub Actions workflows:
- `.github/workflows/ci.yml`: runs tests on push/PR
- `.github/workflows/publish.yml`: builds and publishes to PyPI on tag push (`v*`)

### PyPI trusted publishing setup
1. Create your project on PyPI: `openai-whisper-api-cli`.
2. In PyPI project settings, add a Trusted Publisher:
   - Owner: your GitHub org/user
   - Repository: `openai-whisper-api-cli`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub repo settings, ensure environment `pypi` exists (optional protection rules).

### Release

```bash
python -m pytest
python -m build
python -m twine check dist/*
git tag v0.1.0
git push origin v0.1.0
```

Pushing the tag triggers automated publish.

## Local manual publish (optional)

```bash
python -m build
python -m twine upload dist/*
```

## License

MIT (see `LICENSE`).
