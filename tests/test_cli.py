from pathlib import Path

from click.testing import CliRunner

from whisper_cli.cli import cli


class FakeTranscriptions:
    def __init__(self, response="hello world"):
        self.response = response
        self.last_args = None

    def create(self, **kwargs):
        self.last_args = kwargs
        return self.response


class FakeAudio:
    def __init__(self, transcriptions=None):
        self.transcriptions = transcriptions or FakeTranscriptions()


class FakeClient:
    def __init__(self, transcriptions=None):
        self.audio = FakeAudio(transcriptions=transcriptions)


class FakeJsonResponse:
    def model_dump_json(self, indent=2):
        return '{"text": "hi"}'


def test_transcribe_reads_key_from_env_and_prints_stdout(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    media = tmp_path / "sample.mp3"
    media.write_bytes(b"audio")

    transcriptions = FakeTranscriptions(response="transcribed text")
    fake_client = FakeClient(transcriptions=transcriptions)

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr("whisper_cli.cli.build_client", lambda api_key: fake_client)

    result = runner.invoke(cli, ["transcribe", str(media)])

    assert result.exit_code == 0
    assert "transcribed text" in result.output
    assert transcriptions.last_args is not None
    assert transcriptions.last_args["model"] == "gpt-4o-mini-transcribe"
    assert transcriptions.last_args["response_format"] == "text"


def test_transcribe_prefers_cli_key_over_env(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    media = tmp_path / "sample.mp3"
    media.write_bytes(b"audio")

    seen = {}

    def fake_builder(api_key):
        seen["api_key"] = api_key
        return FakeClient()

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr("whisper_cli.cli.build_client", fake_builder)

    result = runner.invoke(cli, ["transcribe", str(media), "--api-key", "cli-key"])

    assert result.exit_code == 0
    assert seen["api_key"] == "cli-key"


def test_transcribe_writes_to_output_file(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    media = tmp_path / "sample.wav"
    media.write_bytes(b"audio")
    output_path = tmp_path / "out.txt"

    monkeypatch.setattr("whisper_cli.cli.build_client", lambda api_key: FakeClient())

    result = runner.invoke(
        cli,
        ["transcribe", str(media), "--api-key", "k", "--output-file", str(output_path)],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").strip() == "hello world"
    assert "Wrote transcription" in result.output


def test_transcribe_supports_model_and_response_format(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    media = tmp_path / "sample.wav"
    media.write_bytes(b"audio")

    transcriptions = FakeTranscriptions(response=FakeJsonResponse())
    fake_client = FakeClient(transcriptions=transcriptions)
    monkeypatch.setattr("whisper_cli.cli.build_client", lambda api_key: fake_client)

    result = runner.invoke(
        cli,
        [
            "transcribe",
            str(media),
            "--api-key",
            "k",
            "--model",
            "gpt-4o-transcribe",
            "--response-format",
            "json",
            "--language",
            "en",
            "--prompt",
            "meeting notes",
            "--temperature",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert '{"text": "hi"}' in result.output
    assert transcriptions.last_args is not None
    assert transcriptions.last_args["model"] == "gpt-4o-transcribe"
    assert transcriptions.last_args["response_format"] == "json"
    assert transcriptions.last_args["language"] == "en"
    assert transcriptions.last_args["prompt"] == "meeting notes"
    assert transcriptions.last_args["temperature"] == 0.0


def test_transcribe_requires_api_key_when_not_in_env(monkeypatch, tmp_path: Path):
    runner = CliRunner()
    media = tmp_path / "sample.mp3"
    media.write_bytes(b"audio")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli, ["transcribe", str(media)])

    assert result.exit_code == 2
    assert "missing API key" in result.output
