"""Click CLI for OpenAI Speech-to-Text API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
from openai import APIError, OpenAI

from whisper_cli import __version__

USER_INPUT_EXIT_CODE = 2
API_EXIT_CODE = 4

TRANSCRIBE_MODELS = (
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe-diarize",
    "whisper-1",
)


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """OpenAI speech-to-text CLI."""


@cli.command("transcribe")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--api-key", default=None, help="OpenAI API key. Falls back to OPENAI_API_KEY.")
@click.option("--model", type=click.Choice(TRANSCRIBE_MODELS), default="gpt-4o-mini-transcribe")
@click.option("--output-file", type=click.Path(dir_okay=False, path_type=str), default=None)
@click.option(
    "--response-format",
    type=click.Choice(["text", "json", "srt", "vtt", "verbose_json"]),
    default="text",
)
@click.option("--language", default=None, help="Optional language code, e.g. en.")
@click.option("--prompt", default=None, help="Optional prompt to guide transcription.")
@click.option("--temperature", type=float, default=None, help="Optional temperature for decoding.")
def transcribe(
    file_path: str,
    api_key: str | None,
    model: str,
    output_file: str | None,
    response_format: str,
    language: str | None,
    prompt: str | None,
    temperature: float | None,
) -> None:
    """Transcribe a local audio file using OpenAI Speech-to-Text."""
    resolved_key = resolve_api_key(api_key)
    client = build_client(resolved_key)

    request_args: dict[str, Any] = {
        "model": model,
        "response_format": response_format,
    }
    if language:
        request_args["language"] = language
    if prompt:
        request_args["prompt"] = prompt
    if temperature is not None:
        request_args["temperature"] = temperature

    try:
        with Path(file_path).open("rb") as audio_file:
            request_args["file"] = audio_file
            response = client.audio.transcriptions.create(**request_args)
    except APIError as exc:
        click.echo(f"API error: {exc}", err=True)
        raise click.exceptions.Exit(API_EXIT_CODE) from exc

    output_text = serialize_transcription_response(response, response_format=response_format)

    if output_file:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        click.echo(f"Wrote transcription to {out_path}")
        return

    click.echo(output_text)


def resolve_api_key(cli_key: str | None) -> str:
    if cli_key:
        return cli_key

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    click.echo("Error: missing API key. Use --api-key or set OPENAI_API_KEY.", err=True)
    raise click.exceptions.Exit(USER_INPUT_EXIT_CODE)


def build_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def serialize_transcription_response(response: Any, response_format: str) -> str:
    if isinstance(response, str):
        return response

    if response_format == "text" and hasattr(response, "text"):
        return str(response.text)

    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json(indent=2))

    if hasattr(response, "model_dump"):
        return str(response.model_dump())

    return str(response)
