"""Command-line interface for deepseek-ocr."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

from deepseek_ocr.factory import ModelFactory, list_available_models
from deepseek_ocr.output.formatters import convert_format


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """DeepSeek-OCR: Modular OCR library for multiple models."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default="deepseek",
    help="Model to use (deepseek, smoldocling, granite)",
    type=click.Choice(["deepseek", "smoldocling", "granite"], case_sensitive=False),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: print to stdout)",
)
@click.option(
    "--format",
    "-f",
    default="markdown",
    type=click.Choice(["markdown", "html", "json", "text"], case_sensitive=False),
    help="Output format",
)
@click.option(
    "--prompt",
    "-p",
    help="Custom prompt/instruction for the model",
)
@click.option(
    "--device",
    "-d",
    help="Device to use (cuda, cpu, cuda:0, etc.)",
)
@click.option(
    "--config",
    "-c",
    help="JSON string with model-specific configuration",
)
def process(
    input_path: str,
    model: str,
    output: Optional[str],
    format: str,
    prompt: Optional[str],
    device: Optional[str],
    config: Optional[str],
) -> None:
    """Process an image with OCR.

    INPUT_PATH: Path to the input image file.
    """
    try:
        # Parse config if provided
        model_config: Dict[str, Any] = {}
        if config:
            try:
                model_config = json.loads(config)
            except json.JSONDecodeError as e:
                click.echo(f"Error parsing config JSON: {e}", err=True)
                sys.exit(1)

        # Create model
        click.echo(f"Loading model: {model}...", err=True)
        ocr_model = ModelFactory.create_model(model, device=device, **model_config)

        # Process image
        click.echo(f"Processing image: {input_path}...", err=True)
        result = ocr_model.process(
            image=input_path, prompt=prompt, output_format=format, **model_config
        )

        # Format output
        formatted_output = convert_format(result, format)

        # Write output
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted_output, encoding="utf-8")
            click.echo(f"Output saved to: {output}", err=True)
        else:
            click.echo(formatted_output)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("input_paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    default="deepseek",
    help="Model to use",
    type=click.Choice(["deepseek", "smoldocling", "granite"], case_sensitive=False),
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory for results",
)
@click.option(
    "--format",
    "-f",
    default="markdown",
    type=click.Choice(["markdown", "html", "json", "text"], case_sensitive=False),
    help="Output format",
)
@click.option(
    "--prompt",
    "-p",
    help="Custom prompt/instruction for the model",
)
@click.option(
    "--device",
    "-d",
    help="Device to use",
)
@click.option(
    "--config",
    "-c",
    help="JSON string with model-specific configuration",
)
def batch(
    input_paths: tuple,
    model: str,
    output_dir: str,
    format: str,
    prompt: Optional[str],
    device: Optional[str],
    config: Optional[str],
) -> None:
    """Process multiple images in batch.

    INPUT_PATHS: One or more paths to input image files.
    """
    if not input_paths:
        click.echo("Error: At least one input path required", err=True)
        sys.exit(1)

    try:
        # Parse config if provided
        model_config: Dict[str, Any] = {}
        if config:
            try:
                model_config = json.loads(config)
            except json.JSONDecodeError as e:
                click.echo(f"Error parsing config JSON: {e}", err=True)
                sys.exit(1)

        # Create model
        click.echo(f"Loading model: {model}...", err=True)
        ocr_model = ModelFactory.create_model(model, device=device, **model_config)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process images
        click.echo(f"Processing {len(input_paths)} images...", err=True)
        results = ocr_model.process_batch(
            list(input_paths), prompt=prompt, output_format=format, **model_config
        )

        # Save results
        for i, (input_path, result) in enumerate(zip(input_paths, results)):
            input_file = Path(input_path)
            output_file = output_path / f"{input_file.stem}.{format}"
            formatted_output = convert_format(result, format)
            output_file.write_text(formatted_output, encoding="utf-8")
            click.echo(f"Saved: {output_file}", err=True)

        click.echo(f"\nProcessed {len(results)} images successfully!", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def list_models() -> None:
    """List all available models."""
    models = list_available_models()
    click.echo("Available models:")
    for model in models:
        try:
            info = ModelFactory.get_model_info(model)
            click.echo(f"  - {model}: {info.description}")
        except Exception:
            click.echo(f"  - {model}")


@main.command()
@click.argument("model_name")
def info(model_name: str) -> None:
    """Show information about a specific model.

    MODEL_NAME: Name of the model to get information about.
    """
    try:
        model_info = ModelFactory.get_model_info(model_name)
        click.echo(f"Model: {model_info.name}")
        click.echo(f"Model ID: {model_info.model_id}")
        click.echo(f"Description: {model_info.description}")
        click.echo(f"Supports batch: {model_info.supports_batch}")
        click.echo(f"Supports PDF: {model_info.supports_pdf}")
        click.echo(f"Default format: {model_info.default_output_format}")
        if model_info.config:
            click.echo(f"Configuration: {json.dumps(model_info.config, indent=2)}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

