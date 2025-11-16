"""Output formatters for OCR results."""

import json
from typing import Any, Dict, Optional

from deepseek_ocr.base import OCRResult


def format_as_markdown(result: OCRResult) -> str:
    """Format OCR result as Markdown.

    Args:
        result: OCRResult object.

    Returns:
        Markdown formatted string.
    """
    return result.text


def format_as_text(result: OCRResult) -> str:
    """Format OCR result as plain text.

    Args:
        result: OCRResult object.

    Returns:
        Plain text string.
    """
    # Remove markdown formatting if present
    text = result.text
    # Simple markdown removal (can be enhanced)
    text = text.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
    return text.strip()


def format_as_html(result: OCRResult) -> str:
    """Format OCR result as HTML.

    Args:
        result: OCRResult object.

    Returns:
        HTML formatted string.
    """
    # Convert markdown-like text to HTML
    html = result.text

    # Basic markdown to HTML conversion
    # Headers
    for i in range(6, 0, -1):
        html = html.replace("#" * i + " ", f"<h{i}>").replace(
            "\n" + "#" * i + " ", f"</h{i}>\n<h{i}>"
        )
        if html.startswith("#" * i + " "):
            html = f"<h{i}>" + html[len("#" * i + " ") :]
        # Close last header
        if html.endswith("\n" + "#" * i):
            html = html[: -len("\n" + "#" * i)] + f"</h{i}>"

    # Bold
    html = html.replace("**", "<strong>").replace("**", "</strong>")
    # Italic
    html = html.replace("*", "<em>").replace("*", "</em>")
    # Code
    html = html.replace("`", "<code>").replace("`", "</code>")
    # Line breaks
    html = html.replace("\n\n", "</p><p>").replace("\n", "<br>")
    html = f"<p>{html}</p>"

    return f"<!DOCTYPE html>\n<html>\n<head><title>OCR Result</title></head>\n<body>\n{html}\n</body>\n</html>"


def format_as_json(result: OCRResult) -> str:
    """Format OCR result as JSON.

    Args:
        result: OCRResult object.

    Returns:
        JSON formatted string.
    """
    data: Dict[str, Any] = {
        "text": result.text,
        "format": result.format,
        "metadata": result.metadata or {},
    }
    if result.raw_output is not None:
        data["raw_output"] = str(result.raw_output)
    return json.dumps(data, indent=2, ensure_ascii=False)


def convert_format(result: OCRResult, target_format: str) -> str:
    """Convert OCR result to a specific format.

    Args:
        result: OCRResult object.
        target_format: Target format ('markdown', 'html', 'json', 'text').

    Returns:
        Formatted string in the target format.

    Raises:
        ValueError: If target_format is not supported.
    """
    formatters = {
        "markdown": format_as_markdown,
        "md": format_as_markdown,
        "html": format_as_html,
        "json": format_as_json,
        "text": format_as_text,
        "txt": format_as_text,
    }

    target_format_lower = target_format.lower()
    if target_format_lower not in formatters:
        raise ValueError(
            f"Unsupported format: {target_format}. "
            f"Supported formats: {', '.join(formatters.keys())}"
        )

    return formatters[target_format_lower](result)

