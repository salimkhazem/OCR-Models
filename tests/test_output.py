
import pytest

from deepseek_ocr.base import OCRResult
from deepseek_ocr.output.formatters import (
    convert_format,
    format_as_html,
    format_as_json,
    format_as_markdown,
    format_as_text,
)


def test_format_as_markdown() -> None:
    """Test markdown formatting."""
    result = OCRResult(text="# Title\n\nSome text", format="markdown")
    formatted = format_as_markdown(result)
    assert formatted == "# Title\n\nSome text"


def test_format_as_text() -> None:
    """Test plain text formatting."""
    result = OCRResult(text="# Title\n\nSome **bold** text", format="markdown")
    formatted = format_as_text(result)
    assert "Title" in formatted
    assert "bold" in formatted
    # Markdown should be removed
    assert "#" not in formatted or formatted.count("#") < result.text.count("#")


def test_format_as_html() -> None:
    """Test HTML formatting."""
    result = OCRResult(text="# Title\n\nSome text", format="markdown")
    formatted = format_as_html(result)
    assert "<html>" in formatted
    assert "<body>" in formatted
    assert "Title" in formatted


def test_format_as_json() -> None:
    """Test JSON formatting."""
    result = OCRResult(
        text="Test text", format="markdown", metadata={"key": "value"}
    )
    formatted = format_as_json(result)
    assert "Test text" in formatted
    assert '"format": "markdown"' in formatted
    assert '"key": "value"' in formatted


def test_convert_format() -> None:
    """Test format conversion."""
    result = OCRResult(text="Test", format="markdown")
    
    # Test all formats
    assert convert_format(result, "markdown") == "Test"
    assert convert_format(result, "md") == "Test"
    assert convert_format(result, "text") is not None
    assert convert_format(result, "html") is not None
    assert convert_format(result, "json") is not None


def test_convert_format_invalid() -> None:
    """Test format conversion with invalid format."""
    result = OCRResult(text="Test", format="markdown")
    with pytest.raises(ValueError, match="Unsupported format"):
        convert_format(result, "invalid_format")

