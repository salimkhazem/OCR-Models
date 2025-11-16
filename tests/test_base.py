
from typing import Any

import pytest
from PIL import Image

from deepseek_ocr.base import (
    ModelInfo,
    ModelLoadError,
    OCRModel,
    OCRResult,
    ProcessingError,
)


class MockOCRModel(OCRModel):
    """Mock OCR model for testing."""

    def load(self) -> None:
        """Load the mock model."""
        self._loaded = True

    def process(
        self,
        image: str | Image.Image,
        prompt: str | None = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> OCRResult:
        """Process a single image."""
        self._ensure_loaded()
        return OCRResult(text="Mock OCR result", format=output_format)

    def process_batch(
        self,
        images: list[str | Image.Image],
        prompt: str | None = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> list[OCRResult]:
        """Process multiple images."""
        return [self.process(img, prompt, output_format, **kwargs) for img in images]

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        return ModelInfo(
            name="Mock Model",
            model_id="mock/model",
            description="A mock model for testing",
        )


def test_ocr_model_initialization() -> None:
    """Test OCR model initialization."""
    model = MockOCRModel("mock/model")
    assert model.model_id == "mock/model"
    assert not model._loaded


def test_ocr_model_load() -> None:
    """Test model loading."""
    model = MockOCRModel("mock/model")
    model.load()
    assert model._loaded


def test_ocr_model_context_manager() -> None:
    """Test context manager usage."""
    with MockOCRModel("mock/model") as model:
        assert model._loaded


def test_ocr_model_process() -> None:
    """Test image processing."""
    model = MockOCRModel("mock/model")
    result = model.process("dummy_path")
    assert isinstance(result, OCRResult)
    assert result.text == "Mock OCR result"


def test_ocr_model_process_batch() -> None:
    """Test batch processing."""
    model = MockOCRModel("mock/model")
    results = model.process_batch(["path1", "path2"])
    assert len(results) == 2
    assert all(isinstance(r, OCRResult) for r in results)


def test_ocr_model_get_model_info() -> None:
    """Test getting model information."""
    model = MockOCRModel("mock/model")
    info = model.get_model_info()
    assert isinstance(info, ModelInfo)
    assert info.name == "Mock Model"


def test_ocr_model_load_image_pil() -> None:
    """Test loading PIL Image."""
    model = MockOCRModel("mock/model")
    img = Image.new("RGB", (100, 100))
    loaded = model._load_image(img)
    assert isinstance(loaded, Image.Image)


def test_model_info() -> None:
    """Test ModelInfo dataclass."""
    info = ModelInfo(
        name="Test Model",
        model_id="test/model",
        description="Test description",
        supports_batch=True,
        supports_pdf=False,
    )
    assert info.name == "Test Model"
    assert info.supports_batch is True


def test_ocr_result() -> None:
    """Test OCRResult dataclass."""
    result = OCRResult(text="Test text", format="markdown")
    assert result.text == "Test text"
    assert result.format == "markdown"

