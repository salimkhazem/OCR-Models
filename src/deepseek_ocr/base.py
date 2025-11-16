"""Base interface for OCR models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image


@dataclass
class ModelInfo:
    """Information about an OCR model."""

    name: str
    model_id: str
    description: str
    supports_batch: bool = True
    supports_pdf: bool = False
    default_output_format: str = "markdown"
    config: Optional[Dict[str, Any]] = None


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    format: str = "markdown"
    metadata: Optional[Dict[str, Any]] = None
    raw_output: Optional[Any] = None


class OCRModelError(Exception):
    """Base exception for OCR model errors."""

    pass


class ModelLoadError(OCRModelError):
    """Raised when model fails to load."""

    pass


class ProcessingError(OCRModelError):
    """Raised when processing fails."""

    pass


class OCRModel(ABC):
    """Abstract base class for OCR models.

    All OCR model implementations must inherit from this class and implement
    the required methods.
    """

    def __init__(self, model_id: str, device: Optional[str] = None):
        """Initialize the OCR model.

        Args:
            model_id: Hugging Face model identifier.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
                If None, will auto-detect.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._loaded = False

    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    @abstractmethod
    def load(self) -> None:
        """Load the model and tokenizer.

        Raises:
            ModelLoadError: If model loading fails.
        """
        pass

    @abstractmethod
    def process(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> OCRResult:
        """Process a single image.

        Args:
            image: Input image (path, Path object, or PIL Image).
            prompt: Optional prompt/instruction for the model.
            output_format: Desired output format (markdown, html, json, text).
            **kwargs: Additional model-specific parameters.

        Returns:
            OCRResult containing the processed text and metadata.

        Raises:
            ProcessingError: If processing fails.
        """
        pass

    @abstractmethod
    def process_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompt: Optional[str] = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Process multiple images in batch.

        Args:
            images: List of input images.
            prompt: Optional prompt/instruction for the model.
            output_format: Desired output format.
            **kwargs: Additional model-specific parameters.

        Returns:
            List of OCRResult objects.

        Raises:
            ProcessingError: If batch processing fails.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the model.

        Returns:
            ModelInfo object with model metadata.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self._loaded:
            self.load()

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """Load image from various input types.

        Args:
            image: Input image (path, Path object, or PIL Image).

        Returns:
            PIL Image object.

        Raises:
            ProcessingError: If image cannot be loaded.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, (str, Path)):
            try:
                return Image.open(image).convert("RGB")
            except Exception as e:
                raise ProcessingError(f"Failed to load image: {e}") from e
        else:
            raise ProcessingError(f"Unsupported image type: {type(image)}")

    def __enter__(self) -> "OCRModel":
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        # Cleanup if needed
        pass

