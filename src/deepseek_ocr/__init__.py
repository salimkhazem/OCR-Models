"""Modular OCR library supporting DeepSeek-OCR, SmolDocling, and Granite-Docling models."""

from deepseek_ocr.factory import ModelFactory, create_model, list_available_models
from deepseek_ocr.base import OCRModel, OCRResult, ModelInfo

__version__ = "0.1.0"
__all__ = [
    "ModelFactory",
    "create_model",
    "list_available_models",
    "OCRModel",
    "OCRResult",
    "ModelInfo",
]

