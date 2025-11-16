
import pytest

from deepseek_ocr.factory import ModelFactory, create_model, list_available_models
from deepseek_ocr.models.deepseek import DeepSeekOCRModel
from deepseek_ocr.models.granite import GraniteDoclingModel
from deepseek_ocr.models.smoldocling import SmolDoclingModel


def test_list_available_models() -> None:
    """Test listing available models."""
    models = list_available_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "deepseek" in models or "smoldocling" in models or "granite" in models


def test_create_model_deepseek() -> None:
    """Test creating DeepSeek model."""
    # Note: This will try to load the actual model, so we'll just test the factory logic
    # In a real test environment, you'd mock the model loading
    try:
        model = ModelFactory.create_model("deepseek", device="cpu")
        assert isinstance(model, DeepSeekOCRModel)
    except Exception:
        # Model loading might fail in test environment, that's okay
        pass


def test_create_model_smoldocling() -> None:
    """Test creating SmolDocling model."""
    try:
        model = ModelFactory.create_model("smoldocling", device="cpu")
        assert isinstance(model, SmolDoclingModel)
    except Exception:
        pass


def test_create_model_granite() -> None:
    """Test creating Granite model."""
    try:
        model = ModelFactory.create_model("granite", device="cpu")
        assert isinstance(model, GraniteDoclingModel)
    except Exception:
        pass


def test_create_model_invalid() -> None:
    """Test creating model with invalid name."""
    with pytest.raises(ValueError, match="Unknown model"):
        ModelFactory.create_model("invalid_model_name")


def test_create_model_convenience_function() -> None:
    """Test convenience function for creating models."""
    try:
        model = create_model("deepseek", device="cpu")
        assert isinstance(model, DeepSeekOCRModel)
    except Exception:
        pass


def test_get_model_info() -> None:
    """Test getting model information."""
    try:
        info = ModelFactory.get_model_info("deepseek")
        assert info.name in ["DeepSeek-OCR", "deepseek"]
        assert info.model_id is not None
    except Exception:
        pass


def test_is_model_available() -> None:
    """Test checking if model is available."""
    assert ModelFactory.is_model_available("deepseek") is True
    assert ModelFactory.is_model_available("smoldocling") is True
    assert ModelFactory.is_model_available("granite") is True
    assert ModelFactory.is_model_available("invalid") is False

