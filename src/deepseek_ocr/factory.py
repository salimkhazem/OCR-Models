"""Factory for creating OCR models."""

from typing import Any, Dict, List, Optional, Type

from deepseek_ocr.base import ModelInfo, OCRModel
from deepseek_ocr.models.deepseek import DeepSeekOCRModel
from deepseek_ocr.models.granite import GraniteDoclingModel
from deepseek_ocr.models.smoldocling import SmolDoclingModel


class ModelFactory:
    """Factory for creating and managing OCR models.

    This factory provides a unified interface for creating different OCR models
    and accessing their information.
    """

    # Model registry mapping names/aliases to model classes
    _MODEL_REGISTRY: Dict[str, Type[OCRModel]] = {
        "deepseek": DeepSeekOCRModel,
        "deepseek-ocr": DeepSeekOCRModel,
        "deepseek_ocr": DeepSeekOCRModel,
        "smoldocling": SmolDoclingModel,
        "smol-docling": SmolDoclingModel,
        "smol_docling": SmolDoclingModel,
        "docling": SmolDoclingModel,  # Alias for SmolDocling
        "granite": GraniteDoclingModel,
        "granite-docling": GraniteDoclingModel,
        "granite_docling": GraniteDoclingModel,
        "granite-docling-258m": GraniteDoclingModel,
    }

    # Model IDs for direct lookup
    _MODEL_IDS: Dict[str, str] = {
        "deepseek": DeepSeekOCRModel.MODEL_ID,
        "smoldocling": SmolDoclingModel.MODEL_ID,
        "granite": GraniteDoclingModel.MODEL_ID,
    }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> OCRModel:
        """Create an OCR model instance.

        Args:
            model_name: Name or alias of the model (e.g., 'deepseek', 'smoldocling').
            model_id: Optional Hugging Face model ID. If None, uses default for model.
            device: Device to run the model on (e.g., 'cuda', 'cpu').
            **kwargs: Additional model-specific parameters.

        Returns:
            OCRModel instance.

        Raises:
            ValueError: If model_name is not recognized.

        Examples:
            >>> model = ModelFactory.create_model("deepseek")
            >>> model = ModelFactory.create_model("smoldocling", device="cuda:0")
            >>> model = ModelFactory.create_model("granite", max_new_tokens=4096)
        """
        # Normalize model name
        model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")

        # Check if it's a direct model ID
        if model_name_lower in cls._MODEL_REGISTRY:
            model_class = cls._MODEL_REGISTRY[model_name_lower]
            return model_class(model_id=model_id, device=device, **kwargs)

        # Check if it's a Hugging Face model ID
        if "/" in model_name:
            # Try to infer model type from model ID
            if "deepseek" in model_name_lower:
                return DeepSeekOCRModel(model_id=model_name, device=device, **kwargs)
            elif "smoldocling" in model_name_lower or "docling" in model_name_lower:
                return SmolDoclingModel(model_id=model_name, device=device, **kwargs)
            elif "granite" in model_name_lower:
                return GraniteDoclingModel(model_id=model_name, device=device, **kwargs)
            else:
                # Default to trying DeepSeek-OCR format
                return DeepSeekOCRModel(model_id=model_name, device=device, **kwargs)

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(cls.list_available_models())}"
        )

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available model names.

        Returns:
            List of model names/aliases.
        """
        # Return unique model names (without duplicates)
        unique_models = set()
        for name in cls._MODEL_REGISTRY.keys():
            # Get the canonical name (first occurrence)
            if name in ["deepseek", "smoldocling", "granite"]:
                unique_models.add(name)
            elif name not in ["deepseek-ocr", "deepseek_ocr", "smol-docling", "smol_docling", "docling", "granite-docling", "granite_docling", "granite-docling-258m"]:
                unique_models.add(name)
        return sorted(list(unique_models) + ["deepseek", "smoldocling", "granite"])

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelInfo:
        """Get information about a model.

        Args:
            model_name: Name or alias of the model.

        Returns:
            ModelInfo object.

        Raises:
            ValueError: If model_name is not recognized.
        """
        model = cls.create_model(model_name)
        return model.get_model_info()

    @classmethod
    def is_model_available(cls, model_name: str) -> bool:
        """Check if a model is available.

        Args:
            model_name: Name or alias of the model.

        Returns:
            True if model is available, False otherwise.
        """
        try:
            model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")
            return (
                model_name_lower in cls._MODEL_REGISTRY
                or "/" in model_name
                or any(model_name_lower in key for key in cls._MODEL_REGISTRY.keys())
            )
        except Exception:
            return False


# Convenience functions
def create_model(
    model_name: str,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> OCRModel:
    """Create an OCR model instance (convenience function).

    Args:
        model_name: Name or alias of the model.
        model_id: Optional Hugging Face model ID.
        device: Device to run the model on.
        **kwargs: Additional model-specific parameters.

    Returns:
        OCRModel instance.
    """
    return ModelFactory.create_model(model_name, model_id, device, **kwargs)


def list_available_models() -> List[str]:
    """List all available model names (convenience function).

    Returns:
        List of model names.
    """
    return ModelFactory.list_available_models()

