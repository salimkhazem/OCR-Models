""" 
DeepSeek-OCR model implementation."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from deepseek_ocr.base import (
    ModelInfo,
    ModelLoadError,
    OCRModel,
    OCRResult,
    ProcessingError,
)


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek-OCR model."""

    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True
    test_compress: bool = False
    save_results: bool = False

    @classmethod
    def from_preset(cls, preset: str) -> "DeepSeekConfig":
        """Create config from preset name.

        Args:
            preset: One of 'tiny', 'small', 'base', 'large', 'gundam'.

        Returns:
            DeepSeekConfig instance.

        Raises:
            ValueError: If preset is unknown.
        """
        presets = {
            "tiny": cls(base_size=512, image_size=512, crop_mode=False),
            "small": cls(base_size=640, image_size=640, crop_mode=False),
            "base": cls(base_size=1024, image_size=1024, crop_mode=False),
            "large": cls(base_size=1280, image_size=1280, crop_mode=False),
            "gundam": cls(base_size=1024, image_size=640, crop_mode=True),
        }
        if preset.lower() not in presets:
            raise ValueError(
                f"Unknown preset: {preset}. Choose from {list(presets.keys())}"
            )
        return presets[preset.lower()]


class DeepSeekOCRModel(OCRModel):
    """DeepSeek-OCR model implementation.

    This model supports various size presets and can process images with
    different configurations for optimal performance.
    """

    MODEL_ID = "deepseek-ai/DeepSeek-OCR"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[DeepSeekConfig] = None,
        preset: Optional[str] = None,
    ):
        """Initialize DeepSeek-OCR model.

        Args:
            model_id: Hugging Face model identifier. Defaults to DeepSeek-OCR.
            device: Device to run the model on.
            config: Model configuration. If None, uses default or preset.
            preset: Preset name ('tiny', 'small', 'base', 'large', 'gundam').
                Overrides config if provided.
        """
        super().__init__(model_id or self.MODEL_ID, device)
        if preset:
            self.config = DeepSeekConfig.from_preset(preset)
        else:
            self.config = config or DeepSeekConfig()

    def load(self) -> None:
        """Load the DeepSeek-OCR model and tokenizer.

        Raises:
            ModelLoadError: If model loading fails.
        """
        if self._loaded:
            return

        try:
            # Set CUDA device if specified
            if self.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
                device_id = self.device.split(":")[-1] if ":" in self.device else "0"
                os.environ["CUDA_VISIBLE_DEVICES"] = device_id

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )

            # Try to use flash_attention_2 if available
            attn_implementation = "flash_attention_2"
            try:
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    _attn_implementation=attn_implementation,
                    trust_remote_code=True,
                    use_safetensors=True,
                )
            except Exception:
                # Fallback to default attention
                self._model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    use_safetensors=True,
                )

            self._model = self._model.eval()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self._model = self._model.cuda()
                self._model = self._model.to(torch.bfloat16)
            else:
                self._model = self._model.to(torch.float32)

            self._loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to load DeepSeek-OCR model: {e}") from e

    def process(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> OCRResult:
        """Process a single image with DeepSeek-OCR.

        Args:
            image: Input image (path, Path object, or PIL Image).
            prompt: Optional prompt. Defaults to "<image>\nFree OCR."
            output_format: Desired output format.
            **kwargs: Additional parameters:
                - base_size: Base size for processing
                - image_size: Image size for processing
                - crop_mode: Whether to use crop mode
                - test_compress: Whether to test compression
                - save_results: Whether to save results

        Returns:
            OCRResult containing the processed text.

        Raises:
            ProcessingError: If processing fails.
        """
        self._ensure_loaded()

        # Prepare image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = self._load_image(image)
        else:
            image_path = ""  # Will use temporary path if needed
            pil_image = image

        # Prepare prompt
        if prompt is None:
            prompt = "<image>\nFree OCR."

        # Get config from kwargs or use instance config
        config = DeepSeekConfig(
            base_size=kwargs.get("base_size", self.config.base_size),
            image_size=kwargs.get("image_size", self.config.image_size),
            crop_mode=kwargs.get("crop_mode", self.config.crop_mode),
            test_compress=kwargs.get("test_compress", self.config.test_compress),
            save_results=kwargs.get("save_results", self.config.save_results),
        )

        try:
            # Use model's infer method if available
            if hasattr(self._model, "infer"):
                output_path = kwargs.get("output_path", "")
                result = self._model.infer(
                    self._tokenizer,
                    prompt=prompt,
                    image_file=image_path if image_path else None,
                    output_path=output_path,
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                    test_compress=config.test_compress,
                    save_results=config.save_results,
                )
                # Extract text from result
                if isinstance(result, str):
                    text = result
                elif isinstance(result, dict) and "text" in result:
                    text = result["text"]
                else:
                    text = str(result)
            else:
                # Fallback: use standard transformers pipeline
                # This is a simplified version - actual implementation may vary
                raise ProcessingError(
                    "Model infer method not available. "
                    "Please ensure you're using the correct model version."
                )

            return OCRResult(
                text=text,
                format=output_format,
                metadata={
                    "model": self.model_id,
                    "config": {
                        "base_size": config.base_size,
                        "image_size": config.image_size,
                        "crop_mode": config.crop_mode,
                    },
                },
                raw_output=result,
            )
        except Exception as e:
            raise ProcessingError(f"Failed to process image: {e}") from e

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
            prompt: Optional prompt for all images.
            output_format: Desired output format.
            **kwargs: Additional parameters.

        Returns:
            List of OCRResult objects.
        """
        results = []
        for image in images:
            result = self.process(
                image=image, prompt=prompt, output_format=output_format, **kwargs
            )
            results.append(result)
        return results

    def get_model_info(self) -> ModelInfo:
        """Get information about the DeepSeek-OCR model.

        Returns:
            ModelInfo object with model metadata.
        """
        return ModelInfo(
            name="DeepSeek-OCR",
            model_id=self.model_id,
            description="DeepSeek-OCR: Contexts Optical Compression model",
            supports_batch=True,
            supports_pdf=False,
            default_output_format="markdown",
            config={
                "base_size": self.config.base_size,
                "image_size": self.config.image_size,
                "crop_mode": self.config.crop_mode,
                "presets": ["tiny", "small", "base", "large", "gundam"],
            },
        )

