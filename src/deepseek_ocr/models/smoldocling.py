"""SmolDocling model implementation !! """
# TODO: Ajouter u n peu plus de flexibilité sur les tâches (ocr, layout, tables, charts...)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from deepseek_ocr.base import (
    ModelInfo,
    ModelLoadError,
    OCRModel,
    OCRResult,
    ProcessingError,
)


@dataclass
class SmolDoclingConfig:
    """Configuration for SmolDocling model."""

    max_new_tokens: int = 2048
    temperature: float = 0.0
    do_sample: bool = False
    task: str = "ocr"  # Options: ocr, layout, tables, charts, etc.

    @classmethod
    def get_task_prompt(cls, task: str) -> str:
        """Get prompt template for a specific task.

        Args:
            task: Task name (ocr, layout, tables, charts, etc.).

        Returns:
            Prompt string for the task.
        """
        prompts = {
            "ocr": "Perform OCR on this document.",
            "layout": "Extract the layout structure of this document.",
            "tables": "Extract all tables from this document.",
            "charts": "Extract chart information from this document.",
        }
        return prompts.get(task.lower(), prompts["ocr"])


class SmolDoclingModel(OCRModel):
    """SmolDocling model implementation.

    This model uses AutoProcessor and AutoModelForVision2Seq from transformers.
    It supports various document processing tasks including OCR, layout analysis,
    table extraction, and chart understanding.
    """

    MODEL_ID = "docling-project/SmolDocling-256M-preview"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[SmolDoclingConfig] = None,
    ):
        """Initialize SmolDocling model.

        Args:
            model_id: Hugging Face model identifier. Defaults to SmolDocling.
            device: Device to run the model on.
            config: Model configuration. If None, uses default.
        """
        super().__init__(model_id or self.MODEL_ID, device)
        self.config = config or SmolDoclingConfig()
        self._processor: Optional[AutoProcessor] = None

    def load(self) -> None:
        """Load the SmolDocling model and processor.

        Raises:
            ModelLoadError: If model loading fails.
        """
        if self._loaded:
            return

        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, trust_remote_code=True
            )

            self._model = self._model.eval()
            if self.device.startswith("cuda") and torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                self._model = self._model.to(torch.float32)

            self._loaded = True
        except Exception as e:
            raise ModelLoadError(f"Failed to load SmolDocling model: {e}") from e

    def process(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        output_format: str = "markdown",
        **kwargs: Any,
    ) -> OCRResult:
        """Process a single image with SmolDocling.

        Args:
            image: Input image (path, Path object, or PIL Image).
            prompt: Optional prompt/instruction. If None, uses task-based prompt.
            output_format: Desired output format.
            **kwargs: Additional parameters:
                - task: Task type (ocr, layout, tables, charts)
                - max_new_tokens: Maximum tokens to generate
                - temperature: Sampling temperature
                - do_sample: Whether to use sampling

        Returns:
            OCRResult containing the processed text.

        Raises:
            ProcessingError: If processing fails.
        """
        self._ensure_loaded()

        # Load image
        pil_image = self._load_image(image)

        # Prepare prompt
        if prompt is None:
            task = kwargs.get("task", self.config.task)
            prompt = SmolDoclingConfig.get_task_prompt(task)
        else:
            task = kwargs.get("task", self.config.task)

        # Get generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", self.config.max_new_tokens)
        temperature = kwargs.get("temperature", self.config.temperature)
        do_sample = kwargs.get("do_sample", self.config.do_sample)

        try:
            # Prepare inputs
            if hasattr(self._processor, "apply_chat_template"):
                # Use chat template if available
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                prompt_text = self._processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = self._processor(
                    images=[pil_image],
                    text=prompt_text,
                    return_tensors="pt",
                )
            else:
                # Fallback: simple text prompt
                inputs = self._processor(
                    images=[pil_image],
                    text=prompt,
                    return_tensors="pt",
                )

            # Move inputs to device
            if self.device.startswith("cuda") and torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )

            # Decode output
            generated_text = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract text (remove prompt if present)
            if prompt in generated_text:
                text = generated_text.replace(prompt, "").strip()
            else:
                text = generated_text.strip()

            return OCRResult(
                text=text,
                format=output_format,
                metadata={
                    "model": self.model_id,
                    "task": task,
                    "config": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "do_sample": do_sample,
                    },
                },
                raw_output=generated_text,
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
        """Get information about the SmolDocling model.

        Returns:
            ModelInfo object with model metadata.
        """
        return ModelInfo(
            name="SmolDocling",
            model_id=self.model_id,
            description="SmolDocling-256M: A compact document understanding model",
            supports_batch=True,
            supports_pdf=False,
            default_output_format="markdown",
            config={
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "tasks": ["ocr", "layout", "tables", "charts"],
            },
        )

