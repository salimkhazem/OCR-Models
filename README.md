# DeepSeek-OCR

A modular Python library for OCR and document processing, supporting multiple state-of-the-art models including DeepSeek-OCR, SmolDocling, and Granite-Docling.

## Features

- **Multiple Model Support**: Unified interface for DeepSeek-OCR, SmolDocling, and Granite-Docling
- **Flexible Output Formats**: Markdown, HTML, JSON, and plain text
- **Easy-to-Use API**: Simple Python API and command-line interface
- **Batch Processing**: Process multiple images efficiently
- **Modular Design**: Clean, extensible architecture

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- PyTorch (will be installed with dependencies)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepSeek-OCR.git
cd DeepSeek-OCR

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies

For DeepSeek-OCR, you may need to install flash-attention separately:

```bash
pip install flash-attn==2.7.3 --no-build-isolation
```

## Quick Start

### Python API

```python
from deepseek_ocr import create_model

# Create a model
model = create_model("deepseek", device="cuda")

# Process an image
result = model.process("path/to/image.jpg", prompt="<image>\nFree OCR.")
print(result.text)
```

### Command Line

```bash
# Process a single image
deepseek-ocr process image.jpg --model deepseek --output result.md

# Process multiple images
deepseek-ocr batch image1.jpg image2.jpg --model smoldocling --output-dir results/

# List available models
deepseek-ocr list-models

# Get model information
deepseek-ocr info deepseek
```

## Supported Models

### DeepSeek-OCR

DeepSeek-OCR is a powerful OCR model with support for various size presets:

```python
from deepseek_ocr.models.deepseek import DeepSeekOCRModel, DeepSeekConfig

# Use a preset configuration
model = DeepSeekOCRModel(preset="gundam")  # Options: tiny, small, base, large, gundam

# Or use custom configuration
config = DeepSeekConfig(base_size=1024, image_size=640, crop_mode=True)
model = DeepSeekOCRModel(config=config)

result = model.process("image.jpg")
```

**Presets:**
- `tiny`: base_size=512, image_size=512, crop_mode=False
- `small`: base_size=640, image_size=640, crop_mode=False
- `base`: base_size=1024, image_size=1024, crop_mode=False
- `large`: base_size=1280, image_size=1280, crop_mode=False
- `gundam`: base_size=1024, image_size=640, crop_mode=True

### SmolDocling

SmolDocling is a compact document understanding model:

```python
from deepseek_ocr.models.smoldocling import SmolDoclingModel, SmolDoclingConfig

model = SmolDoclingModel(device="cuda")

# Process with different tasks
result = model.process("image.jpg", task="ocr")
result = model.process("image.jpg", task="layout")
result = model.process("image.jpg", task="tables")
result = model.process("image.jpg", task="charts")
```

### Granite-Docling

Granite-Docling is IBM's document understanding model:

```python
from deepseek_ocr.models.granite import GraniteDoclingModel

model = GraniteDoclingModel(device="cuda")
result = model.process("image.jpg", task="ocr")
```

## API Reference

### Factory Functions

#### `create_model(model_name, model_id=None, device=None, **kwargs)`

Create an OCR model instance.

**Parameters:**
- `model_name` (str): Model name ('deepseek', 'smoldocling', 'granite')
- `model_id` (str, optional): Hugging Face model ID
- `device` (str, optional): Device ('cuda', 'cpu', 'cuda:0', etc.)
- `**kwargs`: Model-specific configuration

**Returns:** `OCRModel` instance

**Example:**
```python
model = create_model("deepseek", device="cuda", preset="gundam")
```

#### `list_available_models()`

List all available model names.

**Returns:** List of model names

### OCRModel Interface

All models implement the `OCRModel` interface:

#### `load()`

Load the model and tokenizer.

#### `process(image, prompt=None, output_format="markdown", **kwargs)`

Process a single image.

**Parameters:**
- `image`: Image path, Path object, or PIL Image
- `prompt` (str, optional): Custom prompt/instruction
- `output_format` (str): Output format ('markdown', 'html', 'json', 'text')
- `**kwargs`: Model-specific parameters

**Returns:** `OCRResult` object

#### `process_batch(images, prompt=None, output_format="markdown", **kwargs)`

Process multiple images.

**Parameters:**
- `images`: List of images
- `prompt` (str, optional): Prompt for all images
- `output_format` (str): Output format
- `**kwargs`: Model-specific parameters

**Returns:** List of `OCRResult` objects

#### `get_model_info()`

Get model information.

**Returns:** `ModelInfo` object

### Output Formats

The library supports multiple output formats:

```python
from deepseek_ocr.output.formatters import convert_format

result = model.process("image.jpg")

# Convert to different formats
markdown = convert_format(result, "markdown")
html = convert_format(result, "html")
json_str = convert_format(result, "json")
text = convert_format(result, "text")
```

## Examples

### Basic Usage

```python
from deepseek_ocr import create_model

# Create and use a model
with create_model("deepseek") as model:
    result = model.process("document.jpg")
    print(result.text)
```

### Batch Processing

```python
from deepseek_ocr import create_model

model = create_model("smoldocling")
images = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = model.process_batch(images)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result.text[:100]}...")
```

### Custom Prompts

```python
model = create_model("granite")

# Custom OCR prompt
result = model.process(
    "document.jpg",
    prompt="Extract all text from this document, preserving formatting."
)

# Layout analysis
result = model.process(
    "document.jpg",
    prompt="Analyze the layout structure of this document."
)
```

### Model Comparison

```python
from deepseek_ocr import create_model

image = "test_image.jpg"
models = ["deepseek", "smoldocling", "granite"]

for model_name in models:
    model = create_model(model_name, device="cuda")
    result = model.process(image)
    print(f"{model_name}: {len(result.text)} characters")
```

## Command Line Interface

### Process Command

```bash
deepseek-ocr process INPUT_PATH [OPTIONS]
```

**Options:**
- `--model, -m`: Model to use (deepseek, smoldocling, granite)
- `--output, -o`: Output file path
- `--format, -f`: Output format (markdown, html, json, text)
- `--prompt, -p`: Custom prompt
- `--device, -d`: Device (cuda, cpu, cuda:0, etc.)
- `--config, -c`: JSON string with model-specific configuration

**Example:**
```bash
deepseek-ocr process doc.jpg --model deepseek --output result.md --format markdown
```

### Batch Command

```bash
deepseek-ocr batch INPUT_PATHS... [OPTIONS]
```

**Options:**
- `--output-dir, -o`: Output directory (required)
- Other options same as process command

**Example:**
```bash
deepseek-ocr batch img1.jpg img2.jpg --model smoldocling --output-dir results/
```

### List Models Command

```bash
deepseek-ocr list-models
```

### Info Command

```bash
deepseek-ocr info MODEL_NAME
```

## Configuration

### DeepSeek-OCR Configuration

```python
from deepseek_ocr.models.deepseek import DeepSeekConfig

config = DeepSeekConfig(
    base_size=1024,
    image_size=640,
    crop_mode=True,
    test_compress=False,
    save_results=False
)
model = DeepSeekOCRModel(config=config)
```

### SmolDocling/Granite Configuration

```python
from deepseek_ocr.models.smoldocling import SmolDoclingConfig

config = SmolDoclingConfig(
    max_new_tokens=2048,
    temperature=0.0,
    do_sample=False,
    task="ocr"
)
model = SmolDoclingModel(config=config)
```

## Development

### Running Tests

```bash
pytest tests/
```

### Linting

```bash
ruff check src/
```

### Type Checking

```bash
mypy src/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) by DeepSeek AI
- [SmolDocling](https://huggingface.co/docling-project/SmolDocling-256M-preview) by Docling Project
- [Granite-Docling](https://huggingface.co/ibm-granite/granite-docling-258M) by IBM

## Citation

If you use this library in your research, please cite the original models:

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

