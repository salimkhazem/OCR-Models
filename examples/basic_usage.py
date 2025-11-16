
from deepseek_ocr import create_model


def example_basic_ocr() -> None:
    """Basic OCR example."""
    print("Example 1: Basic OCR")
    print("-" * 50)

    # Create a model
    model = create_model("deepseek", device="cpu")

    #    Process an image (replace with actual image path)
    # result = model.process("path/to/image.jpg")
    # print(result.text)

    print("Model created successfully!")
    print()


def example_different_models() -> None:
    """Example using different models."""
    print("Example 2: Using Different Models")
    print("-" * 50)

    models = ["deepseek", "smoldocling", "granite"]

    for model_name in models:
        try:
            model = create_model(model_name, device="cpu")
            info = model.get_model_info()
            print(f"{model_name}: {info.description}")
        except Exception as e:
            print(f"{model_name}: Error - {e}")

    print()


def example_custom_prompt() -> None:
    """Example with custom prompt."""
    print("Example 3: Custom Prompt")
    print("-" * 50)

    model = create_model("smoldocling", device="cpu")

    # Custom prompt example
    custom_prompt = "Extract all text from this document, preserving the structure."

    # result = model.process("path/to/image.jpg", prompt=custom_prompt)
    # print(result.text)

    print("Custom prompt configured!")
    print()


def example_batch_processing() -> None:
    """Example of batch processing."""
    print("Example 4: Batch Processing")
    print("-" * 50)

    model = create_model("granite", device="cpu")

    # List of images to process
    images = ["image1.jpg", "image2.jpg", "image3.jpg"]

    ## results = model.process_batch(images)
    # for i, result in enumerate(results):
    #     print(f"Image {i+1}: {len(result.text)} characters")

    print("Batch processing configured!")
    print()


def example_output_formats() -> None:
    """Example of different output formats."""
    print("Example 5: Output Formats")
    print("-" * 50)

    from deepseek_ocr.output.formatters import convert_format

    model = create_model("deepseek", device="cpu")

    # result = model.process("path/to/image.jpg") ou png 

    # Different formats
    # markdown = convert_format(result, "markdown")
    # html = convert_format(result, "html")
    # json_str = convert_format(result, "json")
    # text = convert_format(result, "text")

    print("Output format conversion available!")
    print()


if __name__ == "__main__":
    example_basic_ocr()
    example_different_models()
    example_custom_prompt()
    example_batch_processing()
    example_output_formats()

