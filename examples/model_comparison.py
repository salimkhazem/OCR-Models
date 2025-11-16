"""Compare different OCR models."""

from deepseek_ocr import create_model, list_available_models


def compare_models(image_path: str) -> None:
    """Compare output from different models.

    Args:
        image_path: Path to the image to process.
    """
    print("Model Comparison")
    print("=" * 50)

    models = list_available_models()
    results = {}

    for model_name in models:
        try:
            print(f"\nProcessing with {model_name}...")
            model = create_model(model_name, device="cpu")
            # result = model.process(image_path)
            # results[model_name] = result
            print(f"✓ {model_name} completed")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")

    # Print comparison
    print("\n" + "=" * 50)
    print("Comparison Results:")
    print("=" * 50)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Length: {len(result.text)} characters")
        print(f"  Format: {result.format}")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")


if __name__ == "__main__":
    # Replace with actual image path
    image = "path/to/image.jpg"
    # compare_models(image)
    print("Model comparison example ready!")

