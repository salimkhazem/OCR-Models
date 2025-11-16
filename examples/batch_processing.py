"""Batch processing examples for ocr and for all the models """

from pathlib import Path

from deepseek_ocr import create_model
from deepseek_ocr.output.formatters import convert_format


def process_directory(input_dir: str, output_dir: str, model_name: str = "deepseek") -> None:
    """Process all images in a directory.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save results.
        model_name: Model to use for processing.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images")
    print(f"Using model: {model_name}")

    # Create model
    model = create_model(model_name, device="cpu")

    # Process images
    results = model.process_batch(
        [str(img) for img in image_files],
        output_format="markdown"
    )

    # Save results
    for image_file, result in zip(image_files, results):
        output_file = output_path / f"{image_file.stem}.md"
        formatted = convert_format(result, "markdown")
        output_file.write_text(formatted, encoding="utf-8")
        print(f"Saved: {output_file}")


def process_with_different_formats(
    images: list[str],
    model_name: str = "smoldocling"
) -> None:
    """Process images and save in multiple formats.

    Args:
        images: List of image paths.
        model_name: Model to use for processing.
    """
    model = create_model(model_name, device="cpu")
    results = model.process_batch(images)

    formats = ["markdown", "html", "json", "text"]

    for i, (image_path, result) in enumerate(zip(images, results)):
        image_name = Path(image_path).stem
        print(f"\nProcessing {image_name}...")

        for fmt in formats:
            formatted = convert_format(result, fmt)
            output_file = f"{image_name}_output.{fmt}"
            # Path(output_file).write_text(formatted, encoding="utf-8")
            print(f"  {fmt}: {len(formatted)} characters")


if __name__ == "__main__":
    # Example usage
    # process_directory("input_images/", "output_results/", "deepseek")
    print("Batch processing examples ready!")

