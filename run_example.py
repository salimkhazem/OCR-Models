#!/usr/bin/env python3
"""Simple example script to test the installation of the package"""

import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from deepseek_ocr import create_model, list_available_models
    
    print("=" * 60)
    print("DeepSeek-OCR Installation Test")
    print("=" * 60)
    print()
    
    # Test 1: List available models
    print("1. Available models:")
    models = list_available_models()
    for model in models:
        print(f"   - {model}")
    print()
    
    # Test 2: Try creating a model (without loading)
    print("2. Testing model creation (without loading):")
    try:
        model = create_model("deepseek", device="cpu")
        info = model.get_model_info()
        print(f"   ✓ Created {info.name}")
        print(f"     Model ID: {info.model_id}")
        print(f"     Description: {info.description}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    print()
    
    print("=" * 60)
    print("Installation looks good!")
    print()
    print("Next steps:")
    print("  1. Install dependencies: pip install -e .")
    print("  2. Run CLI: deepseek-ocr list-models")
    print("  3. Process image: deepseek-ocr process image.jpg --model deepseek")
    print("=" * 60)
    
except ImportError as e:
    print(f"Error: {e}")
    print()
    print("Make sure you've installed the package:")
    print("  pip install -e .")
    sys.exit(1)

