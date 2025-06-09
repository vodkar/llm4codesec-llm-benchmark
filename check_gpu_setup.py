#!/usr/bin/env python3
"""
GPU Setup Verification Script

This script checks if your system is properly configured for GPU-accelerated
LLM benchmarking and verifies that models will use CUDA.
"""

import logging
import sys

import torch
import transformers

from benchmark_framework import BenchmarkConfig, HuggingFaceLLM, ModelType, TaskType
from config_manager import BenchmarkConfigManager


def check_torch_cuda():
    """Check PyTorch CUDA configuration."""
    print("=" * 60)
    print("PyTorch CUDA Configuration")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

        # Memory info
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"GPU Memory - Total: {memory_total:.2f}GB")
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB")
        print(f"GPU Memory - Reserved: {memory_reserved:.2f}GB")
        print(f"GPU Memory - Available: {memory_total - memory_reserved:.2f}GB")

        return True
    else:
        print("❌ CUDA not available - models will run on CPU")
        return False


def check_transformers():
    """Check transformers library configuration."""
    print("\n" + "=" * 60)
    print("Transformers Library Configuration")
    print("=" * 60)

    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate available: {hasattr(transformers, 'accelerate')}")

    try:
        import accelerate

        print(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        print("❌ Accelerate not available")

    try:
        import bitsandbytes

        print(f"BitsAndBytes version: {bitsandbytes.__version__}")
    except ImportError:
        print("❌ BitsAndBytes not available (quantization disabled)")


def test_model_loading():
    """Test loading a small model to verify GPU setup."""
    print("\n" + "=" * 60)
    print("Model Loading Test")
    print("=" * 60)

    # Create a test configuration with a smaller model for testing
    test_config = BenchmarkConfig(
        model_name="microsoft/DialoGPT-small",  # Small test model
        model_type=ModelType.HUGGINGFACE,
        task_type=TaskType.BINARY_VULNERABILITY,
        description="Test configuration",
        dataset_path="./test.json",
        output_dir="./test_output",
        max_tokens=50,
        temperature=0.1,
        use_quantization=False,  # Disable for small model
    )

    try:
        print("Loading test model...")
        llm = HuggingFaceLLM(test_config)
        print(f"✓ Model loaded successfully on device: {llm.device}")

        # Test generation
        print("Testing text generation...")
        response, tokens = llm.generate_response(
            "You are a code security expert.",
            "Is this code vulnerable? def test(): pass",
        )
        print(f"✓ Generation successful, used {tokens} tokens")
        print(f"Response preview: {response[:100]}...")

        # Cleanup
        llm.cleanup()
        print("✓ Model cleanup successful")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False


def check_benchmark_config():
    """Check benchmark configuration."""
    print("\n" + "=" * 60)
    print("Benchmark Configuration Test")
    print("=" * 60)

    try:
        # Test configuration creation
        config = BenchmarkConfigManager.create_config(
            model_key="qwen2.5-7b",
            task_key="binary_vulnerability",
            dataset_path="./data/sample_dataset.json",
            output_dir="./test_output",
        )

        print("✓ Configuration created successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Task: {config.task_type.value}")
        print(f"  Quantization: {config.use_quantization}")

        return True

    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        return False


def main():
    """Run all GPU setup checks."""
    print("LLM4CodeSec GPU Setup Verification")
    print(
        "This script will verify that your system is properly configured for GPU acceleration.\n"
    )

    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise

    checks = [
        ("PyTorch CUDA", check_torch_cuda),
        ("Transformers", check_transformers),
        ("Benchmark Config", check_benchmark_config),
        ("Model Loading", test_model_loading),
    ]

    results = {}

    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Setup Verification Summary")
    print("=" * 60)

    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print(
            "\n🎉 All checks passed! Your system is ready for GPU-accelerated benchmarking."
        )
        print("\nNext steps:")
        print("1. Run: poetry run python run_benchmark.py --quick")
        print("2. Monitor GPU usage with: watch -n 1 nvidia-smi")
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon solutions:")
        print(
            "- Ensure you have CUDA-enabled PyTorch: poetry add torch --source pytorch"
        )
        print("- Install missing dependencies: poetry install")
        print("- Check GPU drivers: nvidia-smi")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
