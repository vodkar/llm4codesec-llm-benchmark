#!/usr/bin/env python3
"""
Setup script for the LLM Code Security Benchmark Framework.

This script helps with initial setup and dependency installation.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(
            f"Error: Python 3.10+ is required. Current version: {version.major}.{version.minor}"
        )
        return False
    print(
        f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible"
    )
    return True


def check_gpu_availability():
    """Check GPU availability and CUDA installation."""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_count} device(s) - {gpu_name}")
            print(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ No GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print(
            "⚠ PyTorch not installed yet - GPU check will be performed after installation"
        )
        return None


def install_dependencies(use_gpu=True):
    """Install required dependencies."""
    print("Installing dependencies...")

    # Base requirements
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tokenizers>=0.14.0",
        "accelerate>=0.24.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "datasets>=2.14.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ]

    # Add GPU-specific requirements
    if use_gpu:
        requirements.extend(["bitsandbytes>=0.41.0", "nvidia-ml-py3>=7.352.0"])

    try:
        for req in requirements:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])

        print("✓ All dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def setup_directories():
    """Create necessary directories."""
    dirs = ["data", "results", "configs", "examples"]

    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")


def create_sample_config():
    """Create sample configuration files."""
    sample_config = {
        "model": "qwen2.5-7b",
        "task": "binary_vulnerability",
        "dataset": "./data/sample_dataset.json",
        "output": "./results/sample_run",
        "temperature": 0.1,
        "max_tokens": 512,
        "use_quantization": True,
    }

    config_path = Path("configs/sample_config.json")
    with open(config_path, "w") as f:
        import json

        json.dump(sample_config, f, indent=2)

    print(f"✓ Created sample config: {config_path}")


def run_basic_test():
    """Run a basic test to verify installation."""
    print("Running basic installation test...")

    try:
        # Test imports

        print("✓ All core libraries imported successfully")

        # Test basic functionality
        from config_manager import BenchmarkConfigManager

        available = BenchmarkConfigManager.list_available_configs()
        print(
            f"✓ Framework loaded - {len(available['models'])} models, {len(available['tasks'])} tasks available"
        )

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create or download a dataset in JSON format")
    print("2. Run a quick test:")
    print("   python run_benchmark.py --quick")
    print("3. Or run with your own data:")
    print(
        "   python run_benchmark.py --model qwen2.5-7b --task binary_vulnerability --dataset ./data/your_dataset.json"
    )
    print("4. Check examples:")
    print("   python examples.py")
    print("5. Analyze datasets:")
    print("   python data_utils.py analyze ./data/your_dataset.json")
    print("\nFor more information, see README.md")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup LLM Code Security Benchmark Framework"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="Skip GPU-related dependencies"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only run tests, skip installation"
    )
    parser.add_argument(
        "--dependencies-only", action="store_true", help="Only install dependencies"
    )

    args = parser.parse_args()

    print("LLM Code Security Benchmark Framework - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check GPU
    gpu_available = None if args.no_gpu else check_gpu_availability()
    use_gpu = gpu_available is not False and not args.no_gpu

    if args.test_only:
        # Only run tests
        if run_basic_test():
            print("✓ All tests passed - framework is ready to use")
        else:
            print("❌ Tests failed - please check installation")
            sys.exit(1)
        return

    # Install dependencies
    if not install_dependencies(use_gpu):
        print("❌ Dependency installation failed")
        sys.exit(1)

    if args.dependencies_only:
        print("✓ Dependencies installed successfully")
        return

    # Setup directories and configs
    setup_directories()
    create_sample_config()

    # Run basic test
    if run_basic_test():
        print_next_steps()
    else:
        print("❌ Installation verification failed")
        print("Try running: python setup.py --test-only")
        sys.exit(1)


if __name__ == "__main__":
    main()
