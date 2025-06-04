#!/usr/bin/env python3
"""
Easy-to-use benchmark runner script.

Usage examples:
    # Run binary vulnerability detection with Llama2
    python run_benchmark.py --model llama2-7b --task binary_vulnerability --dataset ./data/vulbench.json

    # Run CWE-79 detection with Qwen
    python run_benchmark.py --model qwen2.5-7b --task cwe79_detection --dataset ./data/xss_data.json

    # Run multiclass classification with DeepSeek
    python run_benchmark.py --model deepseek-coder --task multiclass_vulnerability --dataset ./data/mixed_vulns.json

    # Custom configuration
    python run_benchmark.py --config ./my_config.json
"""

import argparse
import json
import sys
from pathlib import Path

from benchmark_framework import BenchmarkRunner
from config_manager import BenchmarkConfigManager, create_sample_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LLM Code Security Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Configuration options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to configuration JSON file")
    group.add_argument(
        "--quick",
        action="store_true",
        help="Run quick setup with default configuration",
    )

    # Individual configuration parameters
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama2-7b", "qwen2.5-7b", "deepseek-coder", "codebert"],
        help="Model to use for benchmarking",
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "binary_vulnerability",
            "cwe79_detection",
            "cwe89_detection",
            "multiclass_vulnerability",
        ],
        help="Task type for benchmarking",
    )

    parser.add_argument("--dataset", type=str, help="Path to dataset file")

    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Model temperature (default: 0.1)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens for model response (default: 512)",
    )

    parser.add_argument(
        "--no-quantization", action="store_true", help="Disable model quantization"
    )

    parser.add_argument(
        "--create-sample", action="store_true", help="Create sample dataset and exit"
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available model and task configurations",
    )

    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.config and not args.quick:
        if not all([args.model, args.task, args.dataset]):
            raise ValueError(
                "When not using --config or --quick, --model, --task, and --dataset are required"
            )

        if not Path(args.dataset).exists():
            raise FileNotFoundError(f"Dataset file not found: {args.dataset}")


def run_benchmark_from_args(args: argparse.Namespace) -> None:
    """Run benchmark based on command line arguments."""

    if args.create_sample:
        print("Creating sample dataset...")
        create_sample_dataset("./data/sample_dataset.json")
        print("Sample dataset created at: ./data/sample_dataset.json")
        return

    if args.list_configs:
        available = BenchmarkConfigManager.list_available_configs()
        print("\nAvailable Models:")
        for key, name in available["models"].items():
            print(f"  {key}: {name}")

        print("\nAvailable Tasks:")
        for key, desc in available["tasks"].items():
            print(f"  {key}: {desc}")
        return

    if args.quick:
        print("Running quick benchmark with sample data...")
        # Create sample dataset if it doesn't exist
        sample_path = "./data/sample_dataset.json"
        if not Path(sample_path).exists():
            Path("./data").mkdir(exist_ok=True)
            create_sample_dataset(sample_path)

        config = BenchmarkConfigManager.create_config(
            model_key="qwen2.5-7b",  # Use a reasonable default
            task_key="binary_vulnerability",
            dataset_path=sample_path,
            output_dir="./results/quick_test",
        )

    elif args.config:
        print(f"Loading configuration from: {args.config}")
        config_dict = load_config_from_file(args.config)
        config = BenchmarkConfigManager.create_config(**config_dict)

    else:
        # Build configuration from individual arguments
        config = BenchmarkConfigManager.create_config(
            model_key=args.model,
            task_key=args.task,
            dataset_path=args.dataset,
            output_dir=args.output,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            use_quantization=not args.no_quantization,
        )

    print("\nStarting benchmark:")
    print(f"  Model: {config.model_name}")
    print(f"  Task: {config.task_type.value}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Output: {config.output_dir}")
    print()

    # Run the benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark()

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK COMPLETED")
    print("=" * 50)
    print(f"Total samples: {results['benchmark_info']['total_samples']}")
    print(f"Total time: {results['benchmark_info']['total_time_seconds']:.2f} seconds")

    if "accuracy" in results["metrics"]:
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")

    if "f1_score" in results["metrics"]:
        print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")

    if "true_positives" in results["metrics"]:
        print(f"True Positives: {results['metrics']['true_positives']}")
        print(f"True Negatives: {results['metrics']['true_negatives']}")
        print(f"False Positives: {results['metrics']['false_positives']}")
        print(f"False Negatives: {results['metrics']['false_negatives']}")

    print(f"\nDetailed results saved to: {config.output_dir}")


def main() -> None:
    """Main entry point."""
    try:
        args = parse_arguments()

        if not (args.create_sample or args.list_configs):
            validate_args(args)

        run_benchmark_from_args(args)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
