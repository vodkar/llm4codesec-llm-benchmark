#!/usr/bin/env python3
"""
Example usage of the LLM Code Security Benchmark Framework.

This script demonstrates how to:
1. Create sample datasets
2. Configure and run benchmarks
3. Analyze results
"""

import json
from pathlib import Path

from benchmark_framework import BenchmarkRunner
from config_manager import BenchmarkConfigManager, create_sample_dataset
from data_utils import DatasetAnalyzer


def example_1_quick_benchmark():
    """Example 1: Run a quick benchmark with sample data."""
    print("=" * 60)
    print("EXAMPLE 1: Quick Benchmark with Sample Data")
    print("=" * 60)

    # Create sample dataset
    sample_path = "./examples/sample_dataset.json"
    Path("./examples").mkdir(exist_ok=True)
    create_sample_dataset(sample_path)
    print(f"Created sample dataset: {sample_path}")

    # Create configuration
    config = BenchmarkConfigManager.create_config(
        model_key="qwen2.5-7b",
        task_key="binary_vulnerability",
        dataset_path=sample_path,
        output_dir="./examples/results/quick_test",
    )

    print(f"Running benchmark: {config.model_name} on {config.task_type.value}")

    # Run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark()

    # Print results
    print("\nResults:")
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")
    print(f"  Processing time: {results['benchmark_info']['total_time_seconds']:.2f}s")


def example_2_multiple_models():
    """Example 2: Compare multiple models on the same task."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Model Comparison")
    print("=" * 60)

    # Create larger sample dataset
    extended_samples = [
        {
            "id": f"vuln_{i:03d}",
            "code": f"""
def process_user_input_{i}(user_input):
    query = f"SELECT * FROM table WHERE field = {{user_input}}"
    return execute_query(query)
""",
            "label": 1,
            "cwe_type": "CWE-89",
            "metadata": {"language": "python", "synthetic": True},
        }
        for i in range(5)
    ] + [
        {
            "id": f"safe_{i:03d}",
            "code": f"""
def process_user_input_{i}(user_input):
    query = "SELECT * FROM table WHERE field = ?"
    return execute_query(query, (user_input,))
""",
            "label": 0,
            "cwe_type": None,
            "metadata": {"language": "python", "synthetic": True},
        }
        for i in range(5)
    ]

    dataset_path = "./examples/extended_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(extended_samples, f, indent=2)

    print(f"Created extended dataset with {len(extended_samples)} samples")

    # Test multiple models (use smaller/faster models for demo)
    models_to_test = ["qwen2.5-7b"]  # Add more models as available

    results_summary = []

    for model_key in models_to_test:
        print(f"\nTesting model: {model_key}")

        config = BenchmarkConfigManager.create_config(
            model_key=model_key,
            task_key="binary_vulnerability",
            dataset_path=dataset_path,
            output_dir=f"./examples/results/{model_key}_comparison",
        )

        try:
            runner = BenchmarkRunner(config)
            results = runner.run_benchmark()

            summary = {
                "model": model_key,
                "accuracy": results["metrics"]["accuracy"],
                "f1_score": results["metrics"]["f1_score"],
                "processing_time": results["benchmark_info"]["total_time_seconds"],
            }
            results_summary.append(summary)

            print(f"  Accuracy: {summary['accuracy']:.4f}")
            print(f"  F1-Score: {summary['f1_score']:.4f}")

        except Exception as e:
            print(f"  Error with {model_key}: {e}")
            continue

    # Print comparison
    if results_summary:
        print("\nModel Comparison Summary:")
        print("-" * 50)
        for result in results_summary:
            print(
                f"{result['model']:15} | Acc: {result['accuracy']:.4f} | F1: {result['f1_score']:.4f}"
            )


def example_3_different_tasks():
    """Example 3: Test different task types."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Task Types")
    print("=" * 60)

    # Create task-specific datasets
    tasks_data = {
        "binary_vulnerability": [
            {
                "id": "bin_001",
                "code": "strcpy(dest, user_input);",
                "label": 1,
                "metadata": {"language": "c"},
            },
            {
                "id": "bin_002",
                "code": "strncpy(dest, user_input, sizeof(dest)-1);",
                "label": 0,
                "metadata": {"language": "c"},
            },
        ],
        "cwe79_detection": [
            {
                "id": "xss_001",
                "code": "return f'<h1>Hello {user_name}</h1>'",
                "label": 1,
                "cwe_type": "CWE-79",
                "metadata": {"language": "python"},
            },
            {
                "id": "xss_002",
                "code": "return f'<h1>Hello {html.escape(user_name)}</h1>'",
                "label": 0,
                "metadata": {"language": "python"},
            },
        ],
        "multiclass_vulnerability": [
            {
                "id": "multi_001",
                "code": "strcpy(buffer, input);",
                "label": "CWE-120",
                "cwe_type": "CWE-120",
                "metadata": {"language": "c"},
            },
            {
                "id": "multi_002",
                "code": "return f'<div>{user_data}</div>'",
                "label": "CWE-79",
                "cwe_type": "CWE-79",
                "metadata": {"language": "python"},
            },
            {
                "id": "multi_003",
                "code": "strncpy(buffer, input, sizeof(buffer));",
                "label": "SAFE",
                "metadata": {"language": "c"},
            },
        ],
    }

    for task_key, samples in tasks_data.items():
        print(f"\nTesting task: {task_key}")

        # Save task dataset
        task_dataset_path = f"./examples/{task_key}_dataset.json"
        with open(task_dataset_path, "w") as f:
            json.dump(samples, f, indent=2)

        # Run benchmark
        config = BenchmarkConfigManager.create_config(
            model_key="qwen2.5-7b",
            task_key=task_key,
            dataset_path=task_dataset_path,
            output_dir=f"./examples/results/{task_key}_test",
        )

        try:
            runner = BenchmarkRunner(config)
            results = runner.run_benchmark()

            print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            if "f1_score" in results["metrics"]:
                print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")


def example_4_dataset_analysis():
    """Example 4: Analyze dataset characteristics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Dataset Analysis")
    print("=" * 60)

    # Create a more complex dataset for analysis
    complex_dataset = []

    # Add various vulnerability types
    vulnerabilities = [
        ("CWE-79", "return f'<h1>{user_input}</h1>'", "python"),
        ("CWE-89", "f'SELECT * FROM users WHERE id = {user_id}'", "python"),
        ("CWE-120", "strcpy(dest, src);", "c"),
        ("CWE-190", "int result = a + b;", "c"),
        ("CWE-476", "return obj->field;", "c"),
    ]

    for i, (cwe, code_template, lang) in enumerate(vulnerabilities):
        # Add vulnerable samples
        for j in range(3):
            complex_dataset.append(
                {
                    "id": f"{cwe.lower()}_{j:02d}",
                    "code": code_template,
                    "label": cwe,
                    "cwe_type": cwe,
                    "severity": ["low", "medium", "high"][j % 3],
                    "metadata": {"language": lang, "category": "vulnerability"},
                }
            )

    # Add safe samples
    safe_codes = [
        ("html.escape(user_input)", "python"),
        ("execute_query('SELECT * FROM users WHERE id = ?', (user_id,))", "python"),
        ("strncpy(dest, src, sizeof(dest));", "c"),
    ]

    for i, (code, lang) in enumerate(safe_codes):
        for j in range(2):
            complex_dataset.append(
                {
                    "id": f"safe_{i:02d}_{j:02d}",
                    "code": code,
                    "label": "SAFE",
                    "cwe_type": None,
                    "severity": None,
                    "metadata": {"language": lang, "category": "safe"},
                }
            )

    # Save dataset
    analysis_dataset_path = "./examples/analysis_dataset.json"
    with open(analysis_dataset_path, "w") as f:
        json.dump(complex_dataset, f, indent=2)

    print(f"Created complex dataset with {len(complex_dataset)} samples")

    # Analyze dataset
    analyzer = DatasetAnalyzer(analysis_dataset_path)
    report = analyzer.generate_report("./examples/dataset_analysis_report.json")

    # Print analysis summary
    print("\nDataset Analysis:")
    print(f"  Total samples: {report['basic_stats']['total_samples']}")
    print(f"  Unique labels: {report['basic_stats']['unique_labels']}")
    print(f"  Label distribution: {report['basic_stats']['label_distribution']}")
    print(f"  CWE distribution: {report['basic_stats'].get('cwe_distribution', 'N/A')}")
    print(
        f"  Language distribution: {report['basic_stats'].get('language_distribution', 'N/A')}"
    )

    balance_info = report["class_balance"]
    print(f"  Class balance ratio: {balance_info.get('imbalance_ratio', 'N/A'):.2f}")

    if report["duplicates"]:
        print(f"  Found {len(report['duplicates'])} duplicate groups")


def main():
    """Run all examples."""
    print("LLM Code Security Benchmark Framework - Examples")
    print("This script demonstrates various usage patterns of the framework.")
    print("\nNote: These examples use actual models and may take time to run.")
    print(
        "Make sure you have the required dependencies installed and sufficient resources."
    )

    # Create examples directory
    Path("./examples").mkdir(exist_ok=True)
    Path("./examples/results").mkdir(exist_ok=True)

    try:
        # Run examples
        example_1_quick_benchmark()
        example_2_multiple_models()
        example_3_different_tasks()
        example_4_dataset_analysis()

        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Check the ./examples/ directory for generated datasets and results.")

    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This might be due to:")
        print("1. Missing model dependencies")
        print("2. Insufficient system resources")
        print("3. Network connectivity issues")
        print("\nCheck the requirements and try running individual examples.")


if __name__ == "__main__":
    main()
