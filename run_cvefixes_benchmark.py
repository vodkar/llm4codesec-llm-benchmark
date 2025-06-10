#!/usr/bin/env python3
"""
CVEFixes Benchmark Runner

Main entry point for running CVEFixes benchmark experiments.
This script provides a simplified interface to run CVEFixes benchmarks
using predefined configurations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from benchmark.benchmark_framework import BenchmarkConfig, ModelType, TaskType
from entrypoints.run_cvefixes_benchmark import CVEFixesBenchmarkRunner, setup_logging


def load_experiment_config(config_file: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_config_from_experiment(exp_config: dict, 
                                base_config: dict) -> BenchmarkConfig:
    """Create BenchmarkConfig from experiment configuration."""
    
    # Get dataset configuration
    dataset_name = exp_config["dataset"]
    dataset_config = base_config["dataset_configurations"][dataset_name]
    
    # Get model configuration
    model_name = exp_config["model"]
    model_config = base_config["model_configurations"][model_name]
    
    # Get prompt strategy
    prompt_name = exp_config["prompt_strategy"]
    prompt_config = base_config["prompt_strategies"][prompt_name]
    
    # Map task type
    task_type_map = {
        "binary_vulnerability": TaskType.BINARY_VULNERABILITY,
        "multiclass_vulnerability": TaskType.MULTICLASS_VULNERABILITY,
        "binary_cwe_specific": TaskType.BINARY_CWE_SPECIFIC,
    }
    
    # Map model type
    model_type_map = {
        "qwen": ModelType.QWEN,
        "llama": ModelType.LLAMA,
        "deepseek": ModelType.DEEPSEEK,
        "codebert": ModelType.CODEBERT,
        "custom": ModelType.CUSTOM,
    }
    
    task_type = task_type_map[dataset_config["task_type"]]
    model_type = model_type_map[model_config["model_type"]]
    
    # Create output directory
    output_dir = Path(base_config["execution_settings"]["output_base_dir"]) / exp_config["name"]
    
    return BenchmarkConfig(
        model_name=model_config["model_name"],
        model_type=model_type,
        task_type=task_type,
        description=exp_config["description"],
        dataset_path=dataset_config["dataset_path"],
        output_dir=str(output_dir),
        batch_size=base_config["execution_settings"]["batch_size"],
        max_tokens=model_config["max_tokens"],
        temperature=model_config["temperature"],
        use_quantization=model_config["use_quantization"],
        cwe_type=dataset_config.get("cwe_type"),
        system_prompt_template=prompt_config["system_prompt"],
        user_prompt_template=prompt_config["user_prompt"],
    )


def run_single_experiment(exp_name: str, config_file: str, 
                         sample_limit: int = None) -> bool:
    """Run a single experiment."""
    try:
        # Load configuration
        base_config = load_experiment_config(config_file)
        
        # Find experiment configuration
        exp_config = None
        for exp in base_config["experiment_configs"]:
            if exp["name"] == exp_name:
                exp_config = exp
                break
        
        if not exp_config:
            logging.error(f"Experiment '{exp_name}' not found in configuration")
            return False
        
        # Apply sample limit if specified
        if sample_limit:
            exp_config["sample_limit"] = sample_limit
        
        # Create benchmark configuration
        benchmark_config = create_config_from_experiment(exp_config, base_config)
        
        # Validate dataset exists
        if not Path(benchmark_config.dataset_path).exists():
            logging.error(f"Dataset not found: {benchmark_config.dataset_path}")
            logging.info("Please run prepare_cvefixes_datasets.py first to create datasets")
            return False
        
        # Run benchmark
        runner = CVEFixesBenchmarkRunner(benchmark_config)
        results = runner.run_benchmark(exp_config.get("sample_limit"))
        
        logging.info(f"Experiment '{exp_name}' completed successfully")
        return True
        
    except Exception as e:
        logging.exception(f"Error running experiment '{exp_name}': {e}")
        return False


def run_multiple_experiments(exp_names: list, config_file: str,
                           sample_limit: int = None) -> dict:
    """Run multiple experiments."""
    results = {}
    
    for exp_name in exp_names:
        logging.info(f"Starting experiment: {exp_name}")
        success = run_single_experiment(exp_name, config_file, sample_limit)
        results[exp_name] = success
        
        if success:
            logging.info(f"✓ {exp_name} completed successfully")
        else:
            logging.error(f"✗ {exp_name} failed")
    
    return results


def list_available_experiments(config_file: str):
    """List all available experiments in the configuration."""
    config = load_experiment_config(config_file)
    
    print(f"\n{'='*80}")
    print(f"Available CVEFixes Experiments")
    print(f"{'='*80}")
    
    for exp in config["experiment_configs"]:
        dataset_info = config["dataset_configurations"][exp["dataset"]]
        model_info = config["model_configurations"][exp["model"]]
        prompt_info = config["prompt_strategies"][exp["prompt_strategy"]]
        
        print(f"\nName: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Dataset: {dataset_info['description']}")
        print(f"Model: {model_info['description']}")
        print(f"Prompt: {prompt_info['name']}")
        print(f"Dataset Path: {dataset_info['dataset_path']}")
        print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CVEFixes benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available experiments
  python run_cvefixes_benchmark.py --list

  # Run a single experiment
  python run_cvefixes_benchmark.py --experiment cvefixes_binary_basic

  # Run multiple experiments
  python run_cvefixes_benchmark.py --experiment cvefixes_binary_basic cvefixes_method_basic

  # Run with sample limit for testing
  python run_cvefixes_benchmark.py --experiment cvefixes_binary_basic --sample-limit 100

  # Run all experiments
  python run_cvefixes_benchmark.py --all

  # Use custom configuration file
  python run_cvefixes_benchmark.py --config custom_config.json --experiment test_exp
        """
    )
    
    # Experiment selection
    parser.add_argument(
        "--experiment", "--exp",
        nargs="+",
        help="Name(s) of experiment(s) to run"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments defined in configuration"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="cvefixes_experiments_config.json",
        help="Path to experiment configuration file (default: cvefixes_experiments_config.json)"
    )
    
    # Execution options
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit number of samples for testing"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        # Validate configuration file
        if not Path(args.config).exists():
            parser.error(f"Configuration file not found: {args.config}")
        
        # List experiments if requested
        if args.list:
            list_available_experiments(args.config)
            return 0
        
        # Determine experiments to run
        if args.all:
            config = load_experiment_config(args.config)
            exp_names = [exp["name"] for exp in config["experiment_configs"]]
        elif args.experiment:
            exp_names = args.experiment
        else:
            parser.error("Must specify --experiment, --all, or --list")
        
        # Run experiments
        if len(exp_names) == 1:
            success = run_single_experiment(exp_names[0], args.config, args.sample_limit)
            return 0 if success else 1
        else:
            results = run_multiple_experiments(exp_names, args.config, args.sample_limit)
            
            # Print summary
            print(f"\n{'='*80}")
            print(f"Experiment Results Summary")
            print(f"{'='*80}")
            
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            
            for exp_name, success in results.items():
                status = "✓ SUCCESS" if success else "✗ FAILED"
                print(f"{exp_name}: {status}")
            
            print(f"\nTotal: {successful}/{total} experiments successful")
            print(f"{'='*80}")
            
            return 0 if successful == total else 1
        
    except KeyboardInterrupt:
        logging.info("Benchmark execution interrupted by user")
        return 1
    except Exception as e:
        logging.exception(f"Error running benchmark: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
