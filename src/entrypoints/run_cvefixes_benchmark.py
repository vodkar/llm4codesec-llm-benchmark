#!/usr/bin/env python3
"""
CVEFixes Benchmark Runner

A specialized script for running LLM benchmarks on the CVEFixes dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from benchmark.benchmark_framework import BenchmarkConfig, ModelType, TaskType
from benchmark.config_manager import BenchmarkConfigManager
from datasets.loaders.cvefixes_dataset_loader import CVEFixesJSONDatasetLoader


class CVEFixesBenchmarkRunner:
    """Custom benchmark runner for CVEFixes datasets."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = CVEFixesJSONDatasetLoader()

    def run_benchmark(self, sample_limit: Optional[int] = None):
        """Run benchmark with CVEFixes-specific dataset loading."""
        import logging
        import time
        from pathlib import Path

        from benchmark.benchmark_framework import (
            HuggingFaceLLM,
            MetricsCalculator,
            PredictionResult,
            PromptGenerator,
            ResponseParser,
        )

        logging.info("Starting CVEFixes benchmark execution")
        start_time = time.time()

        try:
            # Load dataset using CVEFixes loader
            logging.info(f"Loading CVEFixes dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)

            # Apply sample limit if specified
            if sample_limit and sample_limit < len(samples):
                samples = samples[:sample_limit]
                logging.info(f"Limited to {sample_limit} samples")

            logging.info(f"Loaded {len(samples)} samples")

            # Initialize components
            llm = HuggingFaceLLM(self.config)
            prompt_generator = PromptGenerator()
            response_parser = ResponseParser(self.config.task_type)
            metrics_calculator = MetricsCalculator()

            # Create output directory
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

            # Run predictions
            predictions = []
            system_prompt = (
                self.config.system_prompt_template
                or prompt_generator.get_system_prompt(
                    self.config.task_type, self.config.cwe_type
                )
            )

            for i, sample in enumerate(samples):
                logging.info(f"Processing sample {i + 1}/{len(samples)}: {sample.id}")

                user_prompt = (
                    self.config.user_prompt_template.format(code=sample.code)
                    if self.config.user_prompt_template
                    else prompt_generator.get_user_prompt(
                        self.config.task_type, sample.code, self.config.cwe_type
                    )
                )

                # Generate response
                start_time_sample = time.time()
                response_text, tokens_used = llm.generate_response(
                    system_prompt, user_prompt
                )
                processing_time = time.time() - start_time_sample

                # Parse response
                predicted_label = response_parser.parse_response(response_text)

                prediction = PredictionResult(
                    sample_id=sample.id,
                    predicted_label=predicted_label,
                    true_label=sample.label,
                    confidence=None,
                    response_text=response_text,
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                )

                predictions.append(prediction)

                if (i + 1) % 10 == 0:
                    logging.info(f"Completed {i + 1}/{len(samples)} predictions")

            # Calculate metrics
            if self.config.task_type in [
                TaskType.BINARY_VULNERABILITY,
                TaskType.BINARY_CWE_SPECIFIC,
            ]:
                metrics = metrics_calculator.calculate_binary_metrics(predictions)
            else:
                metrics = metrics_calculator.calculate_multiclass_metrics(predictions)

            # Create results
            results = {
                "config": {
                    "model_name": self.config.model_name,
                    "task_type": self.config.task_type.value,
                    "dataset_path": str(self.config.dataset_path),
                    "cwe_type": self.config.cwe_type,
                    "total_samples": len(samples),
                    "sample_limit": sample_limit,
                },
                "metrics": metrics,
                "predictions": [
                    {
                        "sample_id": p.sample_id,
                        "predicted_label": p.predicted_label,
                        "true_label": p.true_label,
                        "confidence": p.confidence,
                        "response_text": p.response_text,
                        "processing_time": p.processing_time,
                        "tokens_used": p.tokens_used,
                    }
                    for p in predictions
                ],
                "execution_time": time.time() - start_time,
            }

            # Save results
            timestamp = int(time.time())
            model_name_safe = self.config.model_name.replace("/", "_")
            task_name = self.config.task_type.value

            if self.config.cwe_type:
                output_filename = f"{model_name_safe}_{task_name}_{self.config.cwe_type.lower()}_{timestamp}.json"
            else:
                output_filename = f"{model_name_safe}_{task_name}_{timestamp}.json"

            output_path = Path(self.config.output_dir) / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            logging.info(f"Results saved to: {output_path}")
            logging.info(
                f"Total execution time: {time.time() - start_time:.2f} seconds"
            )

            # Print summary
            print(f"\n{'=' * 60}")
            print("CVEFixes Benchmark Results Summary")
            print(f"{'=' * 60}")
            print(f"Model: {self.config.model_name}")
            print(f"Task: {self.config.task_type.value}")
            if self.config.cwe_type:
                print(f"CWE Type: {self.config.cwe_type}")
            print(f"Samples: {len(samples)}")
            print(f"Execution Time: {time.time() - start_time:.2f}s")
            print("\nMetrics:")

            if self.config.task_type in [
                TaskType.BINARY_VULNERABILITY,
                TaskType.BINARY_CWE_SPECIFIC,
            ]:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
            else:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")

            print(f"\nResults saved to: {output_path}")
            print(f"{'=' * 60}")

            return results

        except Exception as e:
            logging.exception(f"Error during benchmark execution: {e}")
            raise


def create_cvefixes_config_from_args(args) -> BenchmarkConfig:
    """Create a BenchmarkConfig from command line arguments for CVEFixes."""

    # Map string task types to enum
    task_type_map = {
        "binary": TaskType.BINARY_VULNERABILITY,
        "multiclass": TaskType.MULTICLASS_VULNERABILITY,
        "cwe_specific": TaskType.BINARY_CWE_SPECIFIC,
    }

    # Map string model types to enum
    model_type_map = {
        "llama": ModelType.LLAMA,
        "qwen": ModelType.QWEN,
        "deepseek": ModelType.DEEPSEEK,
        "codebert": ModelType.CODEBERT,
        "custom": ModelType.CUSTOM,
    }

    task_type = task_type_map.get(args.task_type, TaskType.BINARY_VULNERABILITY)
    model_type = model_type_map.get(args.model_type, ModelType.QWEN)

    return BenchmarkConfig(
        model_name=args.model_name,
        model_type=model_type,
        task_type=task_type,
        description=f"CVEFixes {args.task_type} vulnerability detection",
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_quantization=args.use_quantization,
        cwe_type=args.cwe_type,
        system_prompt_template=args.system_prompt,
        user_prompt_template=args.user_prompt,
    )


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("cvefixes_benchmark.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main entry point for CVEFixes benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run LLM benchmark on CVEFixes dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run binary vulnerability detection
  python src/entrypoints/run_cvefixes_benchmark.py--dataset-path datasets_processed/cvefixes/cvefixes_binary_c.json

  # Run CWE-specific detection
  python src/entrypoints/run_cvefixes_benchmark.py--task-type cwe_specific --cwe-type CWE-119 --dataset-path datasets_processed/cvefixes/cvefixes_cwe_119.json

  # Use custom model
  python src/entrypoints/run_cvefixes_benchmark.py--model-type custom --model-name "microsoft/codebert-base" --dataset-path datasets_processed/cvefixes/cvefixes_binary_c.json

  # Limit samples for testing
  python src/entrypoints/run_cvefixes_benchmark.py--dataset-path datasets_processed/cvefixes/cvefixes_binary_c.json --sample-limit 100
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to CVEFixes dataset JSON file",
    )

    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "qwen", "deepseek", "codebert", "custom"],
        default="qwen",
        help="Type of model to use (default: qwen)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Specific model name (overrides model-type if provided)",
    )

    # Task configuration
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["binary", "multiclass", "cwe_specific"],
        default="binary",
        help="Type of vulnerability detection task (default: binary)",
    )

    parser.add_argument(
        "--cwe-type",
        type=str,
        help="Specific CWE type for cwe_specific task (e.g., CWE-119)",
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)",
    )

    # System configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/cvefixes_experiments",
        help="Output directory for results (default: results/cvefixes_experiments)",
    )

    parser.add_argument(
        "--use-quantization",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)",
    )

    parser.add_argument(
        "--no-quantization", action="store_true", help="Disable quantization"
    )

    # Prompt customization
    parser.add_argument(
        "--system-prompt", type=str, help="Custom system prompt template"
    )

    parser.add_argument(
        "--user-prompt",
        type=str,
        help="Custom user prompt template (use {code} placeholder)",
    )

    # Execution options
    parser.add_argument(
        "--sample-limit", type=int, help="Limit number of samples for testing"
    )

    parser.add_argument(
        "--config-file", type=str, help="Load configuration from JSON file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)

    try:
        # Handle quantization flags
        if args.no_quantization:
            args.use_quantization = False

        # Load configuration from file if provided
        if args.config_file:
            config_manager = BenchmarkConfigManager()
            config = config_manager.load_config(args.config_file)

            # Override with command line arguments if provided
            if args.dataset_path:
                config.dataset_path = args.dataset_path
            if args.model_name:
                config.model_name = args.model_name
            if args.output_dir != "results/cvefixes_experiments":
                config.output_dir = args.output_dir
        else:
            # Create configuration from arguments
            config = create_cvefixes_config_from_args(args)

        # Set model name based on type if not specified
        if not args.model_name:
            model_names = {
                "llama": "meta-llama/Llama-2-7b-chat-hf",
                "qwen": "Qwen/Qwen2.5-7B-Instruct",
                "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
                "codebert": "microsoft/codebert-base",
            }
            config.model_name = model_names.get(
                args.model_type, "Qwen/Qwen2.5-7B-Instruct"
            )

        # Validate CWE type for cwe_specific task
        if args.task_type == "cwe_specific" and not args.cwe_type:
            parser.error("--cwe-type is required when --task-type is cwe_specific")

        # Validate dataset path
        if not Path(args.dataset_path).exists():
            parser.error(f"Dataset file not found: {args.dataset_path}")

        # Create and run benchmark
        runner = CVEFixesBenchmarkRunner(config)
        results = runner.run_benchmark(args.sample_limit)

        return 0

    except KeyboardInterrupt:
        logging.info("Benchmark execution interrupted by user")
        return 1
    except Exception as e:
        logging.exception(f"Error running benchmark: {e}")
        return 1


if __name__ == "__main__":
    main()
