#!/usr/bin/env python3
"""
CASTLE Benchmark Runner

A specialized script for running LLM benchmarks on the CASTLE dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark_framework import BenchmarkConfig, ModelType, TaskType
from castle_dataset_loader import CastleDatasetLoaderFramework
from config_manager import BenchmarkConfigManager


class CastleBenchmarkRunner:
    """Custom benchmark runner for CASTLE datasets."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = CastleDatasetLoaderFramework()
        
    def run_benchmark(self, sample_limit: Optional[int] = None):
        """Run benchmark with CASTLE-specific dataset loading."""
        import logging
        import time
        from pathlib import Path

        from benchmark_framework import (HuggingFaceLLM, MetricsCalculator, PredictionResult, PromptGenerator,
                                         ResponseParser)
        
        logging.info("Starting CASTLE benchmark execution")
        start_time = time.time()
        
        try:
            # Load dataset using CASTLE loader
            logging.info(f"Loading CASTLE dataset from: {self.config.dataset_path}")
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
            system_prompt = self.config.system_prompt_template or prompt_generator.get_system_prompt(
                self.config.task_type, self.config.cwe_type
            )
            
            for i, sample in enumerate(samples):
                logging.info(f"Processing sample {i + 1}/{len(samples)}: {sample.id}")
                
                user_prompt = self.config.user_prompt_template.format(code=sample.code) if self.config.user_prompt_template else prompt_generator.get_user_prompt(
                    self.config.task_type, sample.code, self.config.cwe_type
                )
                
                # Generate response
                start_time_sample = time.time()
                response_text, tokens_used = llm.generate_response(system_prompt, user_prompt)
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
            if self.config.task_type in [TaskType.BINARY_VULNERABILITY, TaskType.BINARY_CWE_SPECIFIC]:
                metrics = metrics_calculator.calculate_binary_metrics(predictions)
            else:
                metrics = metrics_calculator.calculate_multiclass_metrics(predictions)
            
            # Generate results
            total_time = time.time() - start_time
            results = {
                "accuracy": metrics.get("accuracy", 0.0),
                "metrics": metrics,
                "total_samples": len(samples),
                "total_time": total_time,
                "predictions": predictions
            }
            
            # Clean up
            llm.cleanup()
            
            logging.info("CASTLE benchmark completed successfully")
            return results
            
        except Exception as e:
            logging.exception(f"CASTLE benchmark failed: {e}")
            raise


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_castle_config(config_path: str) -> Dict[str, Any]:
    """Load CASTLE experiment configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_benchmark_config(
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any], 
    prompt_config: Dict[str, Any],
    output_dir: str
) -> BenchmarkConfig:
    """
    Create a BenchmarkConfig from experiment configuration.
    
    Args:
        model_config: Model configuration dict
        dataset_config: Dataset configuration dict
        prompt_config: Prompt configuration dict
        output_dir: Output directory path
        
    Returns:
        BenchmarkConfig: Complete benchmark configuration
    """
    # Map string model types to enum
    model_type_map = {
        "LLAMA": ModelType.LLAMA,
        "QWEN": ModelType.QWEN, 
        "DEEPSEEK": ModelType.DEEPSEEK,
        "CODEBERT": ModelType.CODEBERT
    }
    
    # Map string task types to enum
    task_type_map = {
        "binary_vulnerability": TaskType.BINARY_VULNERABILITY,
        "binary_cwe_specific": TaskType.BINARY_CWE_SPECIFIC,
        "multiclass_vulnerability": TaskType.MULTICLASS_VULNERABILITY
    }
    
    return BenchmarkConfig(
        model_name=model_config["model_name"],
        model_type=model_type_map[model_config["model_type"]],
        task_type=task_type_map[dataset_config["task_type"]],
        description=f"{prompt_config['name']} - {dataset_config['description']}",
        dataset_path=dataset_config["dataset_path"],
        output_dir=output_dir,
        batch_size=model_config.get("batch_size", 1),
        max_tokens=model_config.get("max_tokens", 512),
        temperature=model_config.get("temperature", 0.1),
        use_quantization=model_config.get("use_quantization", True),
        cwe_type=dataset_config.get("cwe_type"),
        system_prompt_template=prompt_config["system_prompt"],
        user_prompt_template=prompt_config["user_prompt"]
    )


def run_single_experiment(
    model_key: str,
    dataset_key: str,
    prompt_key: str,
    castle_config: Dict[str, Any],
    sample_limit: Optional[int] = None,
    output_base_dir: str = "results/castle_experiments"
) -> Dict[str, Any]:
    """
    Run a single benchmark experiment.
    
    Args:
        model_key: Model configuration key
        dataset_key: Dataset configuration key  
        prompt_key: Prompt configuration key
        castle_config: CASTLE experiment configuration
        sample_limit: Limit number of samples (for testing)
        output_base_dir: Base output directory
        
    Returns:
        Dict containing experiment results
    """
    logger = logging.getLogger(__name__)
    
    # Get configurations
    model_config = castle_config["model_configurations"][model_key]
    dataset_config = castle_config["dataset_configurations"][dataset_key] 
    prompt_config = castle_config["prompt_strategies"][prompt_key]
    
    # Create output directory
    experiment_name = f"{model_key}_{dataset_key}_{prompt_key}"
    output_dir = Path(output_base_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Dataset: {dataset_config['description']}")
    logger.info(f"Prompt: {prompt_config['name']}")
    
    # Create benchmark configuration
    config = create_benchmark_config(
        model_config, dataset_config, prompt_config, str(output_dir)
    )
    
    # Initialize and run benchmark  
    runner = CastleBenchmarkRunner(config)
    
    try:
        results = runner.run_benchmark(sample_limit=sample_limit)
        
        logger.info(f"Experiment completed successfully")
        logger.info(f"Accuracy: {results.get('accuracy', 'N/A'):.3f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return {
            "experiment_name": experiment_name,
            "status": "success",
            "results": results,
            "output_dir": str(output_dir)
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return {
            "experiment_name": experiment_name,
            "status": "failed",
            "error": str(e),
            "output_dir": str(output_dir)
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run CASTLE benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        default="castle_experiments_config.json",
        help="Path to CASTLE experiments configuration file"
    )
    
    parser.add_argument(
        "--model",
        choices=["llama2-7b", "qwen2.5-7b", "deepseek-coder"],
        help="Model to use for benchmark"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["binary_all", "multiclass_all", "cwe_125", "cwe_190", "cwe_476", "cwe_787"],
        help="Dataset configuration to use"
    )
    
    parser.add_argument(
        "--prompt",
        choices=["basic_security", "detailed_analysis", "cwe_focused", "context_aware", "step_by_step"],
        help="Prompt strategy to use"
    )
    
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="Limit number of samples for testing (default: use all samples)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results/castle_experiments", 
        help="Base output directory for results"
    )
    
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup datasets, don't run experiments"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        
        castle_config = load_castle_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Setup datasets if requested
        if args.setup_only:
            logger.info("Setting up CASTLE datasets...")
            from setup_castle_dataset import create_castle_datasets
            create_castle_datasets()
            logger.info("Dataset setup completed")
            return
        
        # Validate required arguments
        if not all([args.model, args.dataset, args.prompt]):
            logger.error("Model, dataset, and prompt must be specified")
            logger.error("Use --help for usage information")
            sys.exit(1)
        
        # Check if dataset exists
        dataset_config = castle_config["dataset_configurations"][args.dataset]
        dataset_path = Path(dataset_config["dataset_path"])
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            logger.error("Run with --setup-only first to create processed datasets")
            sys.exit(1)
        
        # Run experiment
        results = run_single_experiment(
            model_key=args.model,
            dataset_key=args.dataset,
            prompt_key=args.prompt,
            castle_config=castle_config,
            sample_limit=args.sample_limit,
            output_base_dir=args.output_dir
        )
        
        if results["status"] == "success":
            logger.info("Benchmark completed successfully!")
        else:
            logger.error(f"Benchmark failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()