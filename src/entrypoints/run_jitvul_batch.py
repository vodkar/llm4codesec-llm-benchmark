#!/usr/bin/env python3
"""
Batch experiment runner for JitVul benchmark evaluations.

This script allows running multiple JitVul experiments in batch mode,
supporting different model comparisons, task types, and ablation studies.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.benchmark_framework import BenchmarkConfig
from entrypoints.run_jitvul_benchmark import JitVulBenchmarkRunner


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


class JitVulBatchRunner:
    """Batch experiment runner for JitVul benchmarks."""
    
    def __init__(self, config_file: str, output_base_dir: str = "results/jitvul_batch"):
        """
        Initialize batch runner.
        
        Args:
            config_file: Path to experiment configuration file
            output_base_dir: Base directory for batch results
        """
        self.config_file = config_file
        self.output_base_dir = output_base_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Create output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Track results
        self.batch_results: Dict[str, Any] = {
            "start_time": None,
            "end_time": None,
            "total_duration": None,
            "experiments": {},
            "summary": {}
        }
    
    def run_single_experiment(self, experiment_name: str, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            experiment_name: Name of the experiment
            experiment_config: Experiment configuration
            
        Returns:
            Dictionary with experiment results and metadata
        """
        self.logger.info(f"Starting experiment: {experiment_name}")
        start_time = time.time()
        
        try:
            # Create experiment-specific output directory
            exp_output_dir = os.path.join(self.output_base_dir, experiment_name)
            os.makedirs(exp_output_dir, exist_ok=True)
            
            # Update output directory in config
            exp_config = experiment_config.copy()
            exp_config["config"]["output_dir"] = exp_output_dir
            
            # Create BenchmarkConfig from experiment config
            config = BenchmarkConfig(**exp_config["config"])
            
            # Initialize and run benchmark
            runner = JitVulBenchmarkRunner(config)
            results = runner.run_benchmark()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Prepare result summary
            result_summary: Dict[str, Any] = {
                "status": "success",
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
                "description": experiment_config.get("description", ""),
                "config": exp_config["config"],
                "results": results,
                "output_dir": exp_output_dir
            }
            
            # Save individual experiment results
            result_file = os.path.join(exp_output_dir, f"{experiment_name}_summary.json")
            with open(result_file, 'w') as f:
                json.dump(result_summary, f, indent=2, default=str)
            
            self.logger.info(f"Completed experiment: {experiment_name} (duration: {duration:.2f}s)")
            return result_summary
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            error_summary: Dict[str, Any] = {
                "status": "error",
                "duration": duration,
                "start_time": start_time,
                "end_time": end_time,
                "description": experiment_config.get("description", ""),
                "config": experiment_config["config"],
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            self.logger.error(f"Failed experiment: {experiment_name} - {str(e)}")
            return error_summary
    
    def run_experiment_list(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Run a list of experiments.
        
        Args:
            experiment_names: List of experiment names to run
            
        Returns:
            Dictionary with results for all experiments
        """
        results: Dict[str, Any] = {}
        
        for exp_name in experiment_names:
            if exp_name not in [exp["name"] for exp in self.config["experiments"]]:
                self.logger.error(f"Experiment '{exp_name}' not found in configuration")
                results[exp_name] = {
                    "status": "error",
                    "error": f"Experiment '{exp_name}' not found in configuration"
                }
                continue
            
            # Find experiment config
            exp_config = None
            for exp in self.config["experiments"]:
                if exp["name"] == exp_name:
                    exp_config = exp
                    break
            
            if exp_config is None:
                self.logger.error(f"Configuration for experiment '{exp_name}' not found")
                results[exp_name] = {
                    "status": "error",
                    "error": f"Configuration for experiment '{exp_name}' not found"
                }
                continue
            
            results[exp_name] = self.run_single_experiment(exp_name, exp_config)
        
        return results
    
    def run_batch_config(self, batch_name: str) -> Dict[str, Any]:
        """
        Run a predefined batch configuration.
        
        Args:
            batch_name: Name of the batch configuration
            
        Returns:
            Dictionary with results for all experiments in the batch
        """
        # Find batch config
        batch_config = None
        for batch in self.config.get("batch_configs", []):
            if batch["name"] == batch_name:
                batch_config = batch
                break
        
        if batch_config is None:
            raise ValueError(f"Batch configuration '{batch_name}' not found")
        
        self.logger.info(f"Running batch: {batch_name}")
        self.logger.info(f"Description: {batch_config.get('description', 'N/A')}")
        self.logger.info(f"Experiments: {batch_config['experiments']}")
        
        return self.run_experiment_list(batch_config["experiments"])
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all experiments defined in the configuration.
        
        Returns:
            Dictionary with results for all experiments
        """
        experiment_names = [exp["name"] for exp in self.config["experiments"]]
        return self.run_experiment_list(experiment_names)
    
    def generate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for batch results.
        
        Args:
            results: Results from batch execution
            
        Returns:
            Summary statistics
        """
        summary: Dict[str, Any] = {
            "total_experiments": len(results),
            "successful_experiments": len([r for r in results.values() if r.get("status") == "success"]),
            "failed_experiments": len([r for r in results.values() if r.get("status") == "error"]),
            "total_duration": sum([r.get("duration", 0) for r in results.values()]),
            "average_duration": sum([r.get("duration", 0) for r in results.values()]) / len(results) if results else 0,
            "experiment_details": {}
        }
        
        # Add detailed metrics for successful experiments
        for exp_name, result in results.items():
            if result.get("status") == "success" and "results" in result:
                exp_results = result["results"]
                summary["experiment_details"][exp_name] = {
                    "duration": result.get("duration", 0),
                    "metrics": exp_results.get("metrics", {}),
                    "num_samples": exp_results.get("dataset_info", {}).get("total_samples", 0),
                    "task_type": result.get("config", {}).get("dataset_config", {}).get("task_type", "unknown")
                }
        
        return summary
    
    def save_batch_results(self, results: Dict[str, Any], batch_name: Optional[str] = None) -> str:
        """
        Save complete batch results to file.
        
        Args:
            results: Results from batch execution
            batch_name: Name of the batch (for filename)
            
        Returns:
            Path to saved results file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if batch_name:
            filename = f"batch_{batch_name}_{timestamp}.json"
        else:
            filename = f"batch_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_base_dir, filename)
        
        # Update batch results
        self.batch_results["experiments"] = results
        self.batch_results["summary"] = self.generate_batch_summary(results)
        
        with open(filepath, 'w') as f:
            json.dump(self.batch_results, f, indent=2, default=str)
        
        self.logger.info(f"Batch results saved to: {filepath}")
        return filepath
    
    def run(self, batch_name: Optional[str] = None, experiment_names: Optional[List[str]] = None) -> str:
        """
        Run experiments based on provided parameters.
        
        Args:
            batch_name: Name of batch configuration to run
            experiment_names: List of specific experiments to run
            
        Returns:
            Path to saved results file
        """
        self.batch_results["start_time"] = datetime.now().isoformat()
        start_time = time.time()
        
        try:
            if batch_name:
                results = self.run_batch_config(batch_name)
                save_name = batch_name
            elif experiment_names:
                results = self.run_experiment_list(experiment_names)
                save_name = "custom"
            else:
                results = self.run_all_experiments()
                save_name = "all"
            
            end_time = time.time()
            self.batch_results["end_time"] = datetime.now().isoformat()
            self.batch_results["total_duration"] = end_time - start_time
            
            # Save and return results
            return self.save_batch_results(results, save_name)
            
        except Exception as e:
            self.logger.error(f"Batch execution failed: {str(e)}")
            raise


def main():
    """Main entry point for batch experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run batch JitVul benchmark experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_jitvul_batch.py --config benchmarks/JitVul/jitvul_experiments_config.json
  
  # Run specific batch configuration
  python run_jitvul_batch.py --config benchmarks/JitVul/jitvul_experiments_config.json --batch jitvul_baseline_comparison
  
  # Run specific experiments
  python run_jitvul_batch.py --config benchmarks/JitVul/jitvul_experiments_config.json --experiments jitvul_binary_baseline jitvul_multiclass_baseline
  
  # Run with custom output directory and logging
  python run_jitvul_batch.py --config benchmarks/JitVul/jitvul_experiments_config.json --batch jitvul_full_evaluation --output results/my_batch --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration JSON file"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        help="Name of batch configuration to run"
    )
    
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="List of specific experiment names to run"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/jitvul_batch",
        help="Output directory for batch results"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch and args.experiments:
        parser.error("Cannot specify both --batch and --experiments")
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")
    
    try:
        # Initialize and run batch runner
        runner = JitVulBatchRunner(args.config, args.output)
        
        logger.info("Starting JitVul batch experiment runner")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Output directory: {args.output}")
        
        results_file = runner.run(
            batch_name=args.batch,
            experiment_names=args.experiments
        )
        
        logger.info(f"Batch execution completed successfully")
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        summary = runner.batch_results["summary"]
        logger.info(f"Summary: {summary['successful_experiments']}/{summary['total_experiments']} experiments successful")
        logger.info(f"Total duration: {summary['total_duration']:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Batch execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()