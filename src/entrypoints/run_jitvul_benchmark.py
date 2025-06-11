#!/usr/bin/env python3
"""
JitVul Benchmark Runner

A specialized script for running LLM benchmarks on the JitVul dataset with
flexible configuration options for different vulnerability detection tasks.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.benchmark_framework import BenchmarkConfig, ModelType, TaskType
from datasets.loaders.jitvul_dataset_loader import JitVulDatasetLoaderFramework


class JitVulBenchmarkRunner:
    """Custom benchmark runner for JitVul datasets."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = JitVulDatasetLoaderFramework()
    
    def run_benchmark(self, sample_limit: Optional[int] = None) -> Dict[str, Any]:
        """Run benchmark with JitVul-specific dataset loading."""
        from benchmark.benchmark_framework import (
            HuggingFaceLLM,
            MetricsCalculator,
            PredictionResult,
            PromptGenerator,
            ResponseParser,
            TaskType,
        )
        
        logging.info("Starting JitVul benchmark execution")
        start_time = time.time()
        
        try:
            # Load dataset using JitVul loader
            logging.info(f"Loading JitVul dataset from: {self.config.dataset_path}")
            
            # Determine task type and prepare loading parameters
            if self.config.task_type == TaskType.BINARY_VULNERABILITY:
                task_type = "binary"
            elif self.config.task_type == TaskType.BINARY_CWE_SPECIFIC:
                task_type = "cwe_specific"
                if not self.config.cwe_type:
                    raise ValueError("CWE type must be specified for CWE-specific tasks")
            elif self.config.task_type == TaskType.MULTICLASS_VULNERABILITY:
                task_type = "multiclass"
            else:
                task_type = "binary"  # default
            
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)
            
            logging.info(f"Loaded {len(samples)} samples")
            
            # Initialize components
            llm = HuggingFaceLLM(self.config)
            prompt_generator = PromptGenerator()
            response_parser = ResponseParser(self.config.task_type)
            metrics_calculator = MetricsCalculator()
            
            # Create output directory
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            
            # Run predictions
            predictions: List[PredictionResult] = []
            system_prompt = self.config.system_prompt_template or prompt_generator.get_system_prompt(
                self.config.task_type, self.config.cwe_type
            )
            
            for i, sample in enumerate(samples[:10]):
                logging.info(f"Processing sample {i + 1}/{len(samples)}: {sample.id}")
                
                # Create user prompt with optional call graph context
                code_with_context = self._augment_code_with_context(sample)
                user_prompt = self.config.user_prompt_template.format(code=code_with_context) if self.config.user_prompt_template else prompt_generator.get_user_prompt(
                    self.config.task_type, code_with_context, self.config.cwe_type
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
                    tokens_used=tokens_used
                )
                predictions.append(prediction)
            
            # Calculate metrics
            logging.info("Calculating metrics...")
            if self.config.task_type in [TaskType.BINARY_VULNERABILITY, TaskType.BINARY_CWE_SPECIFIC]:
                metrics = metrics_calculator.calculate_binary_metrics(predictions)
            else:
                metrics = metrics_calculator.calculate_multiclass_metrics(predictions)
            
            # Prepare results
            results: Dict[str, Any] = {
                "benchmark_config": {
                    "model_name": self.config.model_name,
                    "model_type": self.config.model_type.value,
                    "task_type": self.config.task_type.value,
                    "dataset_path": str(self.config.dataset_path),
                    "cwe_type": self.config.cwe_type,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "system_prompt_template": self.config.system_prompt_template,
                    "user_prompt_template": self.config.user_prompt_template
                },
                "dataset_info": {
                    "total_samples": len(samples),
                    "task_type": task_type
                },
                "metrics": metrics,
                "predictions": [
                    {
                        "sample_id": p.sample_id,
                        "predicted_label": p.predicted_label,
                        "true_label": p.true_label,
                        "response_text": p.response_text,
                        "tokens_used": p.tokens_used,
                        "processing_time": p.processing_time
                    }
                    for p in predictions
                ],
                "execution_info": {
                    "total_time": time.time() - start_time,
                    "average_time_per_sample": (time.time() - start_time) / len(samples) if samples else 0,
                    "total_tokens": sum(p.tokens_used or 0 for p in predictions),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Save results
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name_safe = self.config.model_name.replace("/", "_").replace(":", "_")
            task_name = self.config.task_type.value
            cwe_suffix = f"_{self.config.cwe_type}" if self.config.cwe_type else ""
            
            output_filename = f"jitvul_{model_name_safe}_{task_name}{cwe_suffix}_{timestamp}.json"
            output_path = Path(self.config.output_dir) / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Clean up
            llm.cleanup()
            
            # Print summary
            print("JitVul Benchmark Results Summary")
            print(f"{'='*60}")
            print(f"Model: {self.config.model_name}")
            print(f"Task: {self.config.task_type.value}")
            if self.config.cwe_type:
                print(f"CWE Type: {self.config.cwe_type}")
            print(f"Samples: {len(samples)}")
            print(f"Execution Time: {time.time() - start_time:.2f}s")
            print("\nMetrics:")
            
            if self.config.task_type in [TaskType.BINARY_VULNERABILITY, TaskType.BINARY_CWE_SPECIFIC]:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
            else:
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                if 'classification_report' in metrics and isinstance(metrics['classification_report'], dict):
                    report: Dict[str, Any] = metrics['classification_report']
                    if 'macro avg' in report:
                        macro_avg: Dict[str, Any] = report['macro avg']
                        if 'f1-score' in macro_avg:
                            print(f"  Macro F1: {macro_avg['f1-score']:.4f}")
                    if 'weighted avg' in report:
                        weighted_avg: Dict[str, Any] = report['weighted avg']
                        if 'f1-score' in weighted_avg:
                            print(f"  Weighted F1: {weighted_avg['f1-score']:.4f}")
            
            print(f"\nResults saved to: {output_path}")
            print(f"{'='*60}")
            
            return results
            
        except Exception as e:
            logging.exception(f"JitVul benchmark failed: {e}")
            raise
    
    def _augment_code_with_context(self, sample: Any) -> str:
        """
        Augment code with call graph context if available.
        
        Args:
            sample: BenchmarkSample with potential call graph metadata
            
        Returns:
            str: Code with optional call graph context
        """
        code = sample.code
        
        # Check if call graph information is available in metadata
        if sample.metadata and "call_graph" in sample.metadata:
            call_graph = sample.metadata["call_graph"]
            if call_graph:
                # Add call graph as context comment
                context = f"/* Call Graph Context:\n{call_graph}\n*/\n\n"
                code = context + code
        
        return code


def create_jitvul_config_from_args(args: Any) -> BenchmarkConfig:
    """Create a BenchmarkConfig from command line arguments."""
    return BenchmarkConfig(
        model_name=args.model_name,
        model_type=ModelType(args.model_type),
        task_type=TaskType(args.task_type),
        description=f"JitVul benchmark using {args.model_name}",
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        cwe_type=args.cwe_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        system_prompt_template=args.system_prompt,
        user_prompt_template=args.user_prompt
    )


def main():
    """Main entry point for JitVul benchmark runner."""
    parser = argparse.ArgumentParser(description="Run JitVul benchmark evaluation")
    
    # Model configuration
    parser.add_argument("--model-name", required=True, help="Name of the model to evaluate")
    parser.add_argument("--model-type", required=True, help="Type of model")
    
    # Task configuration
    parser.add_argument("--task-type", choices=["binary_vulnerability", "binary_cwe_specific", "multiclass_vulnerability"], 
                       default="binary_vulnerability", help="Type of vulnerability detection task")
    parser.add_argument("--cwe-type", help="Specific CWE type for CWE-specific tasks")
    
    # Dataset configuration
    parser.add_argument("--dataset-path", required=True, help="Path to JitVul dataset file")
    parser.add_argument("--sample-limit", type=int, help="Maximum number of samples to process")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for text generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    
    # Prompt templates
    parser.add_argument("--system-prompt", help="Custom system prompt template")
    parser.add_argument("--user-prompt", help="Custom user prompt template")
    
    # Output configuration
    parser.add_argument("--output-dir", default="results/jitvul", help="Directory to save results")
    parser.add_argument("--config-file", help="JSON configuration file")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load configuration from file if provided
        if args.config_file:
            # Load config from JSON file directly since BenchmarkConfigManager doesn't have load_config
            with open(args.config_file, 'r') as f:
                config_data = json.load(f)
            config = BenchmarkConfig(**config_data)
            # Override with command line arguments
            if args.model_name:
                config.model_name = args.model_name
            if args.dataset_path:
                config.dataset_path = args.dataset_path
            if args.cwe_type:
                config.cwe_type = args.cwe_type
            if args.output_dir != "results/jitvul":
                config.output_dir = args.output_dir
        else:
            config = create_jitvul_config_from_args(args)
        
        # Create and run benchmark
        runner = JitVulBenchmarkRunner(config)
        runner.run_benchmark(sample_limit=args.sample_limit)

        # Print completion message
        print(f"Benchmark completed successfully. Results saved to: {config.output_dir}")
        logging.info("JitVul benchmark completed successfully")
        
    except Exception as e:
        logging.exception(f"JitVul benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()        logging.exception(f"JitVul benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()