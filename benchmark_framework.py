#!/usr/bin/env python3
"""
LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis tasks.
Supports binary and multi-class vulnerability detection with configurable models and datasets.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


class TaskType(Enum):
    """Enumeration of supported task types."""

    BINARY_VULNERABILITY = "binary_vulnerability"
    BINARY_CWE_SPECIFIC = "binary_cwe_specific"
    MULTICLASS_VULNERABILITY = "multiclass_vulnerability"


class ModelType(Enum):
    """Enumeration of supported model types."""

    LLAMA = "meta-llama/Llama-2-7b-chat-hf"
    QWEN = "Qwen/Qwen2.5-7B-Instruct"
    DEEPSEEK = "deepseek-ai/deepseek-coder-6.7b-instruct"
    CODEBERT = "microsoft/codebert-base"
    CUSTOM = "custom"


@dataclass
class BenchmarkSample:
    """Data structure for a single benchmark sample."""

    id: str
    code: str
    label: Union[int, str]
    metadata: Dict[str, Any]
    cwe_type: Optional[str] = None
    severity: Optional[str] = None


@dataclass
class PredictionResult:
    """Data structure for model prediction results."""

    sample_id: str
    predicted_label: Union[int, str]
    true_label: Union[int, str]
    confidence: Optional[float]
    response_text: str
    processing_time: float
    tokens_used: Optional[int] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    model_name: str
    model_type: ModelType
    task_type: TaskType
    description: str
    dataset_path: str
    output_dir: str
    batch_size: int = 1
    max_tokens: int = 512
    temperature: float = 0.1
    use_quantization: bool = True
    cwe_type: Optional[str] = None
    system_prompt_template: Optional[str] = None
    user_prompt_template: Optional[str] = None


class DatasetLoader(Protocol):
    """Protocol for dataset loading implementations."""

    def load_dataset(self, path: str) -> List[BenchmarkSample]:
        """Load dataset from the specified path."""
        ...


class VulBenchLoader:
    """Loader for VulBench dataset format."""

    def load_dataset(self, path: str) -> List[BenchmarkSample]:
        """
        Load VulBench dataset.

        Args:
            path (str): Path to the dataset file

        Returns:
            List[BenchmarkSample]: Loaded samples
        """
        data_path = Path(path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples: List[BenchmarkSample] = []

        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, item in enumerate(data):
                sample = BenchmarkSample(
                    id=item.get("id", f"sample_{i}"),
                    code=item["code"],
                    label=item["label"],
                    metadata=item.get("metadata", {}),
                    cwe_type=item.get("cwe_type"),
                    severity=item.get("severity"),
                )
                samples.append(sample)

        elif data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
            for idx, row in df.iterrows():
                sample = BenchmarkSample(
                    id=row.get("id", f"sample_{idx}"),
                    code=row["code"],
                    label=row["label"],
                    metadata={"source_file": row.get("source_file", "")},
                    cwe_type=row.get("cwe_type"),
                    severity=row.get("severity"),
                )
                samples.append(sample)

        return samples


class PromptGenerator:
    """Generates prompts for different task types."""

    SYSTEM_PROMPTS = {
        TaskType.BINARY_VULNERABILITY: """You are an expert security analyst specializing in static code analysis. 
Your task is to analyze code snippets and determine if they contain security vulnerabilities.

Instructions:
- Analyze the provided code carefully
- Consider common vulnerability patterns (injection, buffer overflow, race conditions, etc.)
- Respond with only "VULNERABLE" or "SAFE" - no additional explanation
- Base your decision on concrete security risks, not coding style issues""",
        TaskType.BINARY_CWE_SPECIFIC: """You are an expert security analyst specializing in static code analysis.
Your task is to analyze code snippets and determine if they contain a specific type of vulnerability: {cwe_type}.

Instructions:
- Analyze the provided code for {cwe_type} vulnerabilities only
- Ignore other types of vulnerabilities
- Respond with only "VULNERABLE" or "SAFE" - no additional explanation
- Focus specifically on {cwe_type} patterns and indicators""",
        TaskType.MULTICLASS_VULNERABILITY: """You are an expert security analyst specializing in static code analysis.
Your task is to analyze code snippets and classify the type of vulnerability present.

Instructions:
- Analyze the provided code carefully
- If vulnerable, identify the primary vulnerability type from: CWE-79, CWE-89, CWE-120, CWE-190, CWE-476, CWE-787
- If no vulnerability is found, respond with "SAFE"
- Respond with only the vulnerability type (e.g., "CWE-79") or "SAFE" - no additional explanation""",
    }

    USER_PROMPTS = {
        TaskType.BINARY_VULNERABILITY: "Analyze this code for security vulnerabilities:\n\n{code}",
        TaskType.BINARY_CWE_SPECIFIC: "Analyze this code for {cwe_type} vulnerabilities:\n\n{code}",
        TaskType.MULTICLASS_VULNERABILITY: "Analyze this code and identify the vulnerability type:\n\n{code}",
    }

    def get_system_prompt(
        self, task_type: TaskType, cwe_type: Optional[str] = None
    ) -> str:
        """Generate system prompt for the given task type."""
        prompt = self.SYSTEM_PROMPTS[task_type]
        if cwe_type and "{cwe_type}" in prompt:
            prompt = prompt.format(cwe_type=cwe_type)
        return prompt

    def get_user_prompt(
        self, task_type: TaskType, code: str, cwe_type: Optional[str] = None
    ) -> str:
        """Generate user prompt for the given task type and code."""
        prompt = self.USER_PROMPTS[task_type]
        if cwe_type and "{cwe_type}" in prompt:
            prompt = prompt.format(code=code, cwe_type=cwe_type)
        else:
            prompt = prompt.format(code=code)
        return prompt


class LLMInterface(ABC):
    """Abstract base class for LLM interfaces."""

    @abstractmethod
    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[str, Optional[int]]:
        """
        Generate response from the model.

        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt

        Returns:
            Tuple[str, Optional[int]]: Response text and token count
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up model resources."""
        pass


class HuggingFaceLLM(LLMInterface):
    """Hugging Face transformers-based LLM interface."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.pipeline = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        logging.info(f"Loading model: {self.config.model_name}")

        # Configure quantization if requested
        quantization_config = None
        if self.config.use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True, padding_side="left"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                trust_remote_code=True,
            )

            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                return_full_text=False,
            )

            logging.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logging.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

    def generate_response(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple[str, Optional[int]]:
        """Generate response using the loaded model."""
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        # Format the prompt based on model type
        formatted_prompt = self._format_prompt(system_prompt, user_prompt)

        try:
            # Generate response
            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response_text = response[0]["generated_text"].strip()

            # Estimate token count
            tokens = self.tokenizer.encode(formatted_prompt + response_text)
            token_count = len(tokens)

            return response_text, token_count

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return f"ERROR: {str(e)}", None

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompt based on model architecture."""
        if "llama" in self.config.model_name.lower():
            return (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
            )
        elif "qwen" in self.config.model_name.lower():
            return f"<|system|>\n{system_prompt}<|endofsystem|>\n<|user|>\n{user_prompt}<|endofuser|>\n<|assistant|>\n"
        elif "deepseek" in self.config.model_name.lower():
            return f"### System:\n{system_prompt}\n\n### User:\n{user_prompt}\n\n### Assistant:\n"
        else:
            # Generic format
            return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        torch.cuda.empty_cache()


class ResponseParser:
    """Parses and normalizes model responses."""

    def __init__(self, task_type: TaskType):
        self.task_type = task_type

    def parse_response(self, response: str) -> Union[int, str]:
        """
        Parse model response into standardized format.

        Args:
            response (str): Raw model response

        Returns:
            Union[int, str]: Parsed response
        """
        response = response.strip().upper()

        if self.task_type == TaskType.BINARY_VULNERABILITY:
            return self._parse_binary_response(response)
        elif self.task_type == TaskType.BINARY_CWE_SPECIFIC:
            return self._parse_binary_response(response)
        elif self.task_type == TaskType.MULTICLASS_VULNERABILITY:
            return self._parse_multiclass_response(response)

        return response

    def _parse_binary_response(self, response: str) -> int:
        """Parse binary classification response."""
        if "VULNERABLE" in response:
            return 1
        elif "SAFE" in response:
            return 0
        else:
            # Try to extract decision from longer responses
            if any(word in response for word in ["YES", "TRUE", "FOUND", "DETECTED"]):
                return 1
            elif any(word in response for word in ["NO", "FALSE", "NONE", "CLEAN"]):
                return 0
            else:
                # Default to safe if unclear
                return 0

    def _parse_multiclass_response(self, response: str) -> str:
        """Parse multiclass response."""
        # Look for CWE patterns
        cwe_pattern = r"CWE-\d+"
        cwe_match = re.search(cwe_pattern, response)

        if cwe_match:
            return cwe_match.group()
        elif "SAFE" in response:
            return "SAFE"
        else:
            return "UNKNOWN"


class MetricsCalculator:
    """Calculates evaluation metrics for benchmark results."""

    @staticmethod
    def calculate_binary_metrics(
        predictions: List[PredictionResult],
    ) -> Dict[str, float]:
        """Calculate metrics for binary classification."""
        y_true = [pred.true_label for pred in predictions]
        y_pred = [pred.predicted_label for pred in predictions]

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    @staticmethod
    def calculate_multiclass_metrics(
        predictions: List[PredictionResult],
    ) -> Dict[str, Any]:
        """Calculate metrics for multiclass classification."""
        y_true = [pred.true_label for pred in predictions]
        y_pred = [pred.predicted_label for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }


class BenchmarkRunner:
    """Main benchmark execution class."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = VulBenchLoader()
        self.prompt_generator = PromptGenerator()
        self.response_parser = ResponseParser(config.task_type)
        self.metrics_calculator = MetricsCalculator()
        self.llm: Optional[LLMInterface] = None

        # Setup logging
        self._setup_logging()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = Path(self.config.output_dir) / "benchmark.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute the complete benchmark.

        Returns:
            Dict[str, Any]: Benchmark results
        """
        logging.info("Starting benchmark execution")
        start_time = time.time()

        try:
            # Load dataset
            logging.info(f"Loading dataset from: {self.config.dataset_path}")
            samples = self.dataset_loader.load_dataset(self.config.dataset_path)
            logging.info(f"Loaded {len(samples)} samples")

            # Initialize model
            logging.info("Initializing model")
            self.llm = HuggingFaceLLM(self.config)

            # Run predictions
            predictions = self._run_predictions(samples)

            # Calculate metrics
            metrics = self._calculate_metrics(predictions)

            # Generate report
            report = self._generate_report(
                samples, predictions, metrics, time.time() - start_time
            )

            # Save results
            self._save_results(report)

            logging.info("Benchmark completed successfully")
            return report

        except Exception as e:
            logging.error(f"Benchmark failed: {e}")
            raise
        finally:
            if self.llm:
                self.llm.cleanup()

    def _run_predictions(
        self, samples: List[BenchmarkSample]
    ) -> List[PredictionResult]:
        """Run model predictions on all samples."""
        predictions: List[PredictionResult] = []

        system_prompt = self.prompt_generator.get_system_prompt(
            self.config.task_type, self.config.cwe_type
        )

        for i, sample in enumerate(samples):
            logging.info(f"Processing sample {i+1}/{len(samples)}: {sample.id}")

            user_prompt = self.prompt_generator.get_user_prompt(
                self.config.task_type, sample.code, self.config.cwe_type
            )

            # Generate response
            start_time = time.time()
            response_text, tokens_used = self.llm.generate_response(
                system_prompt, user_prompt
            )
            processing_time = time.time() - start_time

            # Parse response
            predicted_label = self.response_parser.parse_response(response_text)

            prediction = PredictionResult(
                sample_id=sample.id,
                predicted_label=predicted_label,
                true_label=sample.label,
                confidence=None,  # Could be enhanced to extract confidence
                response_text=response_text,
                processing_time=processing_time,
                tokens_used=tokens_used,
            )

            predictions.append(prediction)

            # Log progress
            if (i + 1) % 10 == 0:
                logging.info(f"Completed {i+1}/{len(samples)} predictions")

        return predictions

    def _calculate_metrics(self, predictions: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        if self.config.task_type in [
            TaskType.BINARY_VULNERABILITY,
            TaskType.BINARY_CWE_SPECIFIC,
        ]:
            return self.metrics_calculator.calculate_binary_metrics(predictions)
        else:
            return self.metrics_calculator.calculate_multiclass_metrics(predictions)

    def _generate_report(
        self,
        samples: List[BenchmarkSample],
        predictions: List[PredictionResult],
        metrics: Dict[str, Any],
        total_time: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""

        report = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.config.model_name,
                "model_type": self.config.model_type.value,
                "task_type": self.config.task_type.value,
                "dataset_path": self.config.dataset_path,
                "cwe_type": self.config.cwe_type,
                "total_samples": len(samples),
                "total_time_seconds": total_time,
            },
            "configuration": asdict(self.config),
            "metrics": metrics,
            "predictions": [asdict(pred) for pred in predictions],
            "sample_analysis": {
                "avg_processing_time": np.mean(
                    [p.processing_time for p in predictions]
                ),
                "total_tokens_used": sum(
                    p.tokens_used for p in predictions if p.tokens_used
                ),
                "error_count": len(
                    [p for p in predictions if "ERROR" in p.response_text]
                ),
            },
        }

        return report

    def _save_results(self, report: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save full report
        report_file = (
            Path(self.config.output_dir) / f"benchmark_report_{timestamp}.json"
        )
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Save metrics summary
        metrics_file = (
            Path(self.config.output_dir) / f"metrics_summary_{timestamp}.json"
        )
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report["metrics"], f, indent=2, ensure_ascii=False, default=str)

        # Save predictions as CSV
        predictions_df = pd.DataFrame(
            [
                asdict(pred)
                for pred in [
                    PredictionResult(**pred_dict) for pred_dict in report["predictions"]
                ]
            ]
        )
        predictions_csv = Path(self.config.output_dir) / f"predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_csv, index=False)

        logging.info(f"Results saved to: {report_file}")


def main():
    """Example usage of the benchmark framework."""

    # Example configuration for binary vulnerability detection
    config = BenchmarkConfig(
        model_name="microsoft/DialoGPT-medium",  # Use a smaller model for testing
        model_type=ModelType.CUSTOM,
        task_type=TaskType.BINARY_VULNERABILITY,
        dataset_path="./data/vulbench_sample.json",
        output_dir="./results",
        batch_size=1,
        max_tokens=128,
        temperature=0.1,
        use_quantization=True,
    )

    # Create and run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run_benchmark()

    print("Benchmark completed!")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    if "f1_score" in results["metrics"]:
        print(f"F1 Score: {results['metrics']['f1_score']:.4f}")


if __name__ == "__main__":
    main()
