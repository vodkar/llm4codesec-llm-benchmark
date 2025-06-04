# LLM Code Security Benchmark Framework

A comprehensive framework for benchmarking Large Language Models on static code analysis and vulnerability detection tasks. This framework supports multiple LLM models, datasets, and evaluation metrics for binary and multi-class classification tasks.

## Features

- **Multiple Model Support**: Llama2, Qwen2.5, DeepSeek Coder, CodeBERT, and custom models
- **Flexible Task Types**:
  - Binary vulnerability detection
  - CWE-specific vulnerability detection  
  - Multi-class vulnerability classification
- **Dataset Agnostic**: Support for VulBench, JitVul, CASTLE, and custom datasets
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Production Ready**: Full type annotations, logging, error handling, and result persistence
- **Easy Configuration**: Template-based configuration management
- **Local Execution**: Run models locally with optional quantization for efficiency

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended for larger models)

### Setup

1. Clone the repository:

```bash
git clone <your-repo-url>
cd llm4codesec-llm-benchmark
```

2. Install dependencies using Poetry (recommended):

```bash
pip install poetry
poetry install
poetry shell
```

Or using pip:

```bash
pip install -r requirements.txt
```

3. Install additional dependencies for specific models (if needed):

```bash
# For Llama models (requires HF token)
huggingface-cli login

# For GPU support
pip install nvidia-ml-py3
```

## Quick Start

### 1. Create Sample Data and Run Quick Test

```bash
# Create sample dataset and run quick benchmark
python run_benchmark.py --quick
```

### 2. List Available Configurations

```bash
# See all available models and tasks
python run_benchmark.py --list-configs
```

### 3. Run Specific Benchmarks

```bash
# Binary vulnerability detection with Qwen2.5
python run_benchmark.py \
    --model qwen2.5-7b \
    --task binary_vulnerability \
    --dataset ./data/vulbench.json \
    --output ./results/qwen_binary

# CWE-79 (XSS) detection with Llama2
python run_benchmark.py \
    --model llama2-7b \
    --task cwe79_detection \
    --dataset ./data/xss_dataset.json \
    --output ./results/llama_xss

# Multi-class vulnerability classification with DeepSeek
python run_benchmark.py \
    --model deepseek-coder \
    --task multiclass_vulnerability \
    --dataset ./data/mixed_vulnerabilities.json \
    --output ./results/deepseek_multiclass
```

## Dataset Format

The framework expects datasets in JSON format with the following structure:

```json
[
    {
        "id": "sample_001",
        "code": "def get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    return db.execute(query)",
        "label": 1,
        "cwe_type": "CWE-89",
        "severity": "high",
        "metadata": {
            "language": "python",
            "source": "synthetic"
        }
    },
    {
        "id": "sample_002", 
        "code": "def get_user(user_id):\n    query = \"SELECT * FROM users WHERE id = ?\"\n    return db.execute(query, (user_id,))",
        "label": 0,
        "cwe_type": null,
        "severity": null,
        "metadata": {
            "language": "python",
            "source": "synthetic"
        }
    }
]
```

### Required Fields

- `id`: Unique identifier for the sample
- `code`: Source code to analyze
- `label`: Ground truth label (0/1 for binary, string for multiclass)

### Optional Fields

- `cwe_type`: CWE identifier (e.g., "CWE-79", "CWE-89")
- `severity`: Vulnerability severity ("low", "medium", "high")
- `metadata`: Additional information about the sample

## Configuration

### Using Configuration Files

Create a JSON configuration file:

```json
{
    "model": "qwen2.5-7b",
    "task": "binary_vulnerability", 
    "dataset": "./data/my_dataset.json",
    "output": "./results/my_experiment",
    "temperature": 0.1,
    "max_tokens": 512,
    "use_quantization": true
}
```

Then run:

```bash
python run_benchmark.py --config ./my_config.json
```

### Available Models

| Model Key | Model Name | Description |
|-----------|------------|-------------|
| `llama2-7b` | meta-llama/Llama-2-7b-chat-hf | Llama2 7B Chat |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | Qwen2.5 7B Instruct |
| `deepseek-coder` | deepseek-ai/deepseek-coder-6.7b-instruct | DeepSeek Coder 6.7B |
| `codebert` | microsoft/codebert-base | Microsoft CodeBERT |

### Available Tasks

| Task Key | Description | Output Format |
|----------|-------------|---------------|
| `binary_vulnerability` | General vulnerability detection | 0 (safe) / 1 (vulnerable) |
| `cwe79_detection` | XSS vulnerability detection | 0 (safe) / 1 (vulnerable) |
| `cwe89_detection` | SQL injection detection | 0 (safe) / 1 (vulnerable) |
| `multiclass_vulnerability` | Vulnerability type classification | "SAFE" / "CWE-XX" |

## Results and Analysis

### Output Structure

Each benchmark run generates:

```
results/
├── benchmark_report_YYYYMMDD_HHMMSS.json    # Complete results
├── metrics_summary_YYYYMMDD_HHMMSS.json     # Metrics only
├── predictions_YYYYMMDD_HHMMSS.csv          # Individual predictions
└── benchmark.log                             # Execution log
```

### Metrics

#### Binary Classification

- Accuracy
- Precision
- Recall  
- F1-score
- Specificity
- Confusion Matrix (TP, TN, FP, FN)

#### Multi-class Classification

- Accuracy
- Per-class Precision, Recall, F1-score
- Macro/Micro averages
- Confusion Matrix

### Example Results Analysis

```python
import json
import pandas as pd

# Load results
with open('./results/benchmark_report_20241203_143022.json', 'r') as f:
    results = json.load(f)

# Print summary
print(f"Model: {results['benchmark_info']['model_name']}")
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"F1-Score: {results['metrics']['f1_score']:.4f}")

# Load predictions for detailed analysis
predictions_df = pd.read_csv('./results/predictions_20241203_143022.csv')
print(predictions_df.groupby(['true_label', 'predicted_label']).size())
```

## Advanced Usage

### Custom Models

To add support for custom models:

```python
from benchmark_framework import BenchmarkConfig, ModelType, TaskType

config = BenchmarkConfig(
    model_name="your-org/your-model",
    model_type=ModelType.CUSTOM,
    task_type=TaskType.BINARY_VULNERABILITY,
    dataset_path="./data/your_dataset.json",
    output_dir="./results/custom_model"
)
```

### Custom Prompts

Override default prompts by modifying the `PromptGenerator` class or providing custom templates in the configuration.

### Batch Processing

For processing multiple configurations:

```python
from config_manager import BenchmarkConfigManager
from benchmark_framework import BenchmarkRunner

models = ["llama2-7b", "qwen2.5-7b", "deepseek-coder"]
tasks = ["binary_vulnerability", "multiclass_vulnerability"]

for model in models:
    for task in tasks:
        config = BenchmarkConfigManager.create_config(
            model_key=model,
            task_key=task,
            dataset_path="./data/benchmark_dataset.json",
            output_dir=f"./results/{model}_{task}"
        )
        
        runner = BenchmarkRunner(config)
        results = runner.run_benchmark()
        print(f"Completed {model} on {task}: {results['metrics']['accuracy']:.4f}")
```

## Performance Optimization

### Memory Management

- Use quantization for large models: `--no-quantization` to disable
- Adjust batch size based on available memory
- Monitor GPU memory usage during execution

### Speed Optimization

- Use GPU when available
- Enable mixed precision training
- Reduce `max_tokens` for faster inference
- Use smaller model variants for initial testing

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Enable quantization
   - Reduce batch size
   - Use smaller model

2. **Model Loading Fails**:
   - Check model name spelling
   - Ensure HuggingFace token is set for gated models
   - Verify internet connection

3. **Dataset Loading Errors**:
   - Check JSON format
   - Verify required fields are present
   - Ensure file path is correct

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows type hints and style guidelines
5. Submit a pull request

## License

[Your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llm4codesec-benchmark,
    title={LLM Code Security Benchmark Framework},
    author={Your Name},
    year={2024},
    url={https://github.com/your-repo}
}
```
