# CASTLE Benchmark Integration

This document describes the integration of the CASTLE (Code Analysis with Security Testing for Large-scale Evaluation) benchmark with the LLM code security evaluation framework.

## Overview

The CASTLE benchmark is a comprehensive dataset for evaluating static analysis tools on C/C++ code vulnerabilities. This integration provides:

- **Flexible Dataset Processing**: Convert CASTLE's C files into structured JSON datasets
- **Multiple Task Types**: Support for binary classification, CWE-specific detection, and multi-class classification
- **Prompt Experimentation**: Various prompt strategies for different analysis approaches
- **Batch Experiments**: Automated execution of multiple model/prompt/dataset combinations

## Quick Start

### 1. Setup CASTLE Dataset

First, ensure the CASTLE source is available and set up the processed datasets:

```powershell
# Setup processed datasets with all CWEs
python setup_castle_dataset.py --create-prompts --update-gitignore

# Or setup only specific CWEs
python setup_castle_dataset.py --cwes CWE-125 CWE-190 CWE-476 CWE-787
```

### 2. Run a Quick Test

```powershell
# Quick test with 50 samples
python run_castle_benchmark.py --model qwen2.5-7b --dataset binary_all --prompt basic_security --sample-limit 50
```

### 3. Run Experiment Plans

```powershell
# List available experiment plans
python run_castle_experiments.py --list-plans

# Run prompt comparison experiment
python run_castle_experiments.py --plan prompt_comparison

# Run comprehensive evaluation
python run_castle_experiments.py --plan comprehensive_evaluation
```

## File Structure

```
├── castle_dataset_loader.py          # CASTLE dataset loader and processing
├── setup_castle_dataset.py           # Dataset setup and processing script
├── run_castle_benchmark.py           # Single experiment runner
├── run_castle_experiments.py         # Batch experiment runner
├── castle_experiments_config.json    # Experiment configurations
├── datasets_processed/               # Processed datasets (gitignored)
│   └── castle/
│       ├── castle_binary.json        # Binary classification dataset
│       ├── castle_multiclass.json    # Multi-class classification dataset
│       ├── castle_cwe_125.json       # CWE-125 specific dataset
│       └── ...                       # Other CWE-specific datasets
└── results/castle_experiments/       # Experiment results
```

## Dataset Types

### 1. Binary Classification (`binary_all`)
- **Task**: Determine if code is vulnerable or safe
- **Labels**: 0 (safe) or 1 (vulnerable)
- **Use Case**: General vulnerability detection

### 2. Multi-class Classification (`multiclass_all`)
- **Task**: Identify the specific vulnerability type
- **Labels**: "CWE-XXX" or "SAFE"
- **Use Case**: Vulnerability categorization

### 3. CWE-Specific Detection
- **Task**: Detect specific vulnerability types
- **Available CWEs**: CWE-125, CWE-134, CWE-190, CWE-22, CWE-253, CWE-327, etc.
- **Labels**: 0 (no target CWE) or 1 (target CWE present)
- **Use Case**: Focused vulnerability detection

## Prompt Strategies

### 1. Basic Security (`basic_security`)
Simple, direct security analysis prompt.

### 2. Detailed Analysis (`detailed_analysis`)
Comprehensive security analysis with specific guidelines.

### 3. CWE-Focused (`cwe_focused`)
Specialized prompt for CWE-specific vulnerability detection.

### 4. Context-Aware (`context_aware`)
Production environment focused analysis.

### 5. Step-by-Step (`step_by_step`)
Systematic analysis approach with defined steps.

## Experiment Plans

### Quick Test (`quick_test`)
- **Purpose**: Fast validation with limited samples
- **Configuration**: Single model, single prompt, 50 samples
- **Duration**: ~5-10 minutes

### Prompt Comparison (`prompt_comparison`)
- **Purpose**: Compare different prompt strategies
- **Configuration**: Single model, all prompts, binary classification
- **Duration**: ~30-60 minutes

### Model Comparison (`model_comparison`)
- **Purpose**: Compare different LLM models
- **Configuration**: All models, best prompt, binary classification
- **Duration**: ~1-3 hours

### CWE-Specific Analysis (`cwe_specific_analysis`)
- **Purpose**: Evaluate CWE-specific detection capabilities
- **Configuration**: Single model, CWE-focused prompt, all CWE datasets
- **Duration**: ~2-4 hours

### Comprehensive Evaluation (`comprehensive_evaluation`)
- **Purpose**: Full evaluation across all configurations
- **Configuration**: All models, key prompts, all datasets
- **Duration**: ~6-12 hours

## Usage Examples

### Basic Usage

```powershell
# Run binary classification with basic prompt
python run_castle_benchmark.py \
    --model qwen2.5-7b \
    --dataset binary_all \
    --prompt basic_security

# Run CWE-125 detection with focused prompt
python run_castle_benchmark.py \
    --model deepseek-coder \
    --dataset cwe_125 \
    --prompt cwe_focused
```

### Batch Experiments

```powershell
# Run model comparison (test all models with same prompt/dataset)
python run_castle_experiments.py --plan model_comparison

# Run with sample limit for testing
python run_castle_experiments.py --plan comprehensive_evaluation --sample-limit 100
```

### Custom Configurations

Edit `castle_experiments_config.json` to:
- Add new models
- Create custom prompt strategies
- Define new experiment plans
- Modify evaluation settings

## Configuration Details

### Model Configurations
```json
{
  "model_name": "meta-llama/Llama-2-7b-chat-hf",
  "model_type": "LLAMA",
  "max_tokens": 512,
  "temperature": 0.1,
  "batch_size": 1
}
```

### Dataset Configurations
```json
{
  "dataset_path": "datasets_processed/castle/castle_binary.json",
  "task_type": "binary_vulnerability",
  "description": "Binary classification: all vulnerability types"
}
```

### Prompt Strategies
```json
{
  "name": "Basic Security Analysis",
  "system_prompt": "You are an expert security analyst...",
  "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}"
}
```

## Results and Analysis

### Output Structure
```
results/castle_experiments/
├── plan_prompt_comparison_20250609_143022/
│   ├── experiment_plan_results.json
│   ├── qwen2.5-7b_binary_all_basic_security/
│   │   ├── benchmark_report_*.json
│   │   ├── metrics_summary_*.json
│   │   └── predictions_*.csv
│   └── ...
```

### Key Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Detection rate for vulnerabilities
- **F1 Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Analysis Features
- Per-CWE performance breakdown
- Confusion matrices
- Error analysis with sample details
- Performance comparisons across experiments

## Advanced Usage

### Custom Prompt Development

1. Edit `castle_experiments_config.json`
2. Add new prompt strategy in `prompt_strategies` section
3. Test with single experiment:
   ```powershell
   python run_castle_benchmark.py --model qwen2.5-7b --dataset binary_all --prompt your_new_prompt
   ```

### Adding New Models

1. Update `model_configurations` in config file
2. Ensure model is supported by the framework
3. Test with quick experiment first

### Dataset Filtering

For specialized experiments, you can filter datasets:

```python
from castle_dataset_loader import CastleDatasetLoader, filter_by_cwe

loader = CastleDatasetLoader()
samples = loader.load_dataset()
cwe_125_samples = filter_by_cwe(samples, "CWE-125")
```

## Troubleshooting

### Common Issues

1. **Missing Datasets**
   ```powershell
   python setup_castle_dataset.py
   ```

2. **GPU Memory Issues**
   - Reduce batch size in model configuration
   - Enable quantization
   - Use smaller models

3. **Slow Experiments**
   - Use `--sample-limit` for testing
   - Start with `quick_test` plan

### Validation Commands

```powershell
# Check if datasets exist
python run_castle_experiments.py --validate-datasets

# Test setup without running experiments
python run_castle_benchmark.py --setup-only
```

## Contributing

To extend the CASTLE integration:

1. **New Task Types**: Add to `TaskType` enum and update dataset loader
2. **New Metrics**: Extend evaluation in benchmark framework
3. **New Prompts**: Add to configuration and test thoroughly
4. **New Models**: Ensure compatibility with framework model types

## References

- [CASTLE Benchmark Paper](https://github.com/CASTLE-Benchmark)
- [Original CASTLE Repository](https://github.com/CASTLE-Benchmark/CASTLE-Benchmark)
- [LLM Benchmark Framework Documentation](README.md)