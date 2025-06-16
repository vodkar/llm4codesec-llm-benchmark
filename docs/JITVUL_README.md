# JitVul Benchmark - Configuration-Based Runner

## Overview

The JitVul benchmark provides comprehensive vulnerability detection evaluation using the JitVul dataset. The system has been refactored to use a unified configuration-based approach that matches the CASTLE benchmark pattern for consistency across all datasets.

## New Configuration-Based System

### Key Improvements ✅
- **Unified Configuration**: JSON-based experiment configuration following CASTLE pattern
- **Consistent CLI**: Same command-line interface across all benchmarks (CASTLE, JitVul, CVEFixes)
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Single Entry Point**: All experiments configurable through JSON files
- **Model Synchronization**: Consistent model definitions across all datasets

### Core Components
- **Configuration File**: `src/configs/jitvul_experiments.json`
- **Refactored Runner**: `src/entrypoints/run_jitvul_benchmark.py`
- **Unified Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/jitvul_dataset_loader.py` (unchanged)

## Configuration Structure

The JitVul configuration follows the same structure as CASTLE for consistency:

```json
{
  "experiment_metadata": {
    "benchmark_name": "JitVul",
    "version": "2.2.0",
    "description": "Vulnerability detection benchmark using JitVul dataset"
  },
  "dataset_configurations": {
    "jitvul_binary": {
      "dataset_name": "JitVul Binary Classification",
      "dataset_path": "benchmarks/JitVul/data/final_benchmark.jsonl",
      "task_type": "binary_vulnerability"
    }
  },
  "prompt_strategies": {
    "detect_vulnerabilities": {
      "strategy_name": "Vulnerability Detection",
      "description": "Detect if code contains security vulnerabilities"
    }
  },
  "model_configurations": {
    "qwen2.5-7b": {
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "model_type": "QWEN",
      "config": {
        "max_tokens": 4096,
        "temperature": 0.1
      }
    }
  },
  "experiment_plans": {
    "basic_evaluation": {
      "datasets": ["jitvul_binary"],
      "models": ["qwen2.5-7b"],
      "prompts": ["detect_vulnerabilities"]
    }
  }
}
```

## Command Line Interface

### New Unified Interface

All benchmark runners now use the same CLI arguments:

#### Single Experiments
```bash
# Run with specific model/dataset/prompt
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen2.5-7b \
  --dataset jitvul_binary \
  --prompt detect_vulnerabilities

# Using unified runner (handles all datasets)
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type jitvul \
  --model qwen2.5-7b \
  --dataset jitvul_binary \
  --prompt detect_vulnerabilities
```

#### Experiment Plans
```bash
# Run predefined experiment plan
python src/entrypoints/run_jitvul_benchmark.py --plan basic_evaluation

# With unified runner
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type jitvul \
  --plan basic_evaluation
```

#### Common Options
```bash
# List available configurations
python src/entrypoints/run_jitvul_benchmark.py --list-configs

# Limit samples and set output directory
python src/entrypoints/run_jitvul_benchmark.py \
  --plan basic_evaluation \
  --sample-limit 100 \
  --output-dir results/jitvul_test
```
## Supported Models

The JitVul benchmark supports all models available in the unified configuration:

- **QWEN Series**: Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-72B, Qwen2.5-Coder variants
- **DeepSeek Series**: DeepSeek-Coder-V2, DeepSeek-R1-Distill variants
- **Llama Series**: Llama-3.2-1B, Llama-3.2-3B, CodeLlama variants
- **Gemma Series**: Gemma-3-1B, Gemma-3-27B
- **Wizard Series**: WizardCoder-Python-34B
- **CodeBERT**: microsoft/codebert-base

## Available Task Types

- **binary_vulnerability**: Binary classification (vulnerable vs. non-vulnerable)
- **multiclass_vulnerability**: Multiclass CWE type prediction
- **cwe_specific**: Targeted vulnerability type detection

## Key Features

### 1. Configuration-Driven Experiments
- **JSON Configuration**: All experiments defined in `jitvul_experiments.json`
- **Flexible Combinations**: Easy model/dataset/prompt combinations
- **Experiment Plans**: Predefined experimental setups
- **Consistent Interface**: Same CLI across all benchmarks

### 2. Enhanced Context Support
- **Call Graph Integration**: Function relationship context
- **Severity Classification**: Vulnerability severity determination
- **Rich Metadata**: Project info, commit details, function hashes

### 3. Comprehensive Evaluation
- **Standard Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **Per-Class Analysis**: Individual CWE type performance
- **Framework Integration**: Uses benchmark framework evaluation system

## Migration from Old System

### Old Commands → New Commands

**Old single experiment:**
```bash
python src/entrypoints/run_jitvul_benchmark.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task-type binary_vulnerability \
  --dataset-path jitvul/ \
  --output-dir results/test_run
```

**New single experiment:**
```bash
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen2.5-7b \
  --dataset jitvul_binary \
  --prompt detect_vulnerabilities \
  --output-dir results/test_run
```

**Old batch experiments:**
```bash
python src/entrypoints/run_jitvul_batch.py --config batch_config.json
```

**New experiment plans:**
```bash
python src/entrypoints/run_jitvul_benchmark.py --plan basic_evaluation
```

## Configuration Examples

### Custom Experiment Plan
```json
{
  "experiment_plans": {
    "my_custom_plan": {
      "datasets": ["jitvul_binary", "jitvul_multiclass"],
      "models": ["qwen2.5-7b", "deepseek-coder-v2-lite"],
      "prompts": ["detect_vulnerabilities"]
    }
  }
}
```

### Adding New Model
```json
{
  "model_configurations": {
    "my_custom_model": {
      "model_name": "custom/my-model",
      "model_type": "CUSTOM",
      "config": {
        "max_tokens": 2048,
        "temperature": 0.0
      }
    }
  }
}
```

## Troubleshooting

### Configuration Issues
- **Model not found**: Check `model_configurations` section in config file
- **Dataset path error**: Verify `dataset_path` in `dataset_configurations`
- **Invalid experiment plan**: Ensure all referenced models/datasets/prompts exist

### Common Command Fixes
```bash
# Check available configurations
python src/entrypoints/run_jitvul_benchmark.py --list-configs

# Run with minimal configuration
python src/entrypoints/run_jitvul_benchmark.py \
  --model qwen2.5-7b \
  --dataset jitvul_binary \
  --prompt detect_vulnerabilities \
  --sample-limit 10
```

## Integration with Framework

The refactored JitVul implementation maintains full compatibility with the benchmark framework:

- **Standard Interfaces**: Uses `BenchmarkSample`, `PredictionResult`, `BenchmarkConfig`  
- **Consistent Patterns**: Matches CASTLE and CVEFixes implementation patterns
- **Framework Integration**: `JitVulDatasetLoaderFramework` unchanged
- **Unified Metrics**: Framework-standard evaluation metrics

## Files Structure

```
src/
├── configs/
│   └── jitvul_experiments.json        # New configuration file
├── entrypoints/
│   ├── run_jitvul_benchmark.py        # New refactored runner
│   └── run_unified_benchmark.py           # Unified runner for all datasets
├── datasets/
│   └── jitvul_dataset_loader.py           # Dataset loader (unchanged)
└── docs/
    └── JITVUL_README.md                   # This updated documentation
```

## Research Applications

This implementation supports various research directions:

### Model Comparison Studies
- Systematic evaluation across different LLMs
- Performance analysis on specific vulnerability types
- Context sensitivity studies

### Prompt Engineering Research
- Vulnerability-specific prompt optimization
- Context augmentation strategies
- Few-shot vs zero-shot performance

### Dataset Analysis
- Vulnerability distribution studies
- Difficulty assessment across CWE types
- Project-specific vulnerability patterns

### Methodological Research
- Evaluation metric comparison
- Sampling strategy effects

## Citation and Attribution

When using this implementation, please cite both the original JitVul dataset and this implementation:

```bibtex
@misc{jitvul_benchmark_implementation,
  title={JitVul Benchmark Implementation for LLM Vulnerability Detection},
  author={LLM4CodeSec Benchmark Framework},
  year={2024},
  url={https://github.com/your-repo/llm4codesec-llm-benchmark}
}
```

## Conclusion

The JitVul benchmark has been successfully refactored to use a unified configuration-based approach that matches the CASTLE benchmark pattern. This provides:

- **Consistent Interface**: Same CLI arguments across all benchmarks
- **Flexible Configuration**: JSON-based experiment definitions 
- **Model Synchronization**: Consistent model support across datasets
- **Simplified Usage**: Single entry point with unified runner
- **Maintained Compatibility**: Full framework integration preserved

The new system makes it easier to define experiments, compare models, and maintain consistency across the entire benchmark suite.

---

**Status**: ✅ REFACTORED AND READY FOR USE  
**Last Updated**: January 2025
