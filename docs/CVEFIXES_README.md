# CVEFixes Benchmark - Configuration-Based Runner

## Overview

The CVEFixes benchmark provides vulnerability detection evaluation using the CVEFixes dataset containing real-world CVE fixes from open-source projects. The system has been refactored to use a unified configuration-based approach that matches the CASTLE benchmark pattern for consistency across all datasets.

## New Configuration-Based System

### Key Improvements ✅
- **Unified Configuration**: JSON-based experiment configuration following CASTLE pattern
- **Consistent CLI**: Same command-line interface across all benchmarks (CASTLE, JitVul, CVEFixes)
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Single Entry Point**: All experiments configurable through JSON files
- **Model Synchronization**: Consistent model definitions across all datasets

### Core Components
- **Configuration File**: `src/configs/cvefixes_experiments.json`
- **Refactored Runner**: `src/entrypoints/run_cvefixes_benchmark_new.py`
- **Unified Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/cvefixes_dataset_loader.py` (unchanged)

## Configuration Structure

The CVEFixes configuration follows the same structure as CASTLE for consistency:

```json
{
  "experiment_metadata": {
    "benchmark_name": "CVEFixes",
    "version": "1.0.0",
    "description": "Real-world vulnerability detection using CVEFixes dataset"
  },
  "dataset_configurations": {
    "cvefixes_function_level": {
      "dataset_name": "CVEFixes Function Level",
      "dataset_path": "benchmarks/CVEFixes/data/cvefixes_function_level.json",
      "task_type": "function_level_vulnerability_detection"
    }
  },
  "prompt_strategies": {
    "vulnerability_detection": {
      "strategy_name": "CVE Vulnerability Detection",
      "description": "Detect vulnerabilities in real-world code changes"
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
      "datasets": ["cvefixes_function_level"],
      "models": ["qwen2.5-7b"],
      "prompts": ["vulnerability_detection"]
    }
  }
## Command Line Interface

### New Unified Interface

All benchmark runners now use the same CLI arguments:

#### Single Experiments
```bash
# Run with specific model/dataset/prompt
python src/entrypoints/run_cvefixes_benchmark_new.py \
  --model qwen2.5-7b \
  --dataset cvefixes_function_level \
  --prompt vulnerability_detection

# Using unified runner (handles all datasets)
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type cvefixes \
  --model qwen2.5-7b \
  --dataset cvefixes_function_level \
  --prompt vulnerability_detection
```

#### Experiment Plans
```bash
# Run predefined experiment plan
python src/entrypoints/run_cvefixes_benchmark_new.py --plan basic_evaluation

# With unified runner
python src/entrypoints/run_unified_benchmark.py \
  --dataset-type cvefixes \
  --plan basic_evaluation
```

#### Common Options
```bash
# List available configurations
python src/entrypoints/run_cvefixes_benchmark_new.py --list-configs

# Limit samples and set output directory
python src/entrypoints/run_cvefixes_benchmark_new.py \
  --plan basic_evaluation \
  --sample-limit 100 \
  --output-dir results/cvefixes_test
```

## Supported Models

The CVEFixes benchmark supports all models available in the unified configuration:

- **QWEN Series**: Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-72B, Qwen2.5-Coder variants
- **DeepSeek Series**: DeepSeek-Coder-V2, DeepSeek-R1-Distill variants
- **Llama Series**: Llama-3.2-1B, Llama-3.2-3B, CodeLlama variants
- **Gemma Series**: Gemma-3-1B, Gemma-3-27B
- **Wizard Series**: WizardCoder-Python-34B
- **CodeBERT**: microsoft/codebert-base

## Available Task Types

- **function_level_vulnerability_detection**: Function-level vulnerability detection
- **file_level_vulnerability_detection**: File-level vulnerability detection
- **cwe_classification**: CWE type classification
## Key Features

### 1. Configuration-Driven Experiments
- **JSON Configuration**: All experiments defined in `cvefixes_experiments.json`
- **Flexible Combinations**: Easy model/dataset/prompt combinations
- **Experiment Plans**: Predefined experimental setups
- **Consistent Interface**: Same CLI across all benchmarks

### 2. Real-World Vulnerability Data
- **Actual CVEs**: Real vulnerabilities from production code
- **Multi-language Support**: C, Java, Python, and other languages
- **Rich Metadata**: CVE IDs, CWE classifications, CVSS scores
- **Database-driven**: SQLite database for flexible querying

### 3. Comprehensive Evaluation
- **Standard Metrics**: Accuracy, Precision, Recall, F1-score, AUC-ROC
- **CVE-specific Analysis**: Per-CVE type performance
- **Framework Integration**: Uses benchmark framework evaluation system

## Migration from Old System

### Old Commands → New Commands

**Old single experiment:**
```bash
python src/entrypoints/run_cvefixes_benchmark.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset-path cvefixes_data.json \
  --output-dir results/test_run
```

**New single experiment:**
```bash
python src/entrypoints/run_cvefixes_benchmark_new.py \
  --model qwen2.5-7b \
  --dataset cvefixes_function_level \
  --prompt vulnerability_detection \
  --output-dir results/test_run
```

**New experiment plans:**
```bash
python src/entrypoints/run_cvefixes_benchmark_new.py --plan basic_evaluation
```

## Configuration Examples

### Custom Experiment Plan
```json
{
  "experiment_plans": {
    "comprehensive_evaluation": {
      "datasets": ["cvefixes_function_level", "cvefixes_file_level"],
      "models": ["qwen2.5-7b", "deepseek-coder-v2-lite", "llama-3.2-3b"],
      "prompts": ["vulnerability_detection"]
    }
  }
}
```

### Adding New Dataset Configuration
```json
{
  "dataset_configurations": {
    "cvefixes_custom": {
      "dataset_name": "CVEFixes Custom Dataset",
      "dataset_path": "benchmarks/CVEFixes/data/custom_dataset.json",
      "task_type": "custom_vulnerability_detection"
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
python src/entrypoints/run_cvefixes_benchmark_new.py --list-configs

# Run with minimal configuration
python src/entrypoints/run_cvefixes_benchmark_new.py \
  --model qwen2.5-7b \
  --dataset cvefixes_function_level \
  --prompt vulnerability_detection \
  --sample-limit 10
```

## Integration with Framework

The refactored CVEFixes implementation maintains full compatibility with the benchmark framework:

- **Standard Interfaces**: Uses `BenchmarkSample`, `PredictionResult`, `BenchmarkConfig`
- **Consistent Patterns**: Matches CASTLE and JitVul implementation patterns  
- **Framework Integration**: `CVEFixesJSONDatasetLoader` unchanged
- **Unified Metrics**: Framework-standard evaluation metrics

## Files Structure

```
src/
├── configs/
│   └── cvefixes_experiments.json      # New configuration file
├── entrypoints/
│   ├── run_cvefixes_benchmark_new.py      # New refactored runner
│   └── run_unified_benchmark.py           # Unified runner for all datasets
├── datasets/
│   └── cvefixes_dataset_loader.py         # Dataset loader (unchanged)
└── docs/
    └── CVEFIXES_README.md                 # This updated documentation
```

## Dataset Structure

### Sample Format

Each processed sample follows the `BenchmarkSample` structure:

```python
@dataclass  
class BenchmarkSample:
    id: str                    # CVE_ID_type_index (e.g., "CVE-2021-1234_file_0")
    code: str                  # Vulnerable code (before fix)
    label: Union[int, str]     # 1 for vulnerable, CWE type for multiclass
    metadata: Dict[str, Any]   # Rich metadata including CVE details
    cwe_type: Optional[str]    # CWE classification (e.g., "CWE-119")
    severity: Optional[str]    # CVSS severity (CRITICAL, HIGH, MEDIUM, LOW)
```

### Metadata Fields

```python
metadata = {
    "cve_id": "CVE-2021-1234",
    "cwe_id": "119", 
    "severity": 7.5,
    "description": "Buffer overflow in...",
    "published_date": "2021-01-15",
    "programming_language": "C",
    "filename": "src/vulnerable.c",
    "commit_hash": "abc123...",
    "repo_url": "https://github.com/org/repo",
    "lines_added": 2,
    "lines_deleted": 1,
    "change_type": "file",
    "code_after": "/* fixed code */"
}
```

## Conclusion

The CVEFixes benchmark has been successfully refactored to use a unified configuration-based approach that matches the CASTLE benchmark pattern. This provides:

- **Consistent Interface**: Same CLI arguments across all benchmarks
- **Flexible Configuration**: JSON-based experiment definitions
- **Model Synchronization**: Consistent model support across datasets
- **Simplified Usage**: Single entry point with unified runner
- **Maintained Compatibility**: Full framework integration preserved

The new system makes it easier to define experiments, compare models across real-world vulnerability data, and maintain consistency across the entire benchmark suite.

---

**Status**: ✅ REFACTORED AND READY FOR USE  
**Last Updated**: January 2025
