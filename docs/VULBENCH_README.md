# VulBench Benchmark - Configuration-Based Runner

## Overview

The VulBench benchmark provides comprehensive vulnerability detection evaluation using the VulBench dataset containing multiple vulnerability datasets (D2A, CTF, MAGMA, Big-Vul, Devign). The system has been implemented to use a unified configuration-based approach that matches the CASTLE benchmark pattern for consistency across all datasets.

## New Configuration-Based System

### Key Features ✅
- **Unified Configuration**: JSON-based experiment configuration following CASTLE pattern
- **Consistent CLI**: Same command-line interface across all benchmarks (CASTLE, JitVul, CVEFixes, VulBench)
- **Multiple Datasets**: Support for 5 VulBench sub-datasets with binary and multiclass variants
- **Flexible Experiments**: Easy definition of model/dataset/prompt combinations
- **Single Entry Point**: All experiments configurable through JSON files
- **Model Synchronization**: Consistent model definitions across all datasets

### Core Components
- **Configuration File**: `src/configs/vulbench_experiments.json`
- **Benchmark Runner**: `src/entrypoints/run_vulbench_benchmark_new.py`
- **Unified Runner**: `src/entrypoints/run_unified_benchmark.py` (handles all datasets)
- **Dataset Loader**: `src/datasets/loaders/vulbench_dataset_loader.py`
- **Data Processor**: `src/scripts/process_vulbench_data.py`

## VulBench Datasets

VulBench contains 5 sub-datasets, each available in both binary and multiclass variants:

| Dataset | Description | Binary Classification | Multiclass Classification |
|---------|-------------|----------------------|---------------------------|
| **D2A** | Detect-to-Assign dataset | Vulnerable vs Non-vulnerable | CWE-specific vulnerability types |
| **CTF** | Capture The Flag challenges | Vulnerable vs Non-vulnerable | CWE-specific vulnerability types |
| **MAGMA** | Generated vulnerabilities | Vulnerable vs Non-vulnerable | CWE-specific vulnerability types |
| **Big-Vul** | Large-scale vulnerability dataset | Vulnerable vs Non-vulnerable | CWE-specific vulnerability types |
| **Devign** | Graph-based vulnerability dataset | Vulnerable vs Non-vulnerable | CWE-specific vulnerability types |

## Configuration Structure

The VulBench configuration follows the same structure as other benchmarks for consistency:

```json
{
  "experiment_metadata": {
    "name": "VulBench Benchmark LLM Evaluation",
    "description": "Comprehensive evaluation of LLMs on VulBench benchmark",
    "version": "1.0",
    "dataset": "VulBench v1.0"
  },
  "dataset_configurations": {
    "binary_d2a": {
      "dataset_path": "datasets_processed/vulbench/vulbench_binary_d2a.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: D2A vulnerability detection"
    },
    "multiclass_big_vul": {
      "dataset_path": "datasets_processed/vulbench/vulbench_multiclass_big_vul.json",
      "task_type": "multiclass_vulnerability",
      "description": "Multi-class classification: Big-Vul vulnerability type identification"
    }
  },
  "prompt_strategies": {
    "vulnerability_detection_basic": {
      "strategy_name": "Basic Vulnerability Detection",
      "description": "Simple binary classification prompt for vulnerability detection"
    },
    "vulnerability_classification_cwe": {
      "strategy_name": "CWE-based Vulnerability Classification",
      "description": "Multi-class classification with specific CWE categories"
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
  }
}
```

## Quick Start

### 1. Data Processing

First, process the raw VulBench data to create structured JSON datasets:

```bash
# Process VulBench data for all datasets
python src/scripts/process_vulbench_data.py

# Process specific dataset
python src/scripts/process_vulbench_data.py --dataset d2a

# Process both binary and multiclass variants
python src/scripts/process_vulbench_data.py --dataset big_vul --binary --multiclass
```

### 2. List Available Configurations

```bash
# Using the unified runner (recommended)
python run_unified_benchmark.py vulbench --list-configs

# Using the direct VulBench runner
python src/entrypoints/run_vulbench_benchmark_new.py --list-configs
```

### 3. Run Specific Experiments

```bash
# Run binary vulnerability detection on D2A dataset with GPT-4
python run_unified_benchmark.py vulbench \
  --plan quick_test \
  --model gpt-4-turbo \
  --dataset binary_d2a \
  --prompt vulnerability_detection_basic

# Run multiclass classification on Big-Vul with multiple models  
python run_unified_benchmark.py vulbench \
  --plan comprehensive_evaluation \
  --model qwen2.5-7b,claude-3.5-sonnet \
  --dataset multiclass_big_vul \
  --prompt vulnerability_classification_cwe
```

### 4. Run Full Experiment Plans

```bash
# Quick test across multiple datasets
python run_unified_benchmark.py vulbench --plan quick_test

# Comprehensive evaluation with all models and prompts
python run_unified_benchmark.py vulbench --plan comprehensive_evaluation

# Binary-only evaluation
python run_unified_benchmark.py vulbench --plan binary_classification_focus
```

## Available Experiment Plans

The configuration includes several pre-defined experiment plans:

### `quick_test`
- **Purpose**: Fast validation of setup
- **Datasets**: Binary D2A, Binary Big-Vul
- **Models**: GPT-4 Turbo, Qwen2.5-7B
- **Prompts**: Basic vulnerability detection
- **Runtime**: ~30 minutes

### `comprehensive_evaluation`
- **Purpose**: Full benchmark evaluation
- **Datasets**: All 10 datasets (5 binary + 5 multiclass)
- **Models**: All 11 configured models
- **Prompts**: All 5 prompt strategies
- **Runtime**: Several hours

### `binary_classification_focus`
- **Purpose**: Focus on binary vulnerability detection
- **Datasets**: All 5 binary datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5, CodeLlama
- **Prompts**: Basic and contextual detection
- **Runtime**: ~2 hours

### `multiclass_analysis`
- **Purpose**: Focus on vulnerability type classification
- **Datasets**: All 5 multiclass datasets
- **Models**: GPT-4, Claude-3.5, Qwen2.5
- **Prompts**: CWE-based and detailed classification
- **Runtime**: ~3 hours

### `model_comparison`
- **Purpose**: Compare different model families
- **Datasets**: Binary and multiclass D2A, Big-Vul
- **Models**: Representative models from each family
- **Prompts**: Standardized prompts for fair comparison
- **Runtime**: ~1.5 hours

## Prompt Strategies

VulBench includes 5 different prompt strategies optimized for vulnerability detection:

1. **`vulnerability_detection_basic`**: Simple binary classification
2. **`vulnerability_detection_contextual`**: Enhanced with context and examples
3. **`vulnerability_classification_cwe`**: Multi-class with CWE categories
4. **`vulnerability_classification_detailed`**: Detailed analysis with reasoning
5. **`code_security_analysis`**: Comprehensive security assessment

## Model Configurations

The benchmark supports 11 different models across various families:

- **OpenAI**: GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo
- **Anthropic**: Claude-3.5 Sonnet, Claude-3 Haiku
- **Open Source**: Qwen2.5-7B, CodeLlama-7B, CodeLlama-13B, DeepSeek-Coder-7B
- **Specialized**: CodeT5+, WizardCoder-15B

## Data Processing Details

### Raw VulBench Structure
```
benchmarks/VulBench/VulBench/
├── d2a/
│   ├── sample_001/
│   │   ├── meta_data.json
│   │   └── src.c
│   └── sample_002/
│       ├── meta_data.json
│       └── src.c
├── ctf/
├── magma/
├── big_vul/
└── devign/
```

### Processed Structure
```
datasets_processed/vulbench/
├── vulbench_binary_d2a.json
├── vulbench_multiclass_d2a.json
├── vulbench_binary_big_vul.json
├── vulbench_multiclass_big_vul.json
├── vulbench_binary_devign.json
├── vulbench_multiclass_devign.json
└── stats/
    ├── d2a_stats.json
    ├── big_vul_stats.json
    └── devign_stats.json
```

### Data Processing Statistics

Based on successful processing:

| Dataset | Samples | Binary Labels | Multiclass CWEs | Status |
|---------|---------|---------------|-----------------|---------|
| D2A     | 69      | 34 Vuln / 35 Safe | 8 CWE types | ✅ Processed |
| Big-Vul | 108     | 54 Vuln / 54 Safe | 12 CWE types | ✅ Processed |
| Devign  | 70      | 35 Vuln / 35 Safe | 9 CWE types | ✅ Processed |
| CTF     | 0       | - | - | ⚠️ Structure Issues |
| MAGMA   | 0       | - | - | ⚠️ Structure Issues |

**Note**: CTF and MAGMA datasets had structural issues during processing (missing `src.c` files in many directories). These datasets may require manual inspection and different processing logic.

## Command Line Options

### Direct VulBench Runner
```bash
python src/entrypoints/run_vulbench_benchmark_new.py [OPTIONS]

Options:
  --list-configs          List all available configurations
  --plan PLAN_NAME        Run a specific experiment plan
  --model MODEL_NAMES     Comma-separated list of model names
  --dataset DATASET_NAMES Comma-separated list of dataset names
  --prompt PROMPT_NAMES   Comma-separated list of prompt strategy names
  --output-dir DIR        Output directory for results
  --max-samples N         Limit number of samples per dataset
  --debug                 Enable debug logging
  --dry-run              Show what would be executed without running
```

### Unified Runner
```bash
python run_unified_benchmark.py vulbench [OPTIONS]

# Same options as above, with 'vulbench' as the dataset type
```

## Output and Results

### Directory Structure
```
results/
└── vulbench/
    ├── {timestamp}_vulbench_results/
    │   ├── experiment_config.json
    │   ├── detailed_results.json
    │   ├── summary_results.json
    │   └── logs/
    │       ├── run.log
    │       └── model_responses/
    └── {timestamp}_vulbench_quick_test/
        └── ...
```

### Result Files

1. **`experiment_config.json`**: Complete experiment configuration
2. **`detailed_results.json`**: Per-sample predictions and metrics
3. **`summary_results.json`**: Aggregated results and statistics
4. **`run.log`**: Execution logs and timing information
5. **`model_responses/`**: Raw model responses for debugging

### Metrics Calculated

For each model-dataset-prompt combination:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged
- **Confusion Matrix**: For detailed error analysis
- **Processing Time**: Model response times
- **Token Statistics**: Input/output token counts

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   ```bash
   # Ensure data is processed first
   python src/scripts/process_vulbench_data.py
   ```

2. **Model Loading Errors**
   ```bash
   # Check API keys are set
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   ```

3. **Memory Issues with Large Models**
   ```bash
   # Use smaller models or reduce batch size
   python run_unified_benchmark.py vulbench --plan quick_test --model qwen2.5-7b
   ```

4. **CTF/MAGMA Processing Issues**
   - These datasets had structural issues during processing
   - Manual inspection may be required
   - Consider excluding from experiments until resolved

### Debugging

Enable debug mode for detailed logging:
```bash
python run_unified_benchmark.py vulbench --debug --plan quick_test
```

Use dry-run to preview experiments:
```bash
python run_unified_benchmark.py vulbench --dry-run --plan comprehensive_evaluation
```

## Integration with Other Benchmarks

VulBench is fully integrated with the unified benchmark system:

```bash
# Run multiple benchmarks in sequence
python run_unified_benchmark.py castle --plan quick_test
python run_unified_benchmark.py jitvul --plan quick_test  
python run_unified_benchmark.py cvefixes --plan quick_test
python run_unified_benchmark.py vulbench --plan quick_test

# Compare results across benchmarks
python scripts/compare_benchmark_results.py \
  results/castle/ \
  results/jitvul/ \
  results/cvefixes/ \
  results/vulbench/
```

## Next Steps

1. **Full Benchmark Execution**: Run comprehensive evaluation to validate end-to-end functionality
2. **CTF/MAGMA Investigation**: Resolve structural issues with these datasets
3. **Result Analysis**: Develop scripts for cross-benchmark comparison
4. **Performance Optimization**: Optimize for large-scale experiments
5. **Documentation Enhancement**: Add more examples and use cases

## Support

For issues specific to VulBench implementation:
1. Check the logs in the output directory
2. Verify data processing completed successfully
3. Ensure all required dependencies are installed
4. Review the experiment configuration for correct paths

For general benchmark framework issues, refer to the main README.md and other benchmark documentation.