# CVEFixes Benchmark Integration

This document describes the integration of the CVEFixes benchmark dataset with the LLM Code Security Benchmark Framework.

## Overview

CVEFixes is a comprehensive dataset containing real-world vulnerabilities from open-source projects. Unlike CASTLE which provides synthetic vulnerability examples, CVEFixes contains actual CVE (Common Vulnerabilities and Exposures) fixes extracted from Git repositories, making it an excellent complement to the existing benchmark suite.

## Key Features

- **Real-world vulnerabilities**: Actual CVEs from production code
- **Multi-language support**: C, Java, Python, and other languages
- **Multi-granularity analysis**: Both file-level and method-level vulnerability detection
- **Rich metadata**: CVE IDs, CWE classifications, CVSS scores, commit information
- **Database-driven**: SQLite database for flexible querying and filtering

## Architecture

### Components

1. **CVEFixesDatasetLoader**: Extracts data from CVEFixes SQLite database
2. **CVEFixesJSONDatasetLoader**: Loads processed JSON datasets
3. **CVEFixesBenchmarkRunner**: Specialized runner for CVEFixes experiments
4. **Dataset preparation scripts**: Tools to create processed datasets

### Data Flow

```
CVEFixes SQLite DB → CVEFixesDatasetLoader → JSON Datasets → Benchmark Runner → Results
```

## Dataset Structure

### Database Schema

The CVEFixes database contains the following key tables:

- **cve**: CVE information, descriptions, CVSS scores
- **fixes**: Links CVEs to commit hashes
- **commits**: Git commit metadata
- **file_change**: File-level changes and code
- **method_change**: Method-level changes and code
- **cwe_classification**: CWE mappings for CVEs
- **repository**: Repository metadata

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

## Setup and Usage

### Prerequisites

1. **CVEFixes Database**: Download the CVEFixes SQLite database
   ```bash
   # Follow instructions at: https://github.com/secureIT-project/CVEfixes
   # Place database at: benchmarks/CVEfixes/Data/CVEfixes.db
   ```

2. **Dependencies**: Install required Python packages
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation

1. **Analyze Database**:
   ```bash
   python prepare_cvefixes_datasets.py \
     --database-path benchmarks/CVEfixes/Data/CVEfixes.db \
     --analyze-only
   ```

2. **Create All Datasets**:
   ```bash
   python prepare_cvefixes_datasets.py \
     --database-path benchmarks/CVEfixes/Data/CVEfixes.db \
     --output-dir datasets_processed/cvefixes
   ```

3. **Create Specific Datasets**:
   ```bash
   # Binary classification only
   python prepare_cvefixes_datasets.py \
     --database-path benchmarks/CVEfixes/Data/CVEfixes.db \
     --dataset-types binary \
     --languages C Java
   
   # CWE-specific datasets
   python prepare_cvefixes_datasets.py \
     --database-path benchmarks/CVEfixes/Data/CVEfixes.db \
     --dataset-types cwe_specific \
     --cwe-types CWE-119 CWE-120 CWE-125
   
   # Limited samples for testing
   python prepare_cvefixes_datasets.py \
     --database-path benchmarks/CVEfixes/Data/CVEfixes.db \
     --sample-limit 1000
   ```

### Running Benchmarks

1. **List Available Experiments**:
   ```bash
   python run_cvefixes_benchmark.py --list
   ```

2. **Run Single Experiment**:
   ```bash
   python run_cvefixes_benchmark.py --experiment cvefixes_binary_basic
   ```

3. **Run Multiple Experiments**:
   ```bash
   python run_cvefixes_benchmark.py \
     --experiment cvefixes_binary_basic cvefixes_method_basic
   ```

4. **Run with Custom Configuration**:
   ```bash
   python run_cvefixes_benchmark.py \
     --config custom_cvefixes_config.json \
     --experiment my_experiment
   ```

5. **Direct Runner Usage**:
   ```bash
   python src/entrypoints/run_cvefixes_benchmark.py \
     --dataset-path datasets_processed/cvefixes/cvefixes_binary_c_file.json \
     --model-type qwen \
     --task-type binary
   ```

## Configuration

### Experiment Configuration

The `cvefixes_experiments_config.json` file defines:

- **Dataset configurations**: Different dataset types and parameters
- **Model configurations**: LLM models and their parameters
- **Prompt strategies**: Different prompting approaches
- **Experiment configs**: Predefined experiment combinations

Example configuration:

```json
{
  "dataset_configurations": {
    "binary_c_file": {
      "dataset_path": "datasets_processed/cvefixes/cvefixes_binary_c_file.json",
      "task_type": "binary_vulnerability",
      "description": "Binary classification: C file-level vulnerability detection"
    }
  },
  "model_configurations": {
    "qwen2.5-7b": {
      "model_name": "Qwen/Qwen2.5-7B-Instruct",
      "model_type": "qwen",
      "max_tokens": 512,
      "temperature": 0.1,
      "use_quantization": true
    }
  },
  "prompt_strategies": {
    "basic_security": {
      "name": "Basic Security Analysis",
      "system_prompt": "You are an expert security analyst...",
      "user_prompt": "Analyze this code for security vulnerabilities:\n\n{code}"
    }
  }
}
```

## Task Types

### Binary Vulnerability Detection

- **Task**: Determine if code contains any security vulnerability
- **Labels**: 1 (vulnerable) or 0 (safe)
- **Note**: All CVEFixes samples are vulnerable by definition

### Multi-class CWE Classification

- **Task**: Identify the specific type of vulnerability (CWE)
- **Labels**: CWE types (e.g., "CWE-119", "CWE-120", etc.)
- **Use case**: Fine-grained vulnerability classification

### CWE-Specific Detection

- **Task**: Detect specific vulnerability types
- **Labels**: 1 (contains target CWE) or 0 (doesn't contain target CWE)
- **Use case**: Specialized detection for specific vulnerability classes

## Dataset Types

### File-Level Analysis

- Analyzes entire files that were modified to fix vulnerabilities
- Provides broader context but may include unrelated code
- Good for understanding vulnerability patterns in larger codebases

### Method-Level Analysis

- Focuses on specific methods that were changed
- More targeted analysis with less noise
- Better for understanding localized vulnerability patterns

## Programming Languages

Currently supported languages:
- **C**: Primary focus, largest number of samples
- **Java**: Good coverage for enterprise applications
- **Python**: Growing collection of vulnerabilities
- **Others**: Limited samples available

## Quality Filters

The dataset preparation includes quality filters:

- **Minimum code length**: Excludes trivial code snippets
- **Maximum code length**: Excludes extremely large files
- **Non-null checks**: Ensures both vulnerable and fixed code exist
- **CWE mapping**: Optional requirement for CWE classification

## Comparison with CASTLE

| Aspect | CASTLE | CVEFixes |
|--------|--------|----------|
| **Source** | Synthetic vulnerabilities | Real-world CVEs |
| **Scale** | ~10K samples | ~100K+ samples |
| **Languages** | Primarily C | Multi-language |
| **Metadata** | Basic CWE, description | Rich CVE data, CVSS scores |
| **Quality** | Consistent, curated | Variable, real-world |
| **Use Case** | Controlled evaluation | Realistic assessment |

## Best Practices

### Dataset Selection

1. **Start with file-level C datasets** for initial evaluation
2. **Use method-level datasets** for focused analysis
3. **Apply sample limits** during development and testing
4. **Consider severity filtering** for critical vulnerabilities only

### Experimental Design

1. **Compare with CASTLE results** to understand model behavior differences
2. **Use multiple prompt strategies** to assess robustness
3. **Analyze by CWE type** to identify model strengths/weaknesses
4. **Consider temporal aspects** (older vs. newer CVEs)

### Performance Considerations

1. **Large datasets**: CVEFixes can be much larger than CASTLE
2. **Memory usage**: Consider sample limits for resource-constrained environments
3. **Processing time**: Real-world code may be more complex to analyze

## Troubleshooting

### Common Issues

1. **Database not found**:
   ```
   Error: CVEFixes database not found at benchmarks/CVEfixes/Data/CVEfixes.db
   Solution: Download and place the CVEFixes database in the correct location
   ```

2. **Empty datasets**:
   ```
   Issue: No samples found for specific language/CWE combination
   Solution: Check database contents, adjust filters, or try different parameters
   ```

3. **Memory errors**:
   ```
   Issue: Out of memory when loading large datasets
   Solution: Use --sample-limit to reduce dataset size
   ```

4. **Model loading failures**:
   ```
   Issue: CUDA out of memory or model loading errors
   Solution: Enable quantization or use smaller models
   ```

### Debugging

1. **Enable debug logging**: `--log-level DEBUG`
2. **Check database statistics**: Use `--analyze-only` flag
3. **Validate datasets**: Inspect generated JSON files
4. **Test with small samples**: Use `--sample-limit 10`

## Future Enhancements

1. **Additional languages**: Expand support for more programming languages
2. **Temporal analysis**: Track vulnerability trends over time
3. **Severity-based evaluation**: Focus on critical/high severity vulnerabilities
4. **Cross-project analysis**: Compare performance across different repositories
5. **Integration with CASTLE**: Combined evaluation strategies

## Contributing

To contribute to the CVEFixes integration:

1. Follow the existing code style and patterns
2. Add appropriate logging and error handling
3. Include unit tests for new functionality
4. Update documentation for any new features
5. Test with multiple dataset configurations

## References

- [CVEFixes Project](https://github.com/secureIT-project/CVEfixes)
- [CVEFixes Paper](https://arxiv.org/abs/2111.08625)
- [Common Weakness Enumeration (CWE)](https://cwe.mitre.org/)
- [Common Vulnerabilities and Exposures (CVE)](https://cve.mitre.org/)

## Status

✅ **COMPLETED** - CVEFixes benchmark integration is fully functional
- All type annotation issues have been resolved
- CVEFixes dataset loader with proper SQLite database integration
- CVEFixes benchmark runner compatible with existing framework
- Comprehensive configuration system supporting multiple task types
- Dataset preparation scripts for processing CVEFixes database
- Integration tests confirming compatibility with CASTLE benchmark
- Complete documentation and usage examples

The CVEFixes benchmark is ready for production use and can be run alongside CASTLE benchmarks.
