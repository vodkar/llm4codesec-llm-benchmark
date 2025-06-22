#!/bin/bash

# Exit on any error
set -ex

# Define the benchmark runner alias
run_benchmark="docker-compose run --rm llm4codesec-benchmark python"

echo "Starting comprehensive benchmark experiments..."

# CASTLE Dataset Experiments
echo "=== Running CASTLE Dataset Experiments ==="
echo "Setting up CASTLE dataset..."
$run_benchmark entrypoints/run_setup_castle_dataset.py

# Define CASTLE experiment plans
castle_plans=(
    "small_models_binary"
    "small_models_cwe_specific_analysis"
    "small_models_multiclass"
)

echo "Running CASTLE experiments..."
for plan in "${castle_plans[@]}"; do
    echo "Executing CASTLE plan: $plan"
    $run_benchmark entrypoints/run_castle_experiments.py --plan "$plan"
done

# CVEFixes Dataset Experiments
echo "=== Running CVEFixes Dataset Experiments ==="
echo "Preparing CVEFixes datasets..."
$run_benchmark entrypoints/prepare_cvefixes_datasets.py \
  --database-path datasets_processed/cvefixes/CVEfixes.db \
  --languages C Java Python

# Define CVEFixes experiment plans
cvefixes_plans=(
    "small_models_binary"
    "small_models_cwe_specific_analysis"
    "small_models_multiclass"
)

echo "Running CVEFixes experiments..."
for plan in "${cvefixes_plans[@]}"; do
    echo "Executing CVEFixes plan: $plan"
    $run_benchmark entrypoints/run_cvefixes_benchmark.py --plan "$plan"
done

# JitVul
echo "=== Running JitVul Dataset Experiments ==="
echo "Preparing JitVul datasets..."
$run_benchmark datasets/setup_jitvul_dataset.py \
    --data-file benchmarks/JitVul/data/final_benchmark.jsonl \
    --all

# Define VulBench experiment plans
vulbench_plans=(
    "small_models_binary"
    "small_models_cwe_specific_analysis"
    "small_models_multiclass"
)

echo "Running VulBench experiments..."
for plan in "${cvefixes_plans[@]}"; do
    echo "Executing VulBench plan: $plan"
    $run_benchmark entrypoints/run_jitvul_benchmark.py --plan "$plan"
done

# VulBench Dataset Experiments
echo "=== Running VulBench Dataset Experiments ==="
echo "Preparing VulBench datasets..."
$run_benchmark datasets/setup_jitvul_dataset.py \
    --data-file benchmarks/JitVul/data/final_benchmark.jsonl \
    --all

# Define VulBench experiment plans
vulbench_plans=(
    "small_models_binary"
    "small_models_cwe_specific_analysis"
    "small_models_multiclass"
)

echo "Running VulBench experiments..."
for plan in "${cvefixes_plans[@]}"; do
    echo "Executing VulBench plan: $plan"
    $run_benchmark entrypoints/run_vulbench_benchmark.py --plan "$plan"
done

echo "All benchmark experiments completed successfully!"
