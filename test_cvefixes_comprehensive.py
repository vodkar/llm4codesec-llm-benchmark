#!/usr/bin/env python3
"""
CVEFixes Benchmark Integration Test

Test the complete CVEFixes benchmark pipeline including dataset loading,
sample processing, and benchmark runner functionality.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

try:
    from benchmark.benchmark_framework import BenchmarkSample
    from datasets.loaders.cvefixes_dataset_loader import CVEFixesDatasetLoader, CVEFixesJSONDatasetLoader
    from entrypoints.run_cvefixes_benchmark import CVEFixesBenchmarkRunner
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def test_cvefixes_benchmark_runner():
    """Test the CVEFixes benchmark runner."""
    print("Testing CVEFixes benchmark runner...")
    
    # Create test configuration
    test_config = {
        "experiment_name": "test_cvefixes_experiment",
        "datasets": [
            {
                "name": "test_binary",
                "path": "test_cvefixes_dataset.json",
                "task_type": "binary",
                "sample_limit": 2
            }
        ],
        "models": [
            {
                "name": "test_model",
                "type": "mock",
                "config": {"test": True}
            }
        ],
        "prompt_strategies": ["direct"],
        "output_dir": "test_results"
    }
    
    # Save test config
    with open("test_cvefixes_config.json", "w") as f:
        json.dump(test_config, f, indent=2)
    
    try:
        # Initialize runner
        runner = CVEFixesBenchmarkRunner("test_cvefixes_config.json")
        
        # Test that runner initializes correctly
        assert runner is not None
        assert hasattr(runner, 'config')
        assert hasattr(runner, 'logger')
        
        print("✓ CVEFixes benchmark runner test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CVEFixes benchmark runner test failed: {e}")
        return False
    
    finally:
        # Clean up
        Path("test_cvefixes_config.json").unlink(missing_ok=True)

def test_dataset_metadata_extraction():
    """Test metadata extraction from CVEFixes dataset."""
    print("Testing dataset metadata extraction...")
    
    try:
        # Test with our sample data
        loader = CVEFixesJSONDatasetLoader()
        samples = loader.load_dataset("test_cvefixes_dataset.json")
        
        # Verify metadata structure
        assert len(samples) > 0
        
        for sample in samples:
            assert hasattr(sample, 'metadata')
            assert 'cve_id' in sample.metadata
            assert 'cwe_id' in sample.metadata
            assert 'severity' in sample.metadata
            assert 'programming_language' in sample.metadata
            assert sample.cwe_type is not None
            
        print("✓ Dataset metadata extraction test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset metadata extraction test failed: {e}")
        return False

def test_task_type_processing():
    """Test different task types (binary, multiclass, CWE-specific)."""
    print("Testing task type processing...")
    
    try:
        # Create test samples with different CWE types
        test_samples = [
            {
                "id": "test_1",
                "code": "test code 1",
                "label": 1,
                "cwe_type": "CWE-89",
                "severity": "HIGH",
                "metadata": {"cve_id": "CVE-2023-0001", "cwe_id": "89"}
            },
            {
                "id": "test_2", 
                "code": "test code 2",
                "label": 1,
                "cwe_type": "CWE-78",
                "severity": "MEDIUM",
                "metadata": {"cve_id": "CVE-2023-0002", "cwe_id": "78"}
            }
        ]
        
        test_dataset = {
            "metadata": {
                "name": "test_multiclass",
                "task_type": "multiclass"
            },
            "samples": test_samples
        }
        
        # Save test dataset
        with open("test_multiclass_dataset.json", "w") as f:
            json.dump(test_dataset, f, indent=2)
        
        # Load and test
        loader = CVEFixesJSONDatasetLoader()
        samples = loader.load_dataset("test_multiclass_dataset.json")
        
        # Verify samples loaded correctly
        assert len(samples) == 2
        assert samples[0].cwe_type == "CWE-89"
        assert samples[1].cwe_type == "CWE-78"
        
        print("✓ Task type processing test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Task type processing test failed: {e}")
        return False
        
    finally:
        # Clean up
        Path("test_multiclass_dataset.json").unlink(missing_ok=True)

def run_comprehensive_test():
    """Run comprehensive CVEFixes integration test."""
    print("=" * 60)
    print("CVEFixes Comprehensive Integration Test")
    print("=" * 60)
    
    tests = [
        ("CVEFixes Benchmark Runner", test_cvefixes_benchmark_runner),
        ("Dataset Metadata Extraction", test_dataset_metadata_extraction),
        ("Task Type Processing", test_task_type_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Testing {test_name}...")
        print(f"{'-' * 40}")
        
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = "FAIL"
    
    # Print summary
    print("\n" + "=" * 60)
    print("Comprehensive Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASS" else "✗"
        print(f"{status_symbol} {test_name}: {result}")
        if result == "PASS":
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed! CVEFixes integration is fully functional.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(name)s - %(levelname)s - %(message)s'
    )
    
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
