#!/usr/bin/env python3
"""
Test script for CVEFixes dataset loader functionality.

This script tests the CVEFixes integration without requiring the full database.
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from benchmark.benchmark_framework import BenchmarkSample
from datasets.loaders.cvefixes_dataset_loader import CVEFixesJSONDatasetLoader


def create_test_dataset():
    """Create a minimal test dataset for CVEFixes."""
    test_data = {
        "metadata": {
            "name": "CVEFixes-Test",
            "version": "1.0",
            "task_type": "binary",
            "programming_language": "C",
            "change_level": "file",
            "total_samples": 2,
            "vulnerable_samples": 2,
            "cwe_distribution": {
                "CWE-119": 1,
                "CWE-120": 1
            },
            "severity_distribution": {
                "HIGH": 1,
                "MEDIUM": 1
            }
        },
        "samples": [
            {
                "id": "CVE-2021-1234_file_0",
                "code": """
#include <stdio.h>
#include <string.h>

void vulnerable_function(char *input) {
    char buffer[10];
    strcpy(buffer, input);  // Vulnerable: no bounds checking
    printf("Input: %s\\n", buffer);
}

int main() {
    char user_input[100];
    gets(user_input);  // Vulnerable: no bounds checking
    vulnerable_function(user_input);
    return 0;
}
""",
                "label": 1,
                "cwe_type": "CWE-119",
                "severity": "HIGH",
                "metadata": {
                    "cve_id": "CVE-2021-1234",
                    "cwe_id": "119",
                    "severity": 7.5,
                    "description": "Buffer overflow in example application",
                    "published_date": "2021-01-15",
                    "programming_language": "C",
                    "filename": "src/vulnerable.c",
                    "commit_hash": "abc123def456",
                    "repo_url": "https://github.com/example/vulnerable-app",
                    "lines_added": 2,
                    "lines_deleted": 1,
                    "change_type": "file"
                }
            },
            {
                "id": "CVE-2021-5678_file_1",
                "code": """
#include <stdio.h>
#include <stdlib.h>

int process_data(char *data) {
    char *buffer = malloc(50);
    if (!buffer) return -1;
    
    // Vulnerable: buffer overflow possible
    sprintf(buffer, "%s_processed", data);
    
    printf("Processed: %s\\n", buffer);
    free(buffer);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        process_data(argv[1]);
    }
    return 0;
}
""",
                "label": 1,
                "cwe_type": "CWE-120",
                "severity": "MEDIUM",
                "metadata": {
                    "cve_id": "CVE-2021-5678",
                    "cwe_id": "120",
                    "severity": 5.5,
                    "description": "Classic buffer overflow in sprintf",
                    "published_date": "2021-03-20",
                    "programming_language": "C",
                    "filename": "src/process.c",
                    "commit_hash": "def456ghi789",
                    "repo_url": "https://github.com/example/data-processor",
                    "lines_added": 3,
                    "lines_deleted": 1,
                    "change_type": "file"
                }
            }
        ]
    }
    
    return test_data


def test_cvefixes_json_loader():
    """Test the CVEFixes JSON dataset loader."""
    print("Testing CVEFixes JSON Dataset Loader...")
    
    # Create test dataset
    test_data = create_test_dataset()
    test_file = Path("test_cvefixes_dataset.json")
    
    try:
        # Save test data
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Load with CVEFixes loader
        loader = CVEFixesJSONDatasetLoader()
        samples = loader.load_dataset(str(test_file))
        
        # Validate results
        assert len(samples) == 2, f"Expected 2 samples, got {len(samples)}"
        
        # Check first sample
        sample1 = samples[0]
        assert isinstance(sample1, BenchmarkSample), "Sample is not BenchmarkSample instance"
        assert sample1.id == "CVE-2021-1234_file_0", f"Wrong ID: {sample1.id}"
        assert sample1.label == 1, f"Wrong label: {sample1.label}"
        assert sample1.cwe_types == "CWE-119", f"Wrong CWE: {sample1.cwe_types}"
        assert sample1.severity == "HIGH", f"Wrong severity: {sample1.severity}"
        assert "strcpy" in sample1.code, "Code doesn't contain expected content"
        
        # Check metadata
        assert sample1.metadata["cve_id"] == "CVE-2021-1234"
        assert sample1.metadata["programming_language"] == "C"
        
        # Check second sample
        sample2 = samples[1]
        assert sample2.id == "CVE-2021-5678_file_1"
        assert sample2.cwe_types == "CWE-120"
        assert sample2.severity == "MEDIUM"
        assert "sprintf" in sample2.code
        
        print("✓ CVEFixes JSON loader test passed!")
        return True
        
    except Exception as e:
        print(f"✗ CVEFixes JSON loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_dataset_creation():
    """Test dataset creation functionality."""
    print("Testing dataset creation...")
    
    try:
        test_data = create_test_dataset()
        
        # Validate structure
        assert "metadata" in test_data
        assert "samples" in test_data
        assert len(test_data["samples"]) == 2
        
        metadata = test_data["metadata"]
        assert metadata["name"] == "CVEFixes-Test"
        assert metadata["total_samples"] == 2
        assert metadata["vulnerable_samples"] == 2
        
        # Validate samples
        for i, sample in enumerate(test_data["samples"]):
            assert "id" in sample
            assert "code" in sample
            assert "label" in sample
            assert "cwe_type" in sample
            assert "severity" in sample
            assert "metadata" in sample
            
            assert sample["label"] == 1  # All vulnerable
            assert sample["cwe_type"].startswith("CWE-")
            assert len(sample["code"]) > 50  # Reasonable code length
        
        print("✓ Dataset creation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_sample_compatibility():
    """Test compatibility with BenchmarkSample structure."""
    print("Testing BenchmarkSample compatibility...")
    
    try:
        # Create a sample directly
        sample = BenchmarkSample(
            id="CVE-2021-TEST_file_0",
            code="int main() { char buf[10]; gets(buf); return 0; }",
            label=1,
            metadata={
                "cve_id": "CVE-2021-TEST",
                "cwe_id": "120",
                "programming_language": "C"
            },
            cwe_types="CWE-120",
            severity="HIGH"
        )
        
        # Validate sample
        assert sample.id == "CVE-2021-TEST_file_0"
        assert sample.label == 1
        assert sample.cwe_types == "CWE-120"
        assert sample.severity == "HIGH"
        assert "gets" in sample.code
        assert sample.metadata["cve_id"] == "CVE-2021-TEST"
        
        print("✓ BenchmarkSample compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"✗ BenchmarkSample compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CVEFixes Integration Test Suite")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    tests = [
        test_dataset_creation,
        test_benchmark_sample_compatibility,
        test_cvefixes_json_loader,
    ]
    
    results = []
    for test_func in tests:
        print(f"\n{'-' * 40}")
        result = test_func()
        results.append(result)
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{test_func.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CVEFixes integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
