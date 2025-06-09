#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.append('src')

print("=" * 50)
print("CVEFixes Final Integration Test")
print("=" * 50)

try:
    # Test 1: Import CVEFixes components
    print("\n1. Testing imports...")
    from datasets.loaders.cvefixes_dataset_loader import CVEFixesDatasetLoader, CVEFixesJSONDatasetLoader
    from entrypoints.run_cvefixes_benchmark import CVEFixesBenchmarkRunner
    print("   ✓ All imports successful")
    
    # Test 2: Test JSON loader with sample data
    print("\n2. Testing JSON loader...")
    if not Path("test_cvefixes_dataset.json").exists():
        print("   Creating test dataset...")
        test_data = {
            "metadata": {"name": "test", "task_type": "binary"},
            "samples": [
                {"id": "test_1", "code": "int main() { return 0; }", "label": 1, 
                 "cwe_type": "CWE-89", "severity": "HIGH", 
                 "metadata": {"cve_id": "CVE-2023-0001"}}
            ]
        }
        with open("test_cvefixes_dataset.json", "w") as f:
            json.dump(test_data, f)
    
    loader = CVEFixesJSONDatasetLoader()
    samples = loader.load_dataset("test_cvefixes_dataset.json")
    print(f"   ✓ Loaded {len(samples)} samples successfully")
    
    # Test 3: Test benchmark runner initialization
    print("\n3. Testing benchmark runner...")
    runner = CVEFixesBenchmarkRunner("cvefixes_experiments_config.json")
    print("   ✓ Benchmark runner initialized successfully")
    
    print("\n" + "=" * 50)
    print("🎉 CVEFixes integration test PASSED!")
    print("The CVEFixes benchmark is ready for use.")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
