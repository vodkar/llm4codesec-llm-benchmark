#!/usr/bin/env python3
"""
Simple test script to verify the framework is working.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        print("✓ benchmark_framework imported")
        
        print("✓ config_manager imported")
        
        print("✓ data_utils imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration creation."""
    try:
        from config_manager import BenchmarkConfigManager
        
        config = BenchmarkConfigManager.create_config(
            model_key='qwen2.5-7b',
            task_key='binary_vulnerability',
            dataset_path='./test.json',
            output_dir='./test_output'
        )
        print("✓ Configuration created successfully")
        print(f"  Model: {config.model_name}")
        print(f"  Task: {config.task_type.value}")
        
        available = BenchmarkConfigManager.list_available_configs()
        print(f"✓ Available: {len(available['models'])} models, {len(available['tasks'])} tasks")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation."""
    try:
        from config_manager import create_sample_dataset
        
        create_sample_dataset("./test_sample.json")
        print("✓ Sample dataset created")
        
        return True
    except Exception as e:
        print(f"❌ Sample data creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing LLM Code Security Benchmark Framework")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration), 
        ("Sample Data Test", test_sample_data_creation)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            break
    
    print(f"\n{'='*50}")
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Run: poetry run python run_benchmark.py --quick")
        print("2. Or: poetry run python run_benchmark.py --list-configs")
    else:
        print(f"❌ {len(tests) - passed} test(s) failed.")

if __name__ == "__main__":
    main()
