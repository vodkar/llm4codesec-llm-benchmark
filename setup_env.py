#!/usr/bin/env python3
"""
Environment setup script for the LLM4CodeSec benchmark project.

This script ensures that the src directory is added to the Python path
so imports work correctly without needing 'src.' prefixes.
"""

import sys
import subprocess
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.absolute()
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Verify the setup
def verify_setup():
    """Verify that the environment is set up correctly."""
    try:
        # Test imports
        from benchmark import benchmark_framework
        from datasets.loaders import castle_dataset_loader
        from entrypoints import run_benchmark
        print("✓ Environment setup successful - all imports working")
        return True
    except ImportError as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def run_script():
    """Entry point for poetry scripts."""
    if len(sys.argv) < 2:
        print("Usage: poetry run <script-name> <script-path> [args...]")
        print("Available scripts:")
        print("  benchmark        - Run general benchmark")
        print("  castle-benchmark - Run CASTLE-specific benchmark")
        print("  test-framework   - Run framework tests")
        return
    
    script_path = sys.argv[1]
    script_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Ensure environment is set up
    verify_setup()
    
    # Run the script
    try:
        if script_path.endswith('.py'):
            subprocess.run([sys.executable, script_path] + script_args, check=True)
        else:
            print(f"Error: {script_path} is not a Python script")
    except subprocess.CalledProcessError as e:
        print(f"Script execution failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"Script not found: {script_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run as script runner
        if sys.argv[1] == "-c":
            # Run code directly (for testing)
            code = sys.argv[2]
            print(f"Adding {src_path} to Python path...")
            verify_setup()
            exec(code)
        else:
            # Run script file
            script_path = sys.argv[1]
            script_args = sys.argv[2:] if len(sys.argv) > 2 else []
            print(f"Adding {src_path} to Python path...")
            verify_setup()
            
            # Import and run the script module
            try:
                import importlib.util
                import os
                
                # Get absolute path
                abs_script_path = Path(script_path).resolve()
                
                # Load the module
                spec = importlib.util.spec_from_file_location("__main__", abs_script_path)
                module = importlib.util.module_from_spec(spec)
                
                # Set command line arguments
                original_argv = sys.argv
                sys.argv = [str(abs_script_path)] + script_args
                
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.argv = original_argv
                    
            except Exception as e:
                print(f"Error running script: {e}")
    else:
        print(f"Adding {src_path} to Python path...")
        verify_setup()
