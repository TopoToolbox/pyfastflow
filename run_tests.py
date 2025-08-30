#!/usr/bin/env python3
"""
Test runner script for PyFastFlow.

This script provides convenient shortcuts for running different test suites.
"""
import sys
import subprocess
import argparse


def run_command(cmd, description=None):
    """Run a command and return the result."""
    if description:
        print(f"→ {description}")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="PyFastFlow test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --imports          # Run only import tests
  python run_tests.py --unit             # Run only unit tests  
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --all              # Run all tests
  python run_tests.py --fast             # Run only fast tests
  python run_tests.py --verbose          # Run with verbose output
        """
    )
    
    parser.add_argument('--imports', action='store_true',
                       help='Run import tests only')
    parser.add_argument('--unit', action='store_true', 
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--fast', action='store_true',
                       help='Run fast tests only (exclude slow tests)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage report')
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = "PYTHONPATH=. python -m pytest"
    
    # Add verbosity
    if args.verbose:
        base_cmd += " -v"
    else:
        base_cmd += " -q"
        
    # Add coverage if requested
    if args.coverage:
        base_cmd += " --cov=pyfastflow --cov-report=html --cov-report=term"
    
    # Add warnings suppression for cleaner output
    base_cmd += " --disable-warnings"
    
    success = True
    
    if args.imports:
        cmd = f"{base_cmd} tests/test_imports.py"
        success = run_command(cmd, "Running import tests")
        
    elif args.unit:
        cmd = f"{base_cmd} tests/unit/"
        success = run_command(cmd, "Running unit tests")
        
    elif args.integration:
        cmd = f"{base_cmd} tests/integration/"
        success = run_command(cmd, "Running integration tests")
        
    elif args.fast:
        cmd = f"{base_cmd} -m 'not slow'"
        success = run_command(cmd, "Running fast tests")
        
    elif args.all:
        # Run all tests
        print("Running complete test suite...")
        
        cmd = f"{base_cmd} tests/test_imports.py"
        if not run_command(cmd, "Import tests"):
            success = False
            
        cmd = f"{base_cmd} tests/unit/"
        if not run_command(cmd, "Unit tests"):
            success = False
            
        cmd = f"{base_cmd} tests/integration/"
        if not run_command(cmd, "Integration tests"):
            success = False
            
    else:
        # Default: run import tests and unit tests
        cmd = f"{base_cmd} tests/test_imports.py tests/unit/"
        success = run_command(cmd, "Running basic test suite (imports + unit tests)")
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())