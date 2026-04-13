#!/usr/bin/env python3
"""
Zero-Copy Optimization Test Runner

This script runs comprehensive tests for the zero-copy tensor optimization system,
including unit tests, integration tests, performance benchmarks, and correctness validation.

Usage:
    python tests/run_zero_copy_tests.py [options]

Options:
    --quick          Run only unit tests (fastest)
    --full           Run all tests including performance benchmarks (default)
    --performance    Run only performance benchmarks
    --gpu            Include GPU tests (requires CUDA)
    --verbose        Verbose output
    --no-cleanup     Don't cleanup test artifacts
"""

import os
import sys
import argparse
import unittest
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

# Import test modules
from yinsh_ml.tests.test_zero_copy_core import *
from yinsh_ml.tests.test_zero_copy_training import *
from yinsh_ml.tests.test_zero_copy_gpu import *
from yinsh_ml.tests.test_zero_copy_performance import run_performance_suite


class ZeroCopyTestRunner:
    """Comprehensive test runner for zero-copy optimizations."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.results = {
            'test_suites': {},
            'summary': {},
            'environment': self._get_environment_info(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _get_environment_info(self):
        """Collect environment information."""
        return {
            'python_version': sys.version,
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'platform': sys.platform
        }
    
    def run_unit_tests(self):
        """Run unit tests for core zero-copy functionality."""
        print("\n" + "="*70)
        print("RUNNING UNIT TESTS - Core Zero-Copy Functionality")
        print("="*70)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add core functionality tests
        from yinsh_ml.tests.test_zero_copy_core import (
            TestZeroCopyConfig, TestZeroCopyTensorFactory, TestInPlaceOperations,
            TestZeroCopyBatchProcessor, TestGlobalFunctions, TestZeroCopyContext,
            TestStatistics
        )
        
        test_classes = [
            TestZeroCopyConfig, TestZeroCopyTensorFactory, TestInPlaceOperations,
            TestZeroCopyBatchProcessor, TestGlobalFunctions, TestZeroCopyContext,
            TestStatistics
        ]
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.args.verbose else 1,
            stream=sys.stdout
        )
        result = runner.run(suite)
        
        # Store results
        self.results['test_suites']['unit_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failure_details': [str(f[1]) for f in result.failures],
            'error_details': [str(e[1]) for e in result.errors]
        }
        
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests for training and state conversion."""
        print("\n" + "="*70)
        print("RUNNING INTEGRATION TESTS - Training & State Conversion")
        print("="*70)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add integration tests
        from yinsh_ml.tests.test_zero_copy_training import (
            TestZeroCopyGameExperience, TestOptimizedYinshTrainer,
            TestOptimizedStateConverter, TestGlobalOptimizedFunctions
        )
        
        test_classes = [
            TestZeroCopyGameExperience, TestOptimizedYinshTrainer,
            TestOptimizedStateConverter, TestGlobalOptimizedFunctions
        ]
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.args.verbose else 1,
            stream=sys.stdout
        )
        result = runner.run(suite)
        
        # Store results
        self.results['test_suites']['integration_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failure_details': [str(f[1]) for f in result.failures],
            'error_details': [str(e[1]) for e in result.errors]
        }
        
        return result.wasSuccessful()
    
    def run_gpu_tests(self):
        """Run GPU optimization tests."""
        if not torch.cuda.is_available() and not self.args.gpu:
            print("\n⚠️  Skipping GPU tests (CUDA not available)")
            return True
        
        print("\n" + "="*70)
        print("RUNNING GPU TESTS - GPU Optimizations")
        print("="*70)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add GPU tests
        from yinsh_ml.tests.test_zero_copy_gpu import (
            TestGPUTransferConfig, TestZeroCopyTensorWrapper, TestPinnedMemoryPool,
            TestSmartPlacementOptimizer, TestZeroCopyGPUManager, TestGlobalFunctions,
            TestGPUOptimizationContext
        )
        
        test_classes = [
            TestGPUTransferConfig, TestZeroCopyTensorWrapper, TestPinnedMemoryPool,
            TestSmartPlacementOptimizer, TestZeroCopyGPUManager, TestGlobalFunctions,
            TestGPUOptimizationContext
        ]
        
        # Add GPU-specific tests only if CUDA is available
        if torch.cuda.is_available():
            from yinsh_ml.tests.test_zero_copy_gpu import TestAsyncTransferManager
            test_classes.append(TestAsyncTransferManager)
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.args.verbose else 1,
            stream=sys.stdout
        )
        result = runner.run(suite)
        
        # Store results
        self.results['test_suites']['gpu_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failure_details': [str(f[1]) for f in result.failures],
            'error_details': [str(e[1]) for e in result.errors]
        }
        
        return result.wasSuccessful()
    
    def run_performance_tests(self):
        """Run performance benchmarks."""
        print("\n" + "="*70)
        print("RUNNING PERFORMANCE BENCHMARKS")
        print("="*70)
        
        try:
            # Run performance benchmark suite
            success = run_performance_suite()
            
            # Get zero-copy statistics
            try:
                from yinsh_ml.memory.zero_copy import get_zero_copy_statistics
                stats = get_zero_copy_statistics()
                
                self.results['test_suites']['performance_tests'] = {
                    'success': success,
                    'zero_copy_stats': {
                        'shared_memory_allocations': stats['tensor_factory'].shared_memory_allocations,
                        'buffer_reuses': stats['tensor_factory'].buffer_reuses,
                        'copy_avoided_count': stats['tensor_factory'].copy_avoided_count,
                        'inplace_operations': stats['inplace_operations'].inplace_operations,
                        'fallback_operations': stats['inplace_operations'].fallback_operations
                    }
                }
            except Exception as e:
                print(f"Warning: Could not collect zero-copy statistics: {e}")
                self.results['test_suites']['performance_tests'] = {
                    'success': success,
                    'zero_copy_stats': None
                }
            
            return success
            
        except Exception as e:
            print(f"Performance tests failed: {e}")
            self.results['test_suites']['performance_tests'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_correctness_validation(self):
        """Run correctness validation tests."""
        print("\n" + "="*70)
        print("RUNNING CORRECTNESS VALIDATION")
        print("="*70)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Add correctness tests
        from yinsh_ml.tests.test_zero_copy_performance import (
            TestCorrectnessValidation, TestRegressionValidation
        )
        
        test_classes = [TestCorrectnessValidation, TestRegressionValidation]
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.args.verbose else 1,
            stream=sys.stdout
        )
        result = runner.run(suite)
        
        # Store results
        self.results['test_suites']['correctness_validation'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.wasSuccessful(),
            'failure_details': [str(f[1]) for f in result.failures],
            'error_details': [str(e[1]) for e in result.errors]
        }
        
        return result.wasSuccessful()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        end_time = time.time()
        self.results['total_runtime'] = end_time - self.start_time
        
        # Calculate summary statistics
        total_tests = sum(
            suite.get('tests_run', 0) 
            for suite in self.results['test_suites'].values()
            if isinstance(suite, dict) and 'tests_run' in suite
        )
        total_failures = sum(
            suite.get('failures', 0) 
            for suite in self.results['test_suites'].values()
            if isinstance(suite, dict) and 'failures' in suite
        )
        total_errors = sum(
            suite.get('errors', 0) 
            for suite in self.results['test_suites'].values()
            if isinstance(suite, dict) and 'errors' in suite
        )
        
        overall_success = all(
            suite.get('success', False) 
            for suite in self.results['test_suites'].values()
            if isinstance(suite, dict) and 'success' in suite
        )
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'overall_success': overall_success,
            'success_rate': (total_tests - total_failures - total_errors) / max(total_tests, 1) * 100
        }
        
        # Print report
        print("\n" + "="*80)
        print("ZERO-COPY OPTIMIZATION TEST REPORT")
        print("="*80)
        
        print(f"🕐 Total Runtime: {self.results['total_runtime']:.2f} seconds")
        print(f"🧪 Total Tests: {total_tests}")
        print(f"✅ Passed: {total_tests - total_failures - total_errors}")
        print(f"❌ Failed: {total_failures}")
        print(f"🔥 Errors: {total_errors}")
        print(f"📊 Success Rate: {self.results['summary']['success_rate']:.1f}%")
        
        print(f"\n🌍 Environment:")
        env = self.results['environment']
        print(f"   Python: {env['python_version'].split()[0]}")
        print(f"   PyTorch: {env['torch_version']}")
        print(f"   NumPy: {env['numpy_version']}")
        print(f"   CUDA: {env['cuda_available']} ({env['cuda_version'] if env['cuda_available'] else 'N/A'})")
        print(f"   Platform: {env['platform']}")
        
        print(f"\n📋 Test Suite Results:")
        for suite_name, suite_results in self.results['test_suites'].items():
            if isinstance(suite_results, dict):
                if 'tests_run' in suite_results:
                    status = "✅ PASS" if suite_results['success'] else "❌ FAIL"
                    print(f"   {suite_name}: {status} ({suite_results['tests_run']} tests)")
                elif 'success' in suite_results:
                    status = "✅ PASS" if suite_results['success'] else "❌ FAIL"
                    print(f"   {suite_name}: {status}")
        
        # Print failures and errors
        if total_failures > 0 or total_errors > 0:
            print(f"\n🚨 Issues Found:")
            for suite_name, suite_results in self.results['test_suites'].items():
                if isinstance(suite_results, dict):
                    if suite_results.get('failure_details'):
                        print(f"\n   {suite_name} Failures:")
                        for detail in suite_results['failure_details'][:3]:  # Limit output
                            print(f"     - {detail[:200]}...")
                    if suite_results.get('error_details'):
                        print(f"\n   {suite_name} Errors:")
                        for detail in suite_results['error_details'][:3]:  # Limit output
                            print(f"     - {detail[:200]}...")
        
        # Print zero-copy statistics if available
        perf_results = self.results['test_suites'].get('performance_tests', {})
        if perf_results.get('zero_copy_stats'):
            stats = perf_results['zero_copy_stats']
            print(f"\n📊 Zero-Copy Statistics:")
            print(f"   Shared Memory Allocations: {stats['shared_memory_allocations']}")
            print(f"   Buffer Reuses: {stats['buffer_reuses']}")
            print(f"   Copies Avoided: {stats['copy_avoided_count']}")
            print(f"   In-Place Operations: {stats['inplace_operations']}")
            print(f"   Fallback Operations: {stats['fallback_operations']}")
        
        # Save detailed report
        if not self.args.no_cleanup:
            report_file = project_root / "tests" / f"zero_copy_test_report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Detailed report saved to: {report_file}")
        
        return overall_success
    
    def run(self):
        """Run the selected test suite."""
        overall_success = True
        
        if self.args.quick:
            print("🚀 Running quick test suite (unit tests only)")
            overall_success &= self.run_unit_tests()
            
        elif self.args.performance:
            print("⚡ Running performance benchmarks only")
            overall_success &= self.run_performance_tests()
            
        else:  # full
            print("🔍 Running full test suite")
            overall_success &= self.run_unit_tests()
            overall_success &= self.run_integration_tests()
            
            if self.args.gpu or torch.cuda.is_available():
                overall_success &= self.run_gpu_tests()
            
            overall_success &= self.run_correctness_validation()
            overall_success &= self.run_performance_tests()
        
        # Generate report
        report_success = self.generate_report()
        
        return overall_success and report_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run zero-copy optimization tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Run only unit tests (fastest)')
    parser.add_argument('--full', action='store_true', default=True,
                        help='Run all tests including performance benchmarks (default)')
    parser.add_argument('--performance', action='store_true',
                        help='Run only performance benchmarks')
    parser.add_argument('--gpu', action='store_true',
                        help='Include GPU tests (requires CUDA)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--no-cleanup', action='store_true',
                        help="Don't cleanup test artifacts")
    
    args = parser.parse_args()
    
    # Adjust defaults based on exclusive options
    if args.quick or args.performance:
        args.full = False
    
    print("🧪 Zero-Copy Optimization Test Suite")
    print(f"📁 Project Root: {project_root}")
    
    # Run tests
    runner = ZeroCopyTestRunner(args)
    success = runner.run()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main() 