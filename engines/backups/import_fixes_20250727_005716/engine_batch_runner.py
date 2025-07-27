#!/usr/bin/env python3
"""
Engine Batch Test Runner
========================

Automated batch processing and testing framework for Arabic NLP engines.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core import create_engine, UnifiedArabicEngine

    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    CORE_ERROR = str(e)


class BatchTestRunner:
    """Automated batch testing framework"""

    def __init__(self):
    self.engine = None
    self.results = []
    self.start_time = None
    self.end_time = None

    def initialize(self):
    """Initialize the batch runner"""
        if not CORE_AVAILABLE:
    return {
    'success': False,
    'error': f'Core engine not available: {CORE_ERROR}',
    }

        try:
    self.engine = create_engine()
    return {'success': True, 'message': 'Batch runner initialized'}
        except Exception as e:
    return {'success': False, 'error': str(e)}

    def run_batch_tests(
    self, test_texts: List[str], engines: List[str] = None
    ) -> Dict[str, Any]:
    """Run batch tests on multiple texts"""
    self.start_time = time.time()

        if not self.engine:
    init_result = self.initialize()
            if not init_result['success']:
    return init_result

        # Use all available engines if none specified
        if engines is None:
    engines = self.engine.get_available_engines()

    batch_results = {
    'start_time': datetime.now().isoformat(),
    'test_count': len(test_texts),
    'engines_tested': engines,
    'results': [],
    'statistics': {
    'total_tests': 0,
    'successful_tests': 0,
    'failed_tests': 0,
    'success_rate': 0.0,
    'avg_processing_time': 0.0,
    'total_processing_time': 0.0,
    },
    }

    total_processing_time = 0.0

        for i, text in enumerate(test_texts):
    test_start = time.time()

            try:
    result = self.engine.process_text(text, analysis_types=engines)
    test_duration = time.time() - test_start
    total_processing_time += test_duration

    test_result = {
    'test_id': i + 1,
    'input_text': text,
    'success': result.get('success', False),
    'processing_time_ms': round(test_duration * 1000, 2),
    'engines_used': result.get('analysis_types', []),
    'result_summary': self._summarize_result(result),
    'error': result.get('error'),
    }

    batch_results['statistics']['total_tests'] += 1
                if test_result['success']:
    batch_results['statistics']['successful_tests'] += 1
                else:
    batch_results['statistics']['failed_tests'] += 1

    batch_results['results'].append(test_result)

            except Exception as e:
    test_duration = time.time() - test_start
    total_processing_time += test_duration

    batch_results['statistics']['total_tests'] += 1
    batch_results['statistics']['failed_tests'] += 1

    batch_results['results'].append(
    {
    'test_id': i + 1,
    'input_text': text,
    'success': False,
    'processing_time_ms': round(test_duration * 1000, 2),
    'error': str(e),
    }
    )

        # Calculate final statistics
    stats = batch_results['statistics']
        if stats['total_tests'] > 0:
    stats['success_rate'] = (
    stats['successful_tests'] / stats['total_tests']
    ) * 100
    stats['avg_processing_time'] = round(
    total_processing_time / stats['total_tests'] * 1000, 2
    )

    stats['total_processing_time'] = round(total_processing_time * 1000, 2)

    self.end_time = time.time()
    batch_results['end_time'] = datetime.now().isoformat()
    batch_results['total_duration_ms'] = round(
    (self.end_time - self.start_time) * 1000, 2
    )

    return batch_results

    def _summarize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of the analysis result"""
    summary = {
    'engines_count': len(result.get('analysis_types', [])),
    'has_results': len(result.get('results', {})) > 0,
    'confidence_avg': 0.0,
    }

        # Calculate average confidence if available
    results = result.get('results', {})
    confidences = []
        for engine_result in results.values():
            if isinstance(engine_result, dict) and 'confidence' in engine_result:
    confidences.append(engine_result['confidence'])

        if confidences:
    summary['confidence_avg'] = round(sum(confidences) / len(confidences), 2)

    return summary

    def run_performance_benchmark(self, iterations: int = 100) -> Dict[str, Any]:
    """Run performance benchmark tests"""
    print(f"🏃 Running performance benchmark ({iterations} iterations)...")

        # Standard test text for consistent benchmarking
    test_text = "كتاب جميل في اللغة العربية"

    benchmark_results = {
    'iterations': iterations,
    'test_text': test_text,
    'start_time': datetime.now().isoformat(),
    'individual_times': [],
    'statistics': {},
    }

        for i in range(iterations):
    start = time.time()
    result = self.engine.process_text(test_text)
    duration = (time.time() - start) * 1000  # Convert to ms
    benchmark_results['individual_times'].append(round(duration, 2))

            if (i + 1) % 10 == 0:
    print(f"  Completed {i} + 1}/{iterations} iterations...")

        # Calculate statistics
    times = benchmark_results['individual_times']
    benchmark_results['statistics'] = {
    'min_time_ms': min(times),
    'max_time_ms': max(times),
    'avg_time_ms': round(sum(times) / len(times), 2),
    'total_time_ms': round(sum(times), 2),
    'throughput_per_second': round(1000 / (sum(times) / len(times)), 2),
    }

    benchmark_results['end_time'] = datetime.now().isoformat()
    return benchmark_results

    def run_stress_test(self, concurrent_requests: int = 50) -> Dict[str, Any]:
    """Run stress test with multiple concurrent-like requests"""
    print(f"⚡ Running stress test ({concurrent_requests} rapid requests)...")

    test_texts = [
    f"اختبار رقم {i} للضغط على النظام" for i in range(concurrent_requests)
    ]

    start_time = time.time()
    results = self.run_batch_tests(test_texts)
    end_time = time.time()

    stress_results = {
    'concurrent_requests': concurrent_requests,
    'total_duration_ms': round((end_time - start_time) * 1000, 2),
    'requests_per_second': round(
    concurrent_requests / (end_time - start_time), 2
    ),
    'batch_results': results,
    'system_stability': (
    'stable' if results['statistics']['success_rate'] > 95 else 'unstable'
    ),
    }

    return stress_results


def create_arabic_test_corpus() -> List[str]:
    """Create a comprehensive Arabic test corpus"""
    return [
        # Basic words
    "كتاب",
    "قلم",
    "بيت",
    "شجرة",
    "نهر",
        # Verbs in different forms
    "يكتب",
    "كتب",
    "اكتب",
    "مكتوب",
        # Complex morphology
    "استكشاف",
    "استخراج",
    "تفسير",
    "مناقشة",
        # Sentences
    "الكتاب على الطاولة",
    "الطالب يدرس في المكتبة",
    "هذا كتاب جميل ومفيد جداً",
        # Different lengths
    "أ",
    "في",
    "من",
    "إلى",
    "الجمهورية العربية السورية",
        # With diacritics
    "كِتَابٌ جَمِيلٌ",
    "الطَّالِبُ المُجْتَهِدُ",
        # Mixed content
    "كتاب 2024",
    "الفصل الأول",
    "صفحة 15",
        # Edge cases
    "",
    "   ",
    "123",
    "abc",
    ]


def main():
    """Main batch testing interface"""
    print("🚀 Arabic NLP Engine Batch Test Runner")
    print("=" * 60)

    runner = BatchTestRunner()

    # Initialize
    print("🔧 Initializing batch test runner...")
    init_result = runner.initialize()

    if not init_result['success']:
    print(f"❌ Initialization failed: {init_result['error']}")
    return

    print(f"✅ {init_result['message']}")

    # Create test corpus
    test_corpus = create_arabic_test_corpus()
    print(f"📚 Created test corpus with {len(test_corpus)} entries")

    # Run batch tests
    print("\n🧪 Running batch tests...")
    batch_results = runner.run_batch_tests(test_corpus)

    # Display results
    stats = batch_results['statistics']
    print(f"📊 Batch Test Results:")
    print(f"  Total Tests: {stats['total_tests']}")
    print(f"  Successful: {stats['successful_tests']}")
    print(f"  Failed: {stats['failed_tests']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Avg Processing Time: {stats['avg_processing_time']:.1f}ms")
    print(f"  Total Duration: {batch_results['total_duration_ms']:.1f}ms")

    # Run performance benchmark
    print("\n🏃 Running performance benchmark...")
    benchmark_results = runner.run_performance_benchmark(iterations=20)

    bench_stats = benchmark_results['statistics']
    print(f"📈 Performance Results:")
    print(f"  Average Time: {bench_stats['avg_time_ms']:.1f}ms")
    print(f"  Min Time: {bench_stats['min_time_ms']:.1f}ms")
    print(f"  Max Time: {bench_stats['max_time_ms']:.1f}ms")
    print(f"  Throughput: {bench_stats['throughput_per_second']:.1f} requests/second")

    # Run stress test
    print("\n⚡ Running stress test...")
    stress_results = runner.run_stress_test(concurrent_requests=25)

    print(f"🔥 Stress Test Results:")
    print(f"  Requests/Second: {stress_results['requests_per_second']:.1f}")
    print(f"  System Stability: {stress_results['system_stability']}")
    print(
    f"  Success Rate: {stress_results['batch_results']['statistics']['success_rate']:.1f}%"
    )

    # Save detailed report
    report_file = Path(__file__).parent / "batch_test_report.json"
    full_report = {
    'batch_tests': batch_results,
    'performance_benchmark': benchmark_results,
    'stress_test': stress_results,
    'generated_at': datetime.now().isoformat(),
    }

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed report saved to: {report_file}")
    except Exception as e:
    print(f"\n⚠️ Could not save report: {e}")

    print("\n🎉 Batch testing complete!")


if __name__ == "__main__":
    main()
