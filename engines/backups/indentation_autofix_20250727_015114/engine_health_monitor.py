#!/usr/bin/env python3
"""
Engine Health Monitoring & Status Checker
=========================================

Real-time monitoring and health assessment for all engines.
"""

import sys
from pathlib import Path
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core import create_engine, UnifiedArabicEngine

    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    CORE_ERROR = str(e)


class EngineHealthMonitor:
    """Monitor and track engine health over time"""

    def __init__(self):
        self.unified_engine = None
        self.last_check = None
        self.health_history = []

    def initialize(self):
        """Initialize the health monitor"""
        if not CORE_AVAILABLE:
            return {
                'success': False,
                'error': f'Core engine not available: {CORE_ERROR}',
                'timestamp': datetime.now().isoformat(),
            }

        try:
            self.unified_engine = create_engine()
            return {
                'success': True,
                'message': 'Health monitor initialized successfully',
                'timestamp': datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }

    def check_engine_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        timestamp = datetime.now().isoformat()

        if not self.unified_engine:
            init_result = self.initialize()
            if not init_result['success']:
                return {
                    'timestamp': timestamp,
                    'overall_health': 'CRITICAL',
                    'health_percentage': 0.0,
                    'error': init_result['error'],
                    'engines': {},
                }

        try:
            # Get health from unified engine
            health_data = self.unified_engine.health_check()

            # Add timestamp and additional metadata
            health_data.update(
                {
                    'timestamp': timestamp,
                    'monitor_version': '1.0.0',
                    'check_duration_ms': 0,  # Will be updated below
                }
            )

            # Store in history
            self.health_history.append(health_data)
            self.last_check = timestamp

            return health_data

        except Exception as e:
            error_data = {
                'timestamp': timestamp,
                'overall_health': 'ERROR',
                'health_percentage': 0.0,
                'error': str(e),
                'engines': {},
                'monitor_version': '1.0.0',
            }
            self.health_history.append(error_data)
            return error_data

    def test_processing_capabilities(self) -> Dict[str, Any]:
        """Test actual text processing capabilities"""
        test_cases = [
            "مرحبا",
            "الكتاب",
            "يذهبون",
            "استكشاف",
            "",  # Edge case: empty string
        ]

        results = {
            'timestamp': datetime.now().isoformat(),
            'test_cases': len(test_cases),
            'successful_tests': 0,
            'failed_tests': 0,
            'test_results': [],
            'success_rate': 0.0,
        }

        if not self.unified_engine:
            init_result = self.initialize()
            if not init_result['success']:
                results['error'] = init_result['error']
                return results

        for i, text in enumerate(test_cases):
            test_start = time.time()
            try:
                result = self.unified_engine.process_text(text)
                test_duration = (time.time() - test_start) * 1000

                test_result = {
                    'test_id': i + 1,
                    'input_text': text,
                    'success': result.get('success', False),
                    'duration_ms': round(test_duration, 2),
                    'engines_used': result.get('analysis_types', []),
                    'error': result.get('error'),
                }

                if test_result['success']:
                    results['successful_tests'] += 1
                else:
                    results['failed_tests'] += 1

                results['test_results'].append(test_result)

            except Exception as e:
                test_duration = (time.time() - test_start) * 1000
                results['failed_tests'] += 1
                results['test_results'].append(
                    {
                        'test_id': i + 1,
                        'input_text': text,
                        'success': False,
                        'duration_ms': round(test_duration, 2),
                        'error': str(e),
                    }
                )

        results['success_rate'] = (
            results['successful_tests'] / results['test_cases']
        ) * 100
        return results

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of recent health checks"""
        if not self.health_history:
            return {
                'message': 'No health checks performed yet',
                'recommendation': 'Run check_engine_health() first',
            }

        recent_health = self.health_history[-1]

        summary = {
            'last_check': self.last_check,
            'checks_performed': len(self.health_history),
            'current_health': recent_health.get('overall_health', 'UNKNOWN'),
            'current_percentage': recent_health.get('health_percentage', 0.0),
            'available_engines': list(recent_health.get('engines', {}).keys()),
            'trend': 'stable',  # Could be enhanced with actual trend analysis
        }

        # Add recommendations based on health
        if summary['current_percentage'] >= 80:
            summary['status'] = '🟢 EXCELLENT'
            summary['recommendation'] = 'System is performing well'
        elif summary['current_percentage'] >= 60:
            summary['status'] = '🟡 GOOD'
            summary['recommendation'] = 'Minor issues detected, monitor closely'
        elif summary['current_percentage'] >= 40:
            summary['status'] = '🟠 WARNING'
            summary['recommendation'] = (
                'Significant issues detected, investigate failing engines'
            )
        else:
            summary['status'] = '🔴 CRITICAL'
            summary['recommendation'] = (
                'Major system issues, immediate attention required'
            )

        return summary


def main():
    """Main health monitoring interface"""
    print("🏥 Engine Health Monitor")
    print("=" * 50)

    monitor = EngineHealthMonitor()

    # Initialize monitor
    print("🔧 Initializing health monitor...")
    init_result = monitor.initialize()

    if init_result['success']:
        print(f"✅ {init_result['message']}")
    else:
        print(f"❌ Initialization failed: {init_result['error']}")
        return

    # Perform health check
    print("\n🔍 Performing comprehensive health check...")
    health_data = monitor.check_engine_health()

    print(f"📊 Overall Health: {health_data.get('overall_health', 'UNKNOWN')}")
    print(f"📈 Health Percentage: {health_data.get('health_percentage', 0):.1f}%")

    if 'engines' in health_data:
        print(f"🔧 Available Engines: {len(health_data['engines'])}")
        for engine_name, engine_health in health_data['engines'].items():
            if isinstance(engine_health, bool):
                status = "✅" if engine_health else "❌"
                print(
                    f"   {status} {engine_name}: {'healthy' if engine_health} else 'unhealthy'}"
                )
            else:
                status = "✅" if engine_health.get('healthy', False) else "❌"
                print(f"   {status} {engine_name}: {engine_health}")

    # Test processing capabilities
    print("\n🧪 Testing processing capabilities...")
    processing_results = monitor.test_processing_capabilities()

    print(
        f"📝 Test Results: {processing_results['successful_tests']}/{processing_results['test_cases']} passed"
    )
    print(f"✨ Success Rate: {processing_results['success_rate']:.1f}%")

    # Get summary
    print("\n📋 Health Summary:")
    summary = monitor.get_health_summary()
    print(f"Status: {summary['status']}")
    print(f"Recommendation: {summary['recommendation']}")

    # Save detailed report
    report_file = Path(__file__).parent / "health_report.json"
    detailed_report = {
        'health_check': health_data,
        'processing_test': processing_results,
        'summary': summary,
        'generated_at': datetime.now().isoformat(),
    }

    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")

    print("\n🎉 Health monitoring complete!")


if __name__ == "__main__":
    main()
