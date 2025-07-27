#!/usr/bin/env python3
"""
Arabic NLP CLI Text Processor
=============================

Command-line interface and SDK wrapper for the Arabic NLP engine system.
"""

import argparse
    import json
    import sys
    from pathlib import Path
    from typing import Dict, Any, List, Optional

# Add parent directory to path,
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core import create_engine, UnifiedArabicEngine, quick_analyze,
    CORE_AVAILABLE = True,
    except ImportError as e:
    CORE_AVAILABLE = False,
    CORE_ERROR = str(e)


class ArabicNLPCLI:
    """Command-line interface for Arabic NLP processing"""

    def __init__(self):
    self.engine = None,
    self.verbose = False,
    def initialize(self, verbose: bool = False):
    """Initialize the CLI processor"""
    self.verbose = verbose,
    if not CORE_AVAILABLE:
    self._print_error(f"Core engine not available: {CORE_ERROR}")
    return False,
    try:
    self.engine = create_engine()
            if self.verbose:
    self._print_success("Arabic NLP engine initialized successfully")
    return True,
    except Exception as e:
    self._print_error(f"Failed to initialize engine: {e}")
    return False,
    def process_text(
    self,
    text: str,
    engines: Optional[List[str]] = None,
    output_format: str = 'json',
    ) -> Dict[str, Any]:
    """Process single text input"""
        if not self.engine:
            if not self.initialize():
    return {'error': 'Engine initialization failed'}

        if self.verbose:
    self._print_info(
    f"Processing text: '{text[:50]}{'...' if len(text) > 50} else ''}"
    )

        try:
    result = self.engine.process_text(text, analysis_types=engines)

            if self.verbose:
    success_status = "✅" if result.get('success') else "❌"
    self._print_info(f"Processing complete {success_status}")

    return result,
    except Exception as e:
    error_result = {'error': str(e), 'input_text': text}
            if self.verbose:
    self._print_error(f"Processing failed: {e}")
    return error_result,
    def process_file(
    self, file_path: str, engines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
    """Process text from a file"""
        try:
    file_path_obj = Path(file_path)
            if not file_path_obj.exists():
    return {'error': f'File not found: {file_path}'}

            with open(file_path_obj, 'r', encoding='utf-8') as f:
    content = f.read().strip()

            if self.verbose:
    self._print_info(
    f"Processing file: {file_path} ({len(content)} characters)"
    )

    return self.process_text(content, engines)

        except Exception as e:
    return {'error': f'File processing failed: {e}'}

    def batch_process(
    self, texts: List[str], engines: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
    """Process multiple texts"""
        if not self.engine:
            if not self.initialize():
    return [{'error': 'Engine initialization failed'}]

        if self.verbose:
    self._print_info(f"Batch processing {len(texts)} texts...")

    results = []
        for i, text in enumerate(texts):
            if self.verbose and i % 10 == 0:
    self._print_info(f"  Processing item {i+1}/{len(texts)}...")

    result = self.process_text(text, engines)
    results.append(result)

        if self.verbose:
    successful = sum(1 for r in results if r.get('success', False))
    self._print_success(
    f"Batch processing complete: {successful}/{len(texts)} successful"
    )

    return results,
    def get_engine_info(self) -> Dict[str, Any]:
    """Get information about available engines"""
        if not self.engine:
            if not self.initialize():
    return {'error': 'Engine initialization failed'}

        try:
    health = self.engine.health_check()
    available_engines = self.engine.get_available_engines()

    return {
    'available_engines': available_engines,
    'engine_count': len(available_engines),
    'health_percentage': health.get('health_percentage', 0),
    'system_status': health.get('overall_health', 'UNKNOWN'),
    'engine_details': health.get('engines', {}),
    }

        except Exception as e:
    return {'error': f'Engine info retrieval failed: {e}'}

    def _print_success(self, message: str):
    """Print success message"""
    print(f"✅ {message}")

    def _print_info(self, message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

    def _print_error(self, message: str):
    """Print error message"""
    print(f"❌ {message}", file=sys.stderr)

    def _print_warning(self, message: str):
    """Print warning message"""
    print(f"⚠️  {message}")


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
    description="Arabic NLP Text Processor - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Process single text,
    python cli_text_processor.py --text "مرحبا بكم"

  # Process with specific engines,
    python cli_text_processor.py --text "كتاب" --engines morphological inflection

  # Process from file,
    python cli_text_processor.py --file input.txt

  # Batch process multiple texts,
    python cli_text_processor.py --batch "كتاب" "قلم" "بيت"

  # Get engine information,
    python cli_text_processor.py --info

  # Pretty print JSON output,
    python cli_text_processor.py --text "كتاب" --pretty
    """,
    )

    # Input options,
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', '-t', type=str, help='Single text to process')
    input_group.add_argument(
    '--file', '-f', type=str, help='Path to file containing text to process'
    )
    input_group.add_argument(
    '--batch', '-b', nargs='+', help='Multiple texts to process'
    )
    input_group.add_argument(
    '--info', '-i', action='store_true', help='Show engine information'
    )

    # Processing options,
    parser.add_argument(
    '--engines',
    '-e',
    nargs='*',
    help='Specific engines to use (default: all available)',
    )

    # Output options,
    parser.add_argument(
    '--output', '-o', type=str, help='Output file path (default: stdout)'
    )
    parser.add_argument(
    '--format',
    choices=['json', 'pretty', 'summary'],
        default='json',
    help='Output format',
    )
    parser.add_argument(
    '--pretty', action='store_true', help='Pretty print JSON output'
    )

    # General options,
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument(
    '--quiet', '-q', action='store_true', help='Suppress all output except results'
    )

    return parser,
    def format_output(data: Any, output_format: str, pretty: bool = False) -> str:
    """Format output data"""
    if output_format == 'json':
        if pretty:
    return json.dumps(data, indent=2, ensure_ascii=False)
        else:
    return json.dumps(data, ensure_ascii=False)

    elif output_format == 'pretty':
    return format_pretty_output(data)

    elif output_format == 'summary':
    return format_summary_output(data)

    else:
    return str(data)


def format_pretty_output(data: Any) -> str:
    """Format data in a human-readable way"""
    if isinstance(data, dict):
        if 'error' in data:
    return f"❌ Error: {data['error']}"

        if 'input_text' in data:
    lines = [f"📝 Input: {data['input_text']}"]

            if 'success' in data:
    status = "✅ Success" if data['success'] else "❌ Failed"
    lines.append(f"📊 Status: {status}")

            if 'analysis_types' in data:
    engines = ", ".join(data['analysis_types'])
    lines.append(f"🔧 Engines: {engines}")

            if 'results' in data and isinstance(data['results'], dict):
    lines.append("📋 Results:")
                for engine, result in data['results'].items():
                    if isinstance(result, dict) and 'confidence' in result:
    conf = result['confidence']
    lines.append(f"  • {engine}: confidence {conf:.2f}")
                    else:
    lines.append(f"  • {engine}: processed")

    return "\n".join(lines)

        if 'available_engines' in data:
    lines = [f"🔧 Available Engines: {len(data['available_engines'])}"]
            for engine in data['available_engines']:
    lines.append(f"  • {engine}")

            if 'health_percentage' in data:
    lines.append(f"❤️ Health: {data['health_percentage']:.1f}%")

    return "\n".join(lines)

    elif isinstance(data, list):
    lines = [f"📚 Batch Results: {len(data)} items"]
    successful = sum(
    1 for item in data if isinstance(item, dict) and item.get('success', False)
    )
    lines.append(f"✅ Successful: {successful}/{len(data)}")
    return "\n".join(lines)

    return json.dumps(data, indent=2, ensure_ascii=False)


def format_summary_output(data: Any) -> str:
    """Format data as a brief summary"""
    if isinstance(data, dict):
        if 'error' in data:
    return f"ERROR: {data['error']}"

        if 'success' in data:
    status = "SUCCESS" if data['success'] else "FAILED"
    text = data.get('input_text', 'N/A')[:30]
    engines = len(data.get('analysis_types', []))
    return f"{status} | Text: {text}... | Engines: {engines}"

    elif isinstance(data, list):
    successful = sum(
    1 for item in data if isinstance(item, dict) and item.get('success', False)
    )
    return f"BATCH: {successful}/{len(data)} successful"

    return str(data)


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle quiet mode,
    if args.quiet:
    args.verbose = False

    # Create CLI processor,
    cli = ArabicNLPCLI()

    # Process based on arguments,
    result = None,
    if args.info:
    result = cli.get_engine_info()

    elif args.text:
        if not cli.initialize(verbose=args.verbose):
    sys.exit(1)
    result = cli.process_text(args.text, args.engines)

    elif args.file:
        if not cli.initialize(verbose=args.verbose):
    sys.exit(1)
    result = cli.process_file(args.file, args.engines)

    elif args.batch:
        if not cli.initialize(verbose=args.verbose):
    sys.exit(1)
    result = cli.batch_process(args.batch, args.engines)

    # Format and output result,
    if result is not None:
        formatted_output = format_output(result, args.format, args.pretty)

        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
    f.write(formatted_output)
                if not args.quiet:
    print(f"✅ Output saved to: {args.output}")
            except Exception as e:
    print(f"❌ Failed to save output: {e}", file=sys.stderr)
    sys.exit(1)
        else:
    print(formatted_output)


if __name__ == "__main__":
    main()
