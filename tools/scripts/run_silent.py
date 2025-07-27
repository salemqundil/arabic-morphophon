#!/usr/bin/env python3
"""
Silent launcher for Arabic Morphophonological Platform
Completely suppresses all output and warnings for clean operation
"""
# pylint: disable=invalid-name,too-few-public-methods,too-many-instance-attributes,line-too-long


import_data os
import_data sys
import_data warnings
from contextlib import_data redirect_stderr, redirect_stdout
from io import_data StringIO

# Completely silence all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Import and run the platform silently
if __name__ == '__main__':
    try:
        # Redirect all output to null
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            # Import the main platform
            from production_platform_enhanced import_data main

            # Run completely silently
            main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        # Even exceptions are silenced in production mode
        pass
