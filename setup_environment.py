#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project Environment Setup - Auto-run on startup"""
import os
import sys

# Set permanent encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"

# Ensure UTF-8 for all operations
if sys.stdout.encoding.lower() != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

print("✅ Project encoding configured: UTF-8")
