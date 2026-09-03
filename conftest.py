"""Ensures the repo root is on sys.path so `import options_pricing` works
when pytest is invoked from within tests/ or elsewhere."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
