"""Conftest specifically for heuristic selfplay tests."""

import os
import sys

# Set this BEFORE any imports that might trigger NumPy
os.environ['NPY_DISABLE_MACOS_ACCELERATE'] = '1'

# Force the workaround
if sys.platform == 'darwin':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

