"""
Read/Write functions used for data in Kipet
"""
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
