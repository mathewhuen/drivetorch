import os
import contextlib


@contextlib.contextmanager
def temp_cwd(x):
    d = os.getcwd()
    os.chdir(x)
    try:
        yield
    finally:
        os.chdir(d)
