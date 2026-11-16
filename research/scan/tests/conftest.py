"""Pin the BLAS thread count before numpy loads anywhere in the tests.

Multithreaded OpenBLAS busy-spins at every tiny matmul on Apple
silicon (measured 12x slower encode on an M4). scan's __init__ pins
it for the package, but pytest imports test modules (and numpy) in
alphabetical order, so pin here first.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")