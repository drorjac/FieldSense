"""Settings for the whole test suite, loaded by pytest before any test module.

torch and pysteps each bring an OpenMP runtime. On macOS, once torch has run in a
process, pysteps' multi-threaded VET segfaults; one OpenMP thread avoids it. The
variable must be set before either library loads.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
