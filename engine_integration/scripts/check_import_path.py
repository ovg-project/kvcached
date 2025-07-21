import importlib.util as iu
import os
import pprint
import sys

import kvcached
import kvcached.page_allocator as pa

# Where is the project root we expect?
print("project root that should be on sys.path:")
print(repr(os.environ.get("KVCACHED_DIR", "<env var not set>")))

# Show the first 10 entries of sys.path
print("\n--- sys.path (head) ---")
pprint.pp(sys.path[:10])

# Inspect package + module locations
print("\npackage kvcached loaded from:", kvcached.__file__)
print("kvcached.__path__ =", list(kvcached.__path__))

spec = iu.find_spec("kvcached.page_allocator")
print("\npage_allocator will be imported from:", spec.origin)

# Force (re-)import and print its __file__
print("page_allocator actually imported from:", pa.__file__)
