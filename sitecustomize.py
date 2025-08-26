import importlib
import os

if os.getenv("KVCACHED_AUTOPATCH", "0").lower() in ("1", "true"):
    try:
        importlib.import_module("kvcached.autopatch")
    except Exception:
        pass
