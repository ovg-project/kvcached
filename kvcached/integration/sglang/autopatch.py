import logging
import os
import types
from typing import Callable

try:
    from wrapt.importer import when_imported  # type: ignore
except Exception:

    def when_imported(module_name: str):  # type: ignore

        def decorator(func: Callable[[types.ModuleType], None]):
            try:
                __import__(module_name)
            except Exception:
                return func
            func(types.ModuleType(module_name))
            return func

        return decorator


logger = logging.getLogger(__name__)


def _env_enabled() -> bool:
    return os.getenv("KVCACHED_AUTOPATCH", "false").lower() in ("true", "1")


def _enable_kvcached() -> bool:
    return os.getenv("ENABLE_KVCACHED", "false").lower() in ("true", "1")


@when_imported("sglang")
def _patch_sglang(_sglang: types.ModuleType) -> None:
    if not _env_enabled():
        return
    # No monkey patching required yet; sglang integration is implemented via
    # environment-guarded changes in their classes. This hook ensures our
    # kvcached package is imported early, so interfaces are available.
    try:
        import kvcached.integration.sglang.interfaces as _  # noqa: F401
        logger.info("kvcached: sglang interfaces available")
    except Exception:
        logger.warning("kvcached: sglang interfaces not available")
