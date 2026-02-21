# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Automatic patching for kvcached integration with LLM serving engines.

This module registers import hooks for vLLM and SGLang to enable kvcached's
elastic KV cache management when these libraries are loaded.
"""

import logging
from importlib import import_module

logger = logging.getLogger(__name__)


def autopatch_all() -> None:
    """Register import hooks for all supported serving engines.

    This function attempts to import the autopatch modules for vLLM and SGLang,
    which register their respective import hooks. Failures are logged at debug
    level since not all engines may be installed.
    """
    # Importing these modules registers their when_imported hooks
    try:
        import_module("kvcached.integration.vllm.autopatch")
        logger.debug("Registered vLLM autopatch hooks")
    except ImportError:
        logger.debug("vLLM autopatch not available (vLLM may not be installed)")
    except Exception as e:
        logger.warning("Failed to register vLLM autopatch hooks: %s", e)

    try:
        import_module("kvcached.integration.sglang.autopatch")
        logger.debug("Registered SGLang autopatch hooks")
    except ImportError:
        logger.debug("SGLang autopatch not available (SGLang may not be installed)")
    except Exception as e:
        logger.warning("Failed to register SGLang autopatch hooks: %s", e)


autopatch_all()
