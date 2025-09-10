# SPDX-FileCopyrightText: 2025 contributors to the kvcached project
#
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def _autopatch_all() -> None:
    # Importing these modules registers their when_imported hooks
    try:
        import_module("kvcached.integration.vllm.autopatch")
    except Exception:
        pass
    try:
        import_module("kvcached.integration.sglang.autopatch")
    except Exception:
        pass


_autopatch_all()
