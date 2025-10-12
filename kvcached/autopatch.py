# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def autopatch_all() -> None:
    # Importing these modules registers their when_imported hooks
    try:
        import_module("kvcached.integration.vllm.autopatch")
    except Exception:
        pass
    try:
        import_module("kvcached.integration.sglang.autopatch")
    except Exception:
        pass


autopatch_all()
