#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""
Regression test: two concurrent wakeup_model() calls for the same sleeping
model must both report success when both of their own upstream wake calls
succeed.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "controller"))

from controller.sleep_manager import SleepConfig, SleepManager


async def test_concurrent_wakeup_both_succeed():
    model_name = "test-model"
    config = SleepConfig(
        min_sleep_duration=0,
        vllm_models_config={
            model_name: {
                "host": "localhost",
                "port": "8000"
            }
        },
    )
    manager = SleepManager(config)
    manager.sleeping_models[model_name] = time.time() - 10

    async def slow_wake(host, port):
        await asyncio.sleep(0.05)
        return True

    manager._call_vllm_wakeup_api = slow_wake  # type: ignore[method-assign]

    results = await asyncio.gather(
        manager.wakeup_model(model_name),
        manager.wakeup_model(model_name),
    )

    assert results == [
        True, True
    ], f"expected both racers to report success, got {results}"
    assert model_name not in manager.sleeping_models


async def main():
    await test_concurrent_wakeup_both_succeed()
    print("OK: concurrent wakeup_model() calls both report success")


if __name__ == "__main__":
    asyncio.run(main())
