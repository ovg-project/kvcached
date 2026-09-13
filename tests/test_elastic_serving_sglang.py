# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""End-to-end KV-cache elasticity under load (SGLang offline engine).

The SGLang half of ``test_elastic_serving.py``. kvcached patches both engines'
KV pools, but until this existed only the vLLM path had an end-to-end check
that the mapped footprint actually grows under load and falls when requests
drain -- the SGLang path was covered only by unit tests of the allocator.

``disable_radix_cache`` is the counterpart of vLLM's
``enable_prefix_caching=False`` and is required for the same reason: with
prefix reuse on, finished requests keep their KV resident and no shrink is
observable.

Run inside the SGLang venv with kvcached enabled, from the `tests` directory:
    ENABLE_KVCACHED=true python test_elastic_serving_sglang.py
"""
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import elasticity


class SglangEngine:
    name = "SGLang"

    def __init__(self) -> None:
        import sglang as sgl
        self._engine = sgl.Engine(
            model_path=elasticity.DEFAULT_MODEL,
            disable_cuda_graph=True,  # matches vLLM's enforce_eager
            mem_fraction_static=0.40,
            context_length=8192,
            disable_radix_cache=True,  # else freed KV stays resident
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> List[str]:
        # `seed` is ignored: SGLang has no per-batch equivalent, and it only
        # made the load phase reproducible -- the verdict is decided by the
        # greedy probe request, which is deterministic on both engines.
        params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        }
        outputs = self._engine.generate(list(prompts), params)
        return [o["text"] for o in outputs]


if __name__ == "__main__":
    elasticity.run(SglangEngine)
