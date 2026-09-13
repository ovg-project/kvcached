# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""End-to-end KV-cache elasticity under load (vLLM offline engine).

Complements ``test_kvcache_manager.py`` (which exercises the manager-level
``resize``/``trim`` APIs directly) by driving the *real* engine and watching the
physically mapped KV footprint grow and shrink through the /dev/shm IPC that
``kvtop``/``kvctl`` read. The phases and the verdict live in
``helpers/elasticity.py``; ``test_elastic_serving_sglang.py`` is the same run
against the other engine.

Validated on AMD MI300X (ROCm/HIP) to confirm the hipMemMap (grow) and
hipMemUnmap (shrink) paths; runs on NVIDIA too.

Run inside the vLLM venv with kvcached enabled, from the `tests` directory:
    ENABLE_KVCACHED=true VLLM_USE_V1=1 python test_elastic_serving.py
"""
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import elasticity


class VllmEngine:
    name = "vLLM"

    def __init__(self) -> None:
        from vllm import LLM
        self._llm = LLM(
            model=elasticity.DEFAULT_MODEL,
            enforce_eager=True,
            gpu_memory_utilization=0.40,
            max_model_len=8192,
            enable_prefix_caching=False,  # else freed KV stays resident
            disable_log_stats=True,
        )

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int,
        temperature: float,
        seed: Optional[int] = None,
    ) -> List[str]:
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature,
                                max_tokens=max_tokens,
                                seed=seed)
        return [o.outputs[0].text for o in self._llm.generate(prompts, params)]


if __name__ == "__main__":
    elasticity.run(VllmEngine)
