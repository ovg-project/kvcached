# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test: the vLLM offline engine generates text with kvcached enabled.

Skips without vLLM, so the integration manifest stays collectable. The weights
are not optional: the default is a gated repo, so a run without access to it
fails rather than skips. Point KVCACHED_TEST_MODEL at a local or open model to
avoid that. vLLM picks the accelerator itself, so nothing here names a device.
"""
import os

import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.getenv("KVCACHED_TEST_MODEL", "meta-llama/Llama-3.2-1B")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# min_tokens keeps the assertion below honest: at temperature 0.8 the model may
# sample EOS first and return an empty string, which is valid sampling, not a
# kvcached failure.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32,
                                 min_tokens=8)


def test_offline_generate():
    # Create an LLM.
    llm = LLM(
        model=MODEL,
        enable_prefix_caching=False,  # required with kvcached
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(prompts)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        assert len(output.outputs[0].token_ids) >= 8
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    test_offline_generate()
