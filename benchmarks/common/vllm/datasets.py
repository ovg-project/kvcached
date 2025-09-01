from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

from transformers import PreTrainedTokenizerBase

# Pull the canonical base classes / helpers from the installed vLLM package.
from vllm.benchmarks.datasets import BenchmarkDataset, SampleRequest, is_valid_sequence

__all__ = ["MyCustomDataset"]


class MyCustomDataset(BenchmarkDataset):
    """Simple dataset that reads a JSONL file containing a ``prompt`` field.

    Optionally a ``completion`` field can be present; if so we can plug it in to
    derive the expected output length.  Otherwise a fixed ``output_len`` will
    be used by the caller.
    """

    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.load_data()

    # ---------------------------------------------------------------------
    # Required interface ---------------------------------------------------
    # ---------------------------------------------------------------------
    def load_data(self) -> None:  # noqa: D401 – imperative mood
        """Populate ``self.data`` with a list of dicts taken from the JSONL."""
        path = Path(self.dataset_path)
        if not path.exists():
            raise FileNotFoundError(path)

        self.data: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.data.append(json.loads(line))

        # Deterministic order for reproducibility.
        random.seed(self.random_seed)
        random.shuffle(self.data)

    def sample(
        self,
        tokenizer: PreTrainedTokenizerBase,
        num_requests: int,
        output_len: Optional[int] = None,
        request_id_prefix: str = "",
        **kwargs,
    ) -> List[SampleRequest]:
        """Return ``num_requests`` SampleRequest objects.

        If ``output_len`` is *None* and a ``completion`` field is present in a
        row, we compute its token length dynamically so the benchmark expects the
        full ground-truth generation.  Otherwise we fall back to the fixed
        ``output_len`` provided by the caller.
        """
        requests: List[SampleRequest] = []
        dynamic_output = output_len is None

        for idx, row in enumerate(self.data):
            if len(requests) >= num_requests:
                break

            prompt: str = row["prompt"]
            completion: str = row.get("completion", "")

            prompt_tok = tokenizer(prompt)
            prompt_len = len(prompt_tok.input_ids)

            exp_output_len: int
            if dynamic_output:
                if completion:
                    exp_output_len = len(
                        tokenizer(completion, add_special_tokens=False).input_ids
                    )
                else:
                    # If no completion & dynamic -> skip this sample.
                    continue
            else:
                exp_output_len = output_len  # type: ignore[arg-type]

            if not is_valid_sequence(prompt_len, exp_output_len):
                # Skip pathological lengths.
                continue

            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=exp_output_len,
                    request_id=f"{request_id_prefix}{idx}",
                )
            )

        # If we have fewer than needed, oversample to pad.
        self.maybe_oversample_requests(requests, num_requests, request_id_prefix)
        return requests
