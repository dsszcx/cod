"""Regression: image encoding and prompt forward share the autocast scope."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.e0e4.segment_sam3 import execute_prompts


def check_device(cuda):
    active = {"amp": False}

    @contextlib.contextmanager
    def autocast(device, dtype):
        assert device == "cuda" and dtype == "bf16"
        active["amp"] = True
        try:
            yield
        finally:
            active["amp"] = False

    class Processor:
        images = 0

        def set_image(self, image):
            assert active["amp"] == cuda, "set_image is outside autocast"
            self.images += 1
            return {"sequence": self.images}

        def result(self, state):
            assert active["amp"] == cuda, "prompt forward is outside autocast"
            assert state["sequence"] == self.images
            return {"masks": np.ones((1, 4, 4), dtype=bool), "scores": [0.9]}

        def add_geometric_prompt(self, box, label, state):
            return self.result(state)

        def set_text_prompt(self, prompt, state):
            return self.result(state)

    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                            bfloat16="bf16", autocast=autocast,
                            inference_mode=contextlib.nullcontext)
    processor = Processor()
    prompts = [{"kind": "box", "box": [0.5, 0.5, 0.2, 0.2]},
               {"kind": "text", "text": "bird"}]
    results = execute_prompts(processor, None, prompts, torch)
    assert processor.images == 2, "each prompt must encode a fresh image"
    assert all(r["status"] == "OK" for r in results)
    assert not active["amp"]


if __name__ == "__main__":
    check_device(True)
    check_device(False)
    print("SAM3 autocast regression passed (CUDA and CPU paths).")
