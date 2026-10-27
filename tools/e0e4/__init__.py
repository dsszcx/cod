"""E0-E4 inference-only pipeline (frozen 3ch baseline -> ROI -> Qwen -> SAM3 -> fusion).

Import order is deliberate: nothing here imports torch / transformers / sam3 at
module import time, so `test_e0e4.py` runs on a CPU-only, weight-free machine.
"""
__all__ = ["common"]
