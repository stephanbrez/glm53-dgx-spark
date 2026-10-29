#!/usr/bin/env python3
"""Patch srt/platforms/cuda.py: env-gate is_pin_memory_available.

SGLANG_DISABLE_PIN_MEMORY=1 -> False (pinning disabled). Default unchanged.

Why (GB10/UMA, 2x DGX Spark): the weight loader pins host staging tensors;
freed pinned buffers stay resident in torch's CachingHostAllocator and are
invisible to sglang's memory accounting. On unified memory they are pure waste
(H2D is a memcpy within LPDDR) and accumulate ~15GB+ per rank, which — on top
of the ~95GB/rank of device-resident weights — tips the 121.8GB host into
GLOBAL OOM at end-of-load. Self-verifying: aborts unless the block is unique.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / "orig" / "cuda.py"
DST = HERE / "cuda.py.sm121.py"

OLD = (
    '    def is_pin_memory_available(self, device=None) -> bool:\n'
    '        if device is not None and str(device) == "cpu":\n'
    '            return False\n'
    '        return True\n'
)
NEW = (
    '    def is_pin_memory_available(self, device=None) -> bool:\n'
    '        if device is not None and str(device) == "cpu":\n'
    '            return False\n'
    '        if os.environ.get("SGLANG_DISABLE_PIN_MEMORY", "0") == "1":\n'
    '            return False\n'
    '        return True\n'
)

t = SRC.read_text()
if t.count(OLD) != 1:
    sys.exit(f"expected exactly one pin-gate block, found {t.count(OLD)} — aborting")
if not re.search(r"^import os\b", t, re.M):
    sys.exit("cuda.py has no top-level 'import os' — aborting")
DST.write_text(t.replace(OLD, NEW))
print(f"OK: pin gate patched -> {DST}")
