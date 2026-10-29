#!/usr/bin/env python3
"""Port the beastllama GB10 (sm_121) tilelang DSA smem retune onto sglang 0.5.22's
tilelang_kernel.py.

Only kernel-factory DEFAULT params change (block_I 64->32, num_stages 2->1,
threads 256/384->128). Fork-specific changes from the source recipe (return_lse
calling convention, formatting) are deliberately NOT ported: 0.5.22's
dsa_backend.py calls these kernels with the original signatures.

Why: GB10 opt-in dynamic smem is 101,376 B; the stock 64-wide tiles need
~170-207 KB and fail with tvm InternalError. Source: beastllama's
GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark (their fix #3, tile-space mapped).

Self-verifying: every edit must match the expected old line, else exits
without writing anything.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / "orig" / "tilelang_kernel.py"
DST = HERE / "tilelang_kernel.sm121.py"

EDITS = [
    (271, "    block_I=64,", "    block_I=32,"),                # v1 sparse fwd (prefill)
    (272, "    num_stages=2,", "    num_stages=1,"),
    (273, "    threads=256,", "    threads=128,"),
    (455, "    block_I: int = 64,", "    block_I: int = 32,"),  # v2 sparse fwd
    (470, "    threads = 384", "    threads = 128"),
    (828, "    block_I=64,", "    block_I=32,"),                # decode partial
    (831, "    threads=256,", "    threads=128,"),
    (1010, "    block_I=64,", "    block_I=32,"),               # decode combine
    (1011, "    threads=256,", "    threads=128,"),
    (1091, "    block_I=64,", "    block_I=32,"),               # decode partial fp8
    (1093, "    threads=256,", "    threads=128,"),
    (1628, "    block_I: int = 64,", "    block_I: int = 32,"),  # dual-cache
    (2260, "    block_I: int = 64,", "    block_I: int = 32,"),  # dpsk_v4 combine
]

lines = SRC.read_text().splitlines(keepends=True)
for ln, old, new in EDITS:
    got = lines[ln - 1].rstrip("\n")
    if got != old:
        sys.exit(f"MISMATCH at line {ln}: expected {old!r}, found {got!r} — aborting, nothing written")
    lines[ln - 1] = new + "\n"
DST.write_text("".join(lines))
print(f"OK: applied {len(EDITS)} edits -> {DST}")
