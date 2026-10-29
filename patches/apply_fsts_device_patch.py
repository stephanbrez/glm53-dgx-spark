#!/usr/bin/env python3
"""Patch srt/model_loader/weight_utils.py: fastsafetensors_weights_iterator maps the
WORLD rank to a CUDA ordinal (device = cuda:{pg.rank()}). That is only valid for
single-node TP; on cross-node clusters with 1 GPU per node (2x DGX Spark GB10,
nnodes=2 tp=2), rank 1 lands on cuda:1 -> torch.AcceleratorError 'invalid device
ordinal'. Fix: reduce the world rank to a LOCAL ordinal (rank % device_count).

Self-verifying: aborts unless the target block is found exactly once.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
SRC = HERE / "orig" / "weight_utils.py"
DST = HERE / "weight_utils.sm121.py"

OLD = (
    "    try:\n"
    "        rank = pg.rank()\n"
    "    except Exception:\n"
    "        rank = 0\n"
    "\n"
    "    device = torch.device(f\"cuda:{rank}\")\n"
)
NEW = (
    "    try:\n"
    "        rank = pg.rank()\n"
    "    except Exception:\n"
    "        rank = 0\n"
    "\n"
    "    # GB10 cross-node (1 GPU per node): world rank != local cuda ordinal.\n"
    "    if torch.cuda.is_available() and torch.cuda.device_count() > 0:\n"
    "        rank = rank % torch.cuda.device_count()\n"
    "    device = torch.device(f\"cuda:{rank}\")\n"
)

t = SRC.read_text()
n = t.count(OLD)
if n != 1:
    sys.exit(f"expected exactly one target block, found {n} — aborting")
DST.write_text(t.replace(OLD, NEW))
print(f"OK: fsts device-ordinal patched -> {DST}")
