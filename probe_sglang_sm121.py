#!/usr/bin/env python3
"""Probe the sglang image (run INSIDE the container) for sm121/GB10 gating that
determines the GLM-5.3-Flash launch flags. Prints findings; exit 0 always."""
import re
import pathlib
import sys

try:
    import sglang
except ImportError:
    print("FATAL: no sglang in this env")
    sys.exit(1)

P = pathlib.Path(sglang.__file__).parent
print("PKG", P)
print("VERSION", getattr(sglang, "__version__", "?"))

PATTERNS = {
    "kernels/ops/attention/flash_mla_sm120.py":
        ["_GLM_DSA_MODEL_ARCHS", "Glm5Next", "sm120", "sm121", "fp8"],
    "kernels/ops/attention/dsa_backend.py":
        ["tilelang", "block_I", "num_stages", "threads", "ROCm/HIP",
         "fp8_e4m3 KV cache", "trtllm", "Unsupported architecture"],
    "serving/model_loader.py": ["modelopt"],
}


def show(rel, pats):
    f = P / rel
    print(f"\n=== {rel} exists={f.exists()}")
    if not f.exists():
        return
    lines = f.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        for pat in pats:
            if re.search(pat, line):
                print(f"{i}: {line.strip()[:150]}")
                break


for rel, pats in PATTERNS.items():
    show(rel, pats)

# Whole-package sweep for the decisive strings
print("\n=== package-wide sweep")
HITS = ["GlmMoeDsaForCausalLM", "Glm5NextForConditionalGeneration",
        "Glm5Next", "max_mamba_cache_size", "mamba_full_memory_ratio",
        "SGLANG_HOST_IP", "101376", "169984", "spec_accept"]
seen = set()
for f in P.rglob("*.py"):
    rel = str(f.relative_to(P))
    if rel in seen:
        continue
    try:
        lines = f.read_text(errors="replace").splitlines()
    except Exception:
        continue
    for i, line in enumerate(lines, 1):
        for h in HITS:
            if h in line and (h, rel) not in seen:
                seen.add((h, rel))
                print(f"{h} -> {rel}:{i}: {line.strip()[:110]}")
                break

# glm5_next model file: spec-decoding support surface
g = P / "serving" / "models" / "glm5_next.py"
print(f"\n=== glm5_next.py exists={g.exists()}")
if g.exists():
    lines = g.read_text(errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        if re.search(r"class |Eagle|NextN|nextn", line):
            print(f"{i}: {line.strip()[:130]}")

print("\nDONE")
