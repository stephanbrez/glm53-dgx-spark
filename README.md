# GLM-5.3-Flash NVFP4 on 2x DGX Spark (GB10) — stock SGLang serving recipe

> Serve **`nvidia/GLM-5.3-Flash-NVFP4`** with **stock SGLang 0.5.22** (pinned
> digest + 4-patch overlay) across **2x NVIDIA DGX Spark (GB10, sm_121a,
> aarch64)** at TP=2 over dual-rail RoCE. Launch scripts, memory rules,
> verification gates, and benchmarks included.

## What you need

- **Hardware:** two DGX Spark nodes, each 1x GB10 (121.8 GB unified memory,
  aarch64, driver 580.178.04). Dual-rail RoCE (`rocep1s0f0` +
  `roceP2p1s0f0`) + control NIC (`enp1s0f0np0`), MTU 9000. Fabric rsync
  ~600 MB/s.
- **Model:** `nvidia/GLM-5.3-Flash-NVFP4` (320B MoE / 18B active; MLA + DSA
  sparse attention + KDA linear attention, multimodal). NVIDIA ModelOpt PTQ
  **NVFP4 W4A16** — attention and shared experts in high precision, the
  corruption-free build on sm_121a. ~190 GiB total, ~95 GB/rank at TP2.
  HF cache must live on BOTH nodes (Xet layout: chunks under
  `hub/blobs/xx/<hash>`, per-model symlinks).
- **Engine:** SGLang 0.5.22 official arm64/cu130 image
  (`lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae`)
  — no sglang fork needed.

## Build the overlay (both nodes, identical)

```bash
cd ~/glm53-sglang/patches
docker build -f Dockerfile.sm121 -t glm53-sglang:sm121 .
```

Four patches on top of the pinned base (all in `patches/`):

1. `pip install fastsafetensors` — GPU-direct weight loader.
2. **tilelang DSA shared-memory retune** (`kernels/ops/attention/dsa/
   tilelang_kernel.py`): `block_I 64→32`, `num_stages 2→1`,
   `threads 256/384→128`. GB10 opt-in dynamic smem is 101,376 B; stock
   64-wide tiles need ~202 KB and die with `tvm.error.InternalError`
   (sglang #40286 / #39302). Param-only port onto 0.5.22.
3. **Env-gated pin-memory disable** (`SGLANG_DISABLE_PIN_MEMORY=1` in
   `srt/platforms/cuda.py`, default unchanged) — pinned staging is waste
   on unified memory.
4. **fastsafetensors world-rank→local-device fix**
   (`srt/model_loader/weight_utils.py`): `cuda:{pg.rank()}` is only valid
   single-node; 1-GPU-per-node must use `rank % device_count`, else
   `torch.AcceleratorError: invalid device ordinal` on the worker.

## Lessons learned

1. **Device weights are charged as host RAM outside container accounting.**
   ~95 GB/rank shows in host memory but in no container memcg, so
   `--memory` caps are nearly meaningless here. Keep ~10 GB of the
   121.8 GB free.
2. **The default weight loader OOMs the host at end-of-load.** It stages
   ~19% of the checkpoint (the bf16 attention / shared-expert part) on
   top of the 95 GB of weights. `--load-format fastsafetensors` loads
   GPU-direct: 95 GB in ~85 s, flat host footprint.
3. **Desktop sessions are OOM bait.** GDM costs 4–8 GB and is the
   preferred victim. `sudo systemctl stop gdm` on both nodes before
   serving.
4. **mem-fraction math:** weights+baseline = **0.824** of 124,544 MiB;
   `--mem-fraction-static 0.87` leaves ~5.7 GB for KV + mamba pools
   (fits 131K context). Steady host state ≈ 112 GB of 121.8.
5. **Keep the default `--mamba-ssm-dtype`.** `bfloat16` breaks the KDA
   prefill kernel on 0.5.22 (`kda_prefill_graph_track.py` cannot unify
   bf16/fp32 in the `val` branch).
6. **The chat template thinks always.** `ReasoningToggle
   special_case='always'` — `chat_template_kwargs {"thinking": false}`
   is ignored on this route. Budget 2048+ reasoning tokens for code
   prompts, or parse `reasoning_content` and `content` separately.

## Launch (worker first)

Both nodes source the shared `ENV`; scripts are worker-first:

```bash
bash ~/glm53-sglang/launch-worker.sh   # worker, ~25 s
bash ~/glm53-sglang/launch-head.sh     # head; serving starts here
bash ~/glm53-sglang/watch_boot.sh      # optional: watch the boot
```

`ENV` pins: image + digest, model id, head/worker IPs (10.20.0.11/12),
port 30000 / dist port 29500, and the serving flags:

- `--quantization modelopt_fp4 --attention-backend dsa --dsa-prefill-backend
  tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16
  --moe-runner-backend flashinfer_cutlass --mem-fraction-static 0.87
  --max-mamba-cache-size 40 --context-length 131072 --max-running-requests 8
  --chunked-prefill-size 4096 --disable-flashinfer-autotune
  --load-format fastsafetensors --model-loader-extra-config
  '{"enable_gds":false}'`
- NCCL for dual-rail RoCE: `NCCL_IB_HCA`, `NCCL_IB_MERGE_NICS=1`,
  `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`, `SGLANG_HOST_IP` per rank.

First boot ≈ **5.6 min** (85 s GPU-direct load + JIT + CUDA graphs).
Caches persist under `~/glm53-sglang/cache/` for faster restarts.

## Verify the deployment

`python phase1_verify.py` — **21/21 PASS**:

- Health: server up, `tp_size=2`, SGLang 0.5.22 detected.
- Answer sanity (incl. Hangul): all known answers correct, **zero U+FFFD
  anywhere** — the sm121 NVFP4 token-corruption mode is absent on this
  build.
- Determinism: byte-identical greedy repeats; finite logprobs in healthy
  range.
- JSON-schema constrained output valid; reasoning and tool-call parsers
  behave.
- 10-min soak, 4 streams: **0 failures**, 90 completions, 44.8 tok/s
  aggregate.

## Expected performance (this hardware class, no speculation)

| workload | result |
|---|---|
| single-stream decode | 15.3–15.4 tok/s flat (bandwidth plateau) |
| sharegpt 32 prompts @ r=1 | 48.2 tok/s output (peak 72.0), TPOT 135 ms, 32/32 |
| 10-min 4-stream soak | 44.8 tok/s aggregate, 0 failures |

Single-stream matches independent 2x-Spark measurements (14.5–14.7 tok/s)
on this engine/nvfp4 route. Known headroom in this stack: fp8-KV tilelang
path + mamba-concurrency tuning (~77–83 tok/s @ c8 reported on this
hardware class).

## Repo layout

- `ENV`, `launch-head.sh`, `launch-worker.sh` — frozen production launch.
- `patches/` — `Dockerfile.sm121` + patch sources/appliers + stock
  reference files.
- `phase1_verify.py` — verification gates (thinking-aware rules).
- `bench_quick.py` — benchmark harness (all-token metrics).
- `watch_boot.sh`, `sample_one.sh` — boot watcher + cgroup/host memory
  sampler.
- `phase1_results.json`, `bench_sglang_nvfp4.json`, `bench_vllm_fp4local*.json`,
  `bench_serving_nvfp4.json` — raw results.

## Credits / prior art

- beastllama's `GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark` (tilelang
  retune, memory law), tonyd2wild's GB10 forensics + 2x-Spark vLLM
  recipes, eugr's spark-vllm-docker (fabric/NCCL, fastsafetensors), sglang
  issues #11658, #39302, #40286, SGLang cookbook, NVIDIA DGX Spark
  playbook.
- Checkpoint: `nvidia/GLM-5.3-Flash-NVFP4` (MIT). This recipe: MIT.
