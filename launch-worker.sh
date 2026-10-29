#!/bin/bash
# Phase-1 sglang bring-up: GLM-5.3-Flash-NVFP4, TP=2 across head node(head)+worker node(worker).
# ORDER: run this script on worker node FIRST, then launch-head.sh on head node.
# Prereqs: vLLM stack STOPPED; desktop sessions stopped; checkpoint downloaded;
#          overlay image glm53-sglang:sm121 (v4: retune + pin gate + FSTS + device fix).
# v6: --mamba-ssm-dtype bfloat16 removed (KDA triton kernel compile failure) — see
#     launch-head.sh v6 note.
set -euo pipefail
cd "$(dirname "$0")"
source ./ENV

NAME=sglang_p1w
RANK=1

# GB10 unified-memory ritual (best effort)
sync
if ! echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
  echo "note: drop_caches skipped (sudo not passwordless); watch free mem on first boot"
fi

mkdir -p "$HOME/glm53-dgx-spark/cache/triton" "$HOME/glm53-dgx-spark/cache/flashinfer"
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --network host \
  --ipc host \
  --cap-add IPC_LOCK \
  --device /dev/infiniband \
  --memory 120g --memory-swap 120g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=1048576:1048576 \
  -e SGLANG_HOST_IP="$WORKER_IP" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$IB_IF" \
  -e NCCL_IB_MERGE_NICS=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME="$ETH_IF" \
  -e NCCL_BUFFSIZE=1048576 \
  -e NCCL_MAX_CTAS=8 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e GLOO_SOCKET_IFNAME="$ETH_IF" \
  -e TP_SOCKET_IFNAME="$ETH_IF" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_DISABLE_PIN_MEMORY=1 \
  -e MALLOC_ARENA_MAX=2 \
  -e HF_HUB_OFFLINE=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/glm53-dgx-spark/cache/triton:/root/.triton" \
  -v "$HOME/glm53-dgx-spark/cache/flashinfer:/root/.cache/flashinfer" \
  "$IMAGE" \
  sglang serve \
    --model-path "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 --port "$PORT" \
    --tp-size 2 --nnodes 2 --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IP:$DIST_PORT" \
    --quantization modelopt_fp4 \
    --attention-backend dsa \
    --dsa-prefill-backend "$DSA_PREFILL" \
    --dsa-decode-backend "$DSA_DECODE" \
    --kv-cache-dtype "$KV_DTYPE" \
    --moe-runner-backend "$MOE_BACKEND" \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --context-length "$CTX" \
    --max-running-requests "$MAX_REQ" \
    --chunked-prefill-size "$CHUNKED_PREFILL" \
    --mem-fraction-static "$MEMFRAC" \
    --max-mamba-cache-size 40 \
    --disable-flashinfer-autotune \
    --stream-interval 1 \
    --sleep-on-idle \
    --enable-cache-report \
    ${EXTRA_ARGS:-}

echo "launched $NAME (node-rank $RANK); now run launch-head.sh on head node"
echo "follow: docker logs -f $NAME"
