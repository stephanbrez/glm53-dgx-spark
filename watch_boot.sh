#!/bin/bash
# Watch first boot of sglang TP2: head sglang_p1 (pinus, local) + worker sglang_p1w (cedrus).
# Exits on: /health OK, a container dying, or timeout (arg1 seconds, default 1800).
# Then prints filtered head log + raw tails of both.
T=${1:-1800}
end=$((SECONDS + T))
while [ "$SECONDS" -lt "$end" ]; do
  if curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "HEALTH-OK after ${SECONDS}s"
    break
  fi
  hstate=$(docker inspect -f '{{.State.Status}}' sglang_p1 2>/dev/null)
  wstate=$(ssh -o BatchMode=yes -o ConnectTimeout=5 10.20.0.12 \
    "docker inspect -f '{{.State.Status}}' sglang_p1w" 2>/dev/null)
  if [ "$hstate" != "running" ]; then
    echo "HEAD-NOT-RUNNING state=$hstate after ${SECONDS}s"
    break
  fi
  if [ "$wstate" != "running" ]; then
    echo "WORKER-NOT-RUNNING state=$wstate after ${SECONDS}s"
    break
  fi
  sleep 15
done
if [ "$SECONDS" -ge "$end" ]; then
  echo "WATCH-TIMEOUT after ${T}s head=$(docker inspect -f '{{.State.Status}}' sglang_p1 2>/dev/null)"
fi
echo "=== HEAD LOG (filtered) ==="
docker logs sglang_p1 2>&1 | grep -E -i "error|fail|exception|traceback|assert|nan|kv cache|token pool|mamba|dsa|tilelang|attention|quant|cuda graph|captur|ready|listen|mem|weight|load|NCCL" | sed -n "1,150p"
echo "=== HEAD LOG (last 25 raw) ==="
docker logs --tail 25 sglang_p1 2>&1
echo "=== WORKER LOG (last 25 raw via cedrus) ==="
ssh -o BatchMode=yes 10.20.0.12 "docker logs --tail 25 sglang_p1w 2>&1"
