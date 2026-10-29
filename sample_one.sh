#!/bin/bash
# Per-node memory sampler for one sglang container.
# Usage: sample_one.sh <container> [seconds] [healthcheck 0|1]
# Prints cgroup memory.current / anon / file / shmem / kernel + GPU(UMA) used every 10s.
# Exits on: /health OK (if healthcheck=1), container death, or timeout.
# Cgroup path is derived from the container's main PID (/proc/<pid>/cgroup) — no guessing.
C=${1:-sglang_p1}
T=${2:-300}
HC=${3:-1}
mb() { if [ -n "$1" ] && [ "$1" -gt 0 ] 2>/dev/null; then echo $(( $1 / 1024 / 1024 )); else echo "?"; fi; }
while true; do
  if [ "$HC" = "1" ] && curl -sf http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "HEALTH-OK t=${SECONDS}s"
    break
  fi
  st=$(docker inspect -f '{{.State.Status}}' "$C" 2>/dev/null)
  if [ "$st" != "running" ]; then
    ok=$(docker inspect -f '{{.State.OOMKilled}} {{.State.ExitCode}}' "$C" 2>/dev/null)
    echo "$C DIED state=$st oomkilled/exit=$ok t=${SECONDS}s"
    break
  fi
  pid=$(docker inspect -f '{{.State.Pid}}' "$C" 2>/dev/null)
  rel=$(cut -d: -f3 /proc/$pid/cgroup 2>/dev/null | sed -n 1p)
  cg="/sys/fs/cgroup$rel"
  if [ -n "$rel" ] && [ -f "$cg/memory.stat" ]; then
    cur=$(cat "$cg/memory.current" 2>/dev/null)
    g() { grep -E "^$1 " "$cg/memory.stat" 2>/dev/null | cut -d" " -f2; }
    anon=$(g anon); file=$(g file); shmem=$(g shmem); kern=$(g kernel)
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "t=${SECONDS}s cur=$(mb $cur)M anon=$(mb $anon)M file=$(mb $file)M shmem=$(mb $shmem)M kernel=$(mb $kern)M gpu=${gpu}M"
  else
    echo "t=${SECONDS}s cgroup-miss pid=$pid"
  fi
  if [ "$SECONDS" -ge "$T" ]; then
    echo "TIMEOUT t=${SECONDS}s (still running)"
    break
  fi
  sleep 10
done
echo "=== $C last 12 log lines ==="
docker logs --tail 12 "$C" 2>&1 | grep -v "^$"
