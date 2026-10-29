#!/usr/bin/env python3
"""Decode benchmark against any OpenAI-compatible chat endpoint. v2 metrics.

v2 (thinking-aware): this target's sglang chat template is thinking-ALWAYS-ON;
reasoning deltas are real decode work and requests can spend their whole budget
on reasoning. Therefore the PRIMARY metrics are engine-level:
  - ttft_any: request start -> first delta of ANY kind (content or reasoning)
  - decode_all: (usage.completion_tokens - 1) / (t_done - t_first_any)
  - failures are HTTP errors / U+FFFD only; reasoning-only responses are live
Secondary (informational, only when final content was produced):
  - ttft_answer, decode_answer over the content-only span
Arms measured with this harness (same prompts/protocol):
  arm 1 vLLM+B12X+FP4-Local+MTP :8000 | arm 2 SGLang+NVFP4 :30000 | arm 3 +EAGLE3
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("count_100", "Count from 1 to 100, one number per line.", 640),
    ("count_300", "Count from 1 to 300, one number per line.", 1152),
    ("math", "A train leaves at 3 pm travelling 80 km/h; another leaves the same station at 4 pm at 100 km/h. At what time does the second catch up? Show your steps.", 512),
    ("code", "Write a Python function to find the longest palindromic substring, plus two test calls.", 512),
    ("json", "Return only a JSON array of 5 world capitals, each an object with country and population.", 384),
    ("sql", "Write a SQL query using a window function to find the top 3 sales per region from a table sales(region, rep, amount).", 384),
    ("tool", "Describe the exact function calls (names and arguments) a travel-agent assistant should make to book a one-way flight SFO to JFK on Dec 12 for one passenger.", 384),
    ("summary", "Summarize the plot of The Old Man and the Sea in exactly 5 sentences.", 384),
    ("prose", "Write one solid paragraph explaining why the sea is salty.", 384),
    ("narrative", "Write a ~200-word story about a lighthouse keeper who hears a radio signal from the past.", 384),
]


def _open_stream(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def stream_chat(base, model, prompt, max_tokens, temperature):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    try:
        resp = _open_stream(base + "/v1/chat/completions", body)
    except urllib.error.HTTPError:
        body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = _open_stream(base + "/v1/chat/completions", body)
    t_send = time.time()
    t_first_any = None
    t_first_content = None
    t_done = None
    deltas = 0
    reason_deltas = 0
    usage_tok = None
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        now = time.time()
        u = chunk.get("usage")
        if u and u.get("completion_tokens") is not None:
            usage_tok = u["completion_tokens"]
        ch = (chunk.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        rc = delta.get("reasoning_content") or ""
        c = delta.get("content") or ""
        if rc:
            reason_deltas += 1
            if t_first_any is None:
                t_first_any = now
        if c:
            deltas += 1
            if t_first_any is None:
                t_first_any = now
            if t_first_content is None:
                t_first_content = now
        if ch.get("finish_reason"):
            t_done = now
    if t_done is None:
        t_done = time.time()
    toks = usage_tok if usage_tok is not None else (deltas + reason_deltas)
    span = (t_done - t_first_any) if t_first_any and t_done > t_first_any else None
    return {
        "ttft_any": (t_first_any - t_send) if t_first_any else None,
        "ttft_answer": (t_first_content - t_send) if t_first_content else None,
        "wall": t_done - t_send,
        "tokens": toks,
        "content_deltas": deltas,
        "reason_deltas": reason_deltas,
        "decode_all": ((toks - 1) / span) if span and toks > 1 else None,
        "decode_answer": ((deltas - 1) / (t_done - t_first_content))
        if t_first_content and deltas > 1 and t_done > t_first_content else None,
    }


def single_stream(args, base, model):
    table = {}
    for name, prompt, mt in PROMPTS:
        runs = []
        for _ in range(args.rounds):
            try:
                r = stream_chat(base, model, prompt, mt, args.temperature)
                if r["tokens"] and (r["content_deltas"] + r["reason_deltas"]) > 0:
                    runs.append(r)
                else:
                    print(f"  dead-response {name}", flush=True)
            except Exception as e:
                print(f"  error {name}: {str(e)[:150]}", flush=True)
        good = [r for r in runs if r["decode_all"]]
        if good:
            table[name] = {
                "decode_all_med": round(statistics.median(r["decode_all"] for r in good), 1),
                "ttft_any_med": round(statistics.median(r["ttft_any"] for r in good), 3),
                "tokens_med": statistics.median(r["tokens"] for r in good),
                "decode_answer_med": round(statistics.median(
                    [r["decode_answer"] for r in good if r["decode_answer"]] or [0]), 1),
                "ttft_answer_med": round(statistics.median(
                    [r["ttft_answer"] for r in good if r["ttft_answer"]] or [0]), 3),
                "thinking_runs": sum(1 for r in runs if r["reason_deltas"] > 0),
                "raw": runs,
            }
        else:
            table[name] = {"error": True, "raw": runs}
    return table


def conc_worker(base, model, idx, deadline, temperature):
    reqs = []
    i = 0
    while time.time() < deadline:
        name, prompt, mt = PROMPTS[(idx + i) % len(PROMPTS)]
        try:
            r = stream_chat(base, model, prompt, mt, temperature)
            comb = r["content_deltas"] + r["reason_deltas"]
            r["dead"] = (comb == 0)
            reqs.append(r)
        except Exception as e:
            print(f"  conc stream {idx} error: {str(e)[:150]}", flush=True)
        i += 1
    return reqs


def concurrent_pass(args, base, model):
    deadline = time.time() + args.secs
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.streams) as ex:
        futs = [ex.submit(conc_worker, base, model, i, deadline, args.temperature)
                for i in range(args.streams)]
        per_stream = [f.result() for f in futs]
    wall = time.time() - t0
    allreq = [r for s in per_stream for r in s]
    toks = sum(r["tokens"] or 0 for r in allreq)
    dead = len([r for r in allreq if r.get("dead")])
    http_fails = len(allreq) - len([r for r in allreq if not r.get("dead")]) if False else \
        len([r for r in allreq if r.get("dead")])
    return {
        "streams": args.streams, "secs": args.secs, "requests": len(allreq),
        "dead_responses": dead, "tokens": toks, "wall": round(wall, 1),
        "aggregate_tps": round(toks / wall, 1) if wall else 0,
        "per_stream_tps": [round(sum(r["tokens"] or 0 for r in s) / wall, 1) for s in per_stream],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--secs", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--skip-concurrent", action="store_true")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, model = args.url.rstrip("/"), args.model

    print(f"== bench_quick v2: {model} @ {base} (rounds={args.rounds}, streams={args.streams}x{args.secs}s, temp={args.temperature}) ==", flush=True)
    out = {"url": base, "model": model, "temperature": args.temperature,
           "harness": "v2-all-token-metrics", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.skip_single:
        print("-- single-stream (engine decode, all tokens incl. thinking) --", flush=True)
        t0 = time.time()
        out["single"] = single_stream(args, base, model)
        out["single_wall_s"] = round(time.time() - t0, 1)
        for name, d in out["single"].items():
            if d.get("decode_all_med") is not None:
                th = f" think={d['thinking_runs']}" if d.get("thinking_runs") else ""
                print(f"  {name:11s} decode_all={d['decode_all_med']:7.1f} tok/s  "
                      f"ttft_any={d['ttft_any_med']:6.2f}s  toks={d['tokens_med']:4.0f}  "
                      f"answer_decode={d['decode_answer_med']:6.1f} answer_ttft={d['ttft_answer_med']:6.2f}s{th}", flush=True)
            else:
                print(f"  {name:11s} ERROR (no usable response)", flush=True)
    if not args.skip_concurrent:
        print(f"-- concurrent ({args.streams} streams x {args.secs}s, all tokens) --", flush=True)
        out["concurrent"] = concurrent_pass(args, base, model)
        c = out["concurrent"]
        print(f"  requests={c['requests']} dead={c['dead_responses']} tokens={c['tokens']} "
              f"aggregate={c['aggregate_tps']} tok/s per_stream={c['per_stream_tps']}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written {args.out}", flush=True)


if __name__ == "__main__":
    main()
