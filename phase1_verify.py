#!/usr/bin/env python3
"""Phase-1 verification gates for GLM-5.3-Flash-NVFP4 on sglang (TP2 pinus+cedrus).

v3 rules, informed by the live boot:
- This checkpoint's chat template is thinking-ALWAYS-ON (sglang ReasoningToggleConfig
  special_case='always', no toggle kwarg). Long prompts spend their token budget on
  reasoning_content and can return empty content. Therefore: answer-presence and
  want-match are evaluated on content+reasoning COMBINED; corruption (U+FFFD) is
  counted across both; known-answer prompts get generous budgets.
- logprobs come from /generate meta_info.output_token_logprobs as
  [logprob, token_id, null] tuples.
- soak failure = HTTP error / U+FFFD / totally-empty (both fields). Thinking-budget
  truncations (reasoning present, content empty) are counted separately, not failures.

Hard gates: health, tp_size==2, sanity (known answers), corruption, determinism,
logprobs finite/sane, JSON-schema output, soak zero-failures.
Info gates (WARN): reasoning parser, tool-call parser.
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request

results = []
lock = threading.Lock()


def record(name, status, detail=""):
    with lock:
        results.append({"gate": name, "status": status, "detail": detail})
    print(f"[{status}] {name}: {detail}", flush=True)


def http_json(method, url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:
        return -1, str(e)


def chat(base, model, messages, max_tokens=256, temperature=0.0, extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if extra:
        body.update(extra)
    code, resp = http_json("POST", base + "/v1/chat/completions", body)
    if code != 200:
        raise RuntimeError(f"chat HTTP {code}: {str(resp)[:300]}")
    if isinstance(resp, str):
        raise RuntimeError(f"chat non-JSON {code}: {resp[:300]}")
    return resp


def fields(r):
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or "")


def combined(r):
    c, rc = fields(r)
    return c + "\n" + rc


SANITY = [
    # (name, messages, required substring or None, max_tokens)
    ("math_2plus2", [{"role": "user", "content": "What is 2+2? Answer with just the number."}], "4", 1024),
    ("math_17x23", [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}], "391", 1024),
    ("fact_capital", [{"role": "user", "content": "What is the capital of France? Answer with one word."}], "paris", 1024),
    ("code", [{"role": "user", "content": "Write a Python function is_palindrome(s) that ignores case and spaces."}], None, 2048),
    ("json", [{"role": "user", "content": "Give me a JSON object with keys name and age for Alice, age 30."}], None, 1536),
    ("sql", [{"role": "user", "content": "Write SQL to select the top 5 customers by total spend from table orders(customer_id, amount)."}], None, 2048),
    ("prose", [{"role": "user", "content": "In three sentences, explain why speculative decoding speeds up LLM inference."}], None, 2048),
    ("korean_capital", [{"role": "user", "content": "한국의 수도는 어디인가요? 한 문장으로 답하세요."}], "seoul", 1536),
    ("korean_translate", [{"role": "user", "content": "봄이 오면 산과 들에 꽃이 핍니다. 이 문장을 영어로 번역하세요."}], None, 2048),
    ("korean_explain", [{"role": "user", "content": "파이썬과 자바의 차이를 두 문장으로 설명하세요."}], None, 2048),
]

SOAK_PROMPTS = [
    "Write a Python one-liner that reverses a string.",
    "Name three primary colors.",
    "What is 12 * 12?",
    "Write a haiku about rain.",
    "List two benefits of containerization.",
    "Translate 'good morning' to French.",
    "What planet is closest to the sun?",
    "Write a regex that matches an IPv4 address.",
]


def gate_health(args):
    deadline = time.time() + args.health_timeout
    ok = False
    while time.time() < deadline:
        code, _ = http_json("GET", args.url.rstrip("/") + "/health", timeout=30)
        if code == 200:
            ok = True
            break
        time.sleep(10)
    record("health", "PASS" if ok else "FAIL",
           f"{'server ready' if ok else f'no healthy response in {args.health_timeout}s'}")
    return ok


def gate_server_info(args):
    code, info = http_json("GET", args.url.rstrip("/") + "/get_server_info", timeout=120)
    tp = None
    ver = "?"
    if code == 200 and isinstance(info, dict):
        ver = info.get("version", "?")
        m = re.search(r'"tp_size":\s*(\d+)', json.dumps(info))
        if m:
            tp = int(m.group(1))
    record("tp_size", "PASS" if tp == 2 else "FAIL", f"tp_size={tp} sglang={ver}")
    return tp == 2


def gate_sanity(args, base, model):
    ok = True
    fffd_total = 0
    for name, messages, want, mt in SANITY:
        try:
            r = chat(base, model, messages, max_tokens=mt)
            c, rc = fields(r)
            comb = combined(r)
            fffd = comb.count("\ufffd")
            fffd_total += fffd
            good = (len(c.strip()) + len(rc.strip())) >= 10 and fffd == 0
            if want:
                good = good and want.lower() in comb.lower()
            status = "PASS" if good else "FAIL"
            ok = ok and good
            record(f"sanity/{name}", status,
                   f"content_len={len(c)} reasoning_len={len(rc)} fffd={fffd} "
                   f"want={want} content_head={c[:60]!r}")
        except Exception as e:
            ok = False
            record(f"sanity/{name}", "FAIL", str(e)[:200])
    record("corruption_ufffd", "PASS" if fffd_total == 0 else "FAIL",
           f"U+FFFD count across all sanity outputs (content+reasoning): {fffd_total}")
    return ok and fffd_total == 0


def gate_korean_probe(args, base, model):
    total = 0
    try:
        for _ in range(2):
            for name, messages, _w, mt in [s for s in SANITY if s[0].startswith("korean")]:
                r = chat(base, model, messages, max_tokens=mt)
                total += combined(r).count("\ufffd")
        record("hangul_probe", "PASS" if total == 0 else "FAIL",
               f"U+FFFD over 2 passes x 3 Korean prompts: {total}")
        return total == 0
    except Exception as e:
        record("hangul_probe", "FAIL", str(e)[:200])
        return False


def gate_determinism(args, base, model):
    ok = True
    try:
        for name in ("math_17x23", "prose"):
            entry = [s for s in SANITY if s[0] == name][0]
            a = chat(base, model, entry[1], max_tokens=entry[3])
            b = chat(base, model, entry[1], max_tokens=entry[3])
            ca, ra = fields(a)
            cb, rb = fields(b)
            same = (ca == cb) and (ra == rb)
            ok = ok and same
            record(f"determinism/{name}", "PASS" if same else "FAIL",
                   f"identical={same} content_head={ca[:40]!r}")
        return ok
    except Exception as e:
        record("determinism", "FAIL", str(e)[:200])
        return False


def gate_logprobs(args, base):
    try:
        body = {"text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 24},
                "return_logprob": True}
        code, resp = http_json("POST", base + "/generate", body)
        if code != 200 or isinstance(resp, str):
            record("logprobs", "FAIL", f"HTTP {code}: {str(resp)[:200]}")
            return False
        meta = resp.get("meta_info") or {}
        vals = []
        for row in meta.get("output_token_logprobs") or []:
            if isinstance(row, list) and isinstance(row[0], (int, float)):
                vals.append(row[0])
        finite = bool(vals) and all(v == v and abs(v) != float("inf") for v in vals)
        floor = min(vals) if vals else None
        good = finite and floor is not None and floor > -50
        record("logprobs", "PASS" if good else "FAIL",
               f"n={len(vals)} min={floor} max={max(vals) if vals else None}")
        return good
    except Exception as e:
        record("logprobs", "FAIL", str(e)[:200])
        return False


def gate_json_schema(args, base, model):
    try:
        extra = {"response_format": {"type": "json_schema", "json_schema": {
            "name": "languages",
            "schema": {"type": "object", "properties": {"languages": {"type": "array", "items": {
                "type": "object", "properties": {
                    "name": {"type": "string"},
                    "primary_use": {"type": "string"},
                    "year_created": {"type": "integer"}},
                    "required": ["name", "primary_use", "year_created"]}}},
                "required": ["languages"]}}}}
        r = chat(base, model,
                 [{"role": "user", "content": "List three programming languages with their primary use case and year created."}],
                 max_tokens=2048, extra=extra)
        c, rc = fields(r)
        text = c if c.strip() else rc
        # extract first JSON object from fences/noise
        m = re.search(r"\{.*\}", text, re.S)
        assert m, f"no JSON object found; content_len={len(c)} reasoning_len={len(rc)}"
        data = json.loads(m.group(0))
        langs = data["languages"]
        assert len(langs) == 3, f"expected 3 items, got {len(langs)}"
        for it in langs:
            assert isinstance(it["name"], str) and isinstance(it["primary_use"], str)
            assert isinstance(it["year_created"], int)
        record("json_schema", "PASS",
               f"valid schema-constrained output, {len(langs)} items (from content={bool(c.strip())})")
        return True
    except Exception as e:
        record("json_schema", "FAIL", str(e)[:250])
        return False


def gate_reasoning(args, base, model):
    try:
        r = chat(base, model,
                 [{"role": "user", "content": "Explain briefly why the sky is blue."}],
                 max_tokens=2048)
        c, rc = fields(r)
        status = "PASS" if rc.strip() else "WARN"
        record("reasoning_parser", status,
               f"reasoning_len={len(rc)} content_len={len(c)} (thinking-always template)")
        return True
    except Exception as e:
        record("reasoning_parser", "FAIL", str(e)[:200])
        return False


def gate_toolcall(args, base, model):
    try:
        extra = {"tools": [{"type": "function", "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}}, "required": ["city"]}}}],
            "tool_choice": "auto"}
        r = chat(base, model,
                 [{"role": "user", "content": "What is the weather in Tokyo right now? Use the tool."}],
                 max_tokens=2048, extra=extra)
        m = r["choices"][0]["message"]
        tc = m.get("tool_calls") or []
        blob = json.dumps(m)
        status = "PASS" if tc else "WARN"
        record("tool_call_parser", status if "\ufffd" not in blob else "FAIL",
               f"parsed_tool_calls={len(tc)} (WARN = answered without calling)")
        return "\ufffd" not in blob
    except Exception as e:
        record("tool_call_parser", "FAIL", str(e)[:200])
        return False


def soak_worker(base, model, idx, seconds):
    done, toks, fails, truncs = 0, 0, 0, 0
    end = time.time() + seconds
    i = 0
    while time.time() < end:
        try:
            r = chat(base, model,
                     [{"role": "user", "content": SOAK_PROMPTS[(idx + i) % len(SOAK_PROMPTS)]}],
                     max_tokens=384, temperature=0.7)
            c, rc = fields(r)
            comb = c + rc
            if "\ufffd" in comb:
                fails += 1
            elif not comb.strip():
                fails += 1
            else:
                done += 1
                if not c.strip() and rc.strip():
                    truncs += 1  # thinking ate the budget: live response, no final answer
            toks += (r.get("usage") or {}).get("completion_tokens", 0)
        except Exception as e:
            fails += 1
            print(f"soak stream {idx}: {str(e)[:150]}", flush=True)
        i += 1
    return done, toks, fails, truncs


def gate_soak(args, base, model):
    if args.skip_soak:
        record("soak", "WARN", "skipped (--skip-soak)")
        return True
    secs = args.soak_mins * 60
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.soak_streams) as ex:
        futs = [ex.submit(soak_worker, base, model, i, secs) for i in range(args.soak_streams)]
        outs = [f.result() for f in futs]
    wall = time.time() - t0
    done = sum(o[0] for o in outs)
    toks = sum(o[1] for o in outs)
    fails = sum(o[2] for o in outs)
    truncs = sum(o[3] for o in outs)
    tps = toks / wall if wall else 0
    good = fails == 0 and tps > 1
    record("soak", "PASS" if good else "FAIL",
           f"{args.soak_streams} streams x {args.soak_mins}min: completions={done} "
           f"failures={fails} thinking_budget_truncations={truncs} tokens={toks} "
           f"wall={wall:.0f}s aggregate={tps:.1f} tok/s "
           f"per_stream={[round(o[1] / secs, 1) for o in outs]}")
    return good


def finish(args, hard_ok):
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    fails = [r["gate"] for r in results if r["status"] == "FAIL"]
    warns = [r["gate"] for r in results if r["status"] == "WARN"]
    print(f"\n=== phase1 summary: {'PASS' if hard_ok and not fails else 'FAIL'} "
          f"({len(results) - len(fails) - len(warns)} pass, {len(warns)} warn, {len(fails)} fail) ===")
    if fails:
        print("failed gates:", ", ".join(fails))
    print(f"results written to {args.out}")
    sys.exit(0 if (hard_ok and not fails) else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--health-timeout", type=int, default=1800)
    ap.add_argument("--soak-streams", type=int, default=4)
    ap.add_argument("--soak-mins", type=float, default=10.0)
    ap.add_argument("--skip-soak", action="store_true")
    ap.add_argument("--out", default="phase1_results.json")
    args = ap.parse_args()
    base, model = args.url.rstrip("/"), args.model

    hard = gate_health(args)
    if not hard:
        finish(args, False)
    hard = gate_server_info(args) and hard
    hard = gate_sanity(args, base, model) and hard
    hard = gate_korean_probe(args, base, model) and hard
    hard = gate_determinism(args, base, model) and hard
    hard = gate_logprobs(args, base) and hard
    hard = gate_json_schema(args, base, model) and hard
    gate_reasoning(args, base, model)          # info
    gate_toolcall(args, base, model)           # info
    hard = gate_soak(args, base, model) and hard
    finish(args, hard)


if __name__ == "__main__":
    main()
