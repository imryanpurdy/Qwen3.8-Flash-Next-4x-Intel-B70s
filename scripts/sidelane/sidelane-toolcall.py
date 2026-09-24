#!/usr/bin/env python3
"""Promotion gate: 20 structural tool calls + multi-tool pick + nested-arguments case."""
import argparse
import json
import time
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}
TOOL_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time for a city.",
        "parameters": {
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
            "required": [],
        },
    },
}
TOOL_NESTED = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "attendees": {"type": "array", "items": {"type": "string"}},
                        "when": {
                            "type": "object",
                            "properties": {"date": {"type": "string"}, "time": {"type": "string"}},
                            "required": ["date", "time"],
                        },
                    },
                    "required": ["title", "attendees", "when"],
                }
            },
            "required": ["event"],
        },
    },
}


def call(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        o = json.loads(r.read().decode("utf-8", errors="replace"))
    return o, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="8022")
    ap.add_argument("--model", default="qwen-256k")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    base = "http://%s:%s" % (args.host, args.port)
    print("ENDPOINT=%s MODEL=%s" % (base, args.model))
    print("FORMULA=20 sequential single-tool calls (name+args structural) + 1 multi-tool pick "
          "+ 1 nested-args; GATE=20/20 + correct pick + nested shape")
    print("PARAMS=temperature=0.0 tool_choice=auto penalties neutral")

    fails = []
    for i in range(1, args.count + 1):
        body = {"model": args.model,
                "messages": [{"role": "user",
                              "content": "What is the weather in Paris right now? Use the tool."}],
                "tools": [TOOL], "tool_choice": "auto",
                "max_tokens": 96, "temperature": 0.0}
        try:
            o, wall = call(base, body, args.timeout)
            ch = o["choices"][0]
            tcs = (ch["message"].get("tool_calls") or [])
            ok = ch.get("finish_reason") == "tool_calls" and len(tcs) == 1
            if ok:
                fn = tcs[0]["function"]
                try:
                    a = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    a = None
                ok = (fn.get("name") == "get_weather" and isinstance(a, dict)
                      and isinstance(a.get("location"), str)
                      and "paris" in a["location"].lower())
            print("TC%02d ok=%s finish=%s wall=%.1fs" % (i, ok, ch.get("finish_reason"), wall))
            if not ok:
                fails.append(i)
        except Exception as exc:
            print("TC%02d ok=False ERROR=%s" % (i, str(exc)[:120]))
            fails.append(i)
    print("TOOLCALL20_VERDICT=%d/%d fails=%s" % (args.count - len(fails), args.count, fails))

    body = {"model": args.model,
            "messages": [{"role": "user",
                          "content": "What time is it in Paris right now? Use the appropriate tool."}],
            "tools": [TOOL, TOOL_TIME], "tool_choice": "auto",
            "max_tokens": 200, "temperature": 0.0}
    o, wall = call(base, body, args.timeout)
    ch = o["choices"][0]
    tcs = ch["message"].get("tool_calls") or []
    pick = tcs[0]["function"]["name"] if tcs else None
    try:
        margs = json.loads(tcs[0]["function"]["arguments"]) if tcs else None
    except Exception:
        margs = None
    print("MULTITOOL picked=%s args=%s finish=%s wall=%.1fs" %
          (pick, json.dumps(margs), ch.get("finish_reason"), wall))
    print("MULTITOOL_VERDICT=%s" %
          ("CORRECT_PICK" if pick == "get_time" and isinstance(margs, dict) else "CHECK_RAW"))

    body = {"model": args.model,
            "messages": [{"role": "user",
                          "content": "Create a calendar event titled 'Rig review' on 2026-09-24 "
                                     "at 10:00 with attendees Alex and Ryan. Use the "
                                     "create_calendar_event tool."}],
            "tools": [TOOL_NESTED], "tool_choice": "auto",
            "max_tokens": 300, "temperature": 0.0}
    o, wall = call(base, body, args.timeout)
    ch = o["choices"][0]
    tcs = ch["message"].get("tool_calls") or []
    print("NESTED_RAW_MESSAGE=%s" % json.dumps({
        "role": ch["message"].get("role"), "content": ch["message"].get("content"),
        "tool_calls": tcs, "finish_reason": ch.get("finish_reason")}))
    ok = False
    if tcs:
        try:
            a = json.loads(tcs[0]["function"]["arguments"])
            ev = a.get("event") or {}
            ok = (tcs[0]["function"]["name"] == "create_calendar_event"
                  and isinstance(ev.get("title"), str)
                  and "rig review" in ev["title"].lower()
                  and isinstance(ev.get("attendees"), list) and len(ev["attendees"]) == 2
                  and isinstance(ev.get("when"), dict)
                  and "2026-09-24" in str(ev["when"].get("date", ""))
                  and "10:00" in str(ev["when"].get("time", "")))
        except Exception:
            ok = False
    print("NESTED_VERDICT=%s wall=%.1fs" % ("PASS" if ok else "FAIL", wall))


if __name__ == "__main__":
    main()
