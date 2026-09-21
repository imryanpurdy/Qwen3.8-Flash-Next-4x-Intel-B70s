"""Sweep vLLM GitHub tracker via search API for the 12-concurrent output-stall symptom family."""
import json, time, urllib.request, urllib.parse, os, sys

OUT = r"C:\Users\imrya\flashnext-recipe\docs\gh_sweep_results.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

QUERIES = [
    'repo:vllm-project/vllm "stuck running" concurrent',
    'repo:vllm-project/vllm "EngineCore" "idle"',
    'repo:vllm-project/vllm "output processing" stall v1',
    'repo:vllm-project/vllm "VLLM_OUTPUT_PROCESS"',
    'repo:vllm-project/vllm output_handler deadlock OR hang OR slow',
    'repo:vllm-project/vllm "shared memory broadcast"',
    'repo:vllm-project/vllm "shm_broadcast"',
    'repo:vllm-project/vllm zmq "message lost" OR "request lost" OR dropped',
    'repo:vllm-project/vllm completions arrive in waves OR burst concurrency',
    'repo:vllm-project/vllm throughput cliff concurrency scheduler',
    'repo:vllm-project/vllm GDN hybrid concurrency stall OR slow OR freeze',
    'repo:vllm-project/vllm "linear attention" concurrent throughput degradation',
    'repo:vllm-project/vllm blocking EngineCore event loop',
    'repo:vllm-project/vllm "duplicate" prefill dp_rank',
    'repo:vllm-project/vllm async_llm output handler slow high concurrency',
    'repo:vllm-project/vllm "12 concurrent"',
    'repo:vllm-project/vllm scheduler does not run full batch concurrency',
    'repo:vllm-project/vllm chunked preemption throughput collapse',
    'repo:vllm-project/vllm "KV cache" groups hybrid preemption',
    'repo:vllm-project/vllm EngineCore stuck input queue',
]

def gh_search(q, per_page=8):
    url = "https://api.github.com/search/issues?q=" + urllib.parse.quote(q) + f"&per_page={per_page}&sort=updated&order=desc"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "hermes-research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

results = {}
for q in QUERIES:
    try:
        d = gh_search(q)
        items = [{"number": i["number"], "title": i["title"], "state": i["state"],
                  "url": i["html_url"], "created": i["created_at"], "updated": i["updated_at"],
                  "is_pr": "pull_request" in i, "labels": [l["name"] for l in i.get("labels", [])]}
                 for i in d.get("items", [])]
        results[q] = {"total": d.get("total_count"), "items": items, "error": None}
        print(f"[{len(results)}] total={d.get('total_count')} :: {q}")
        for it in items:
            print(f"    #{it['number']} {'PR ' if it['is_pr'] else '   '} {it['title']}")
    except Exception as e:
        results[q] = {"error": str(e)}
        print(f"ERROR {q}: {e}")
    time.sleep(7)  # unauthenticated search API rate limit

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT}")
