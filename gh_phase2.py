"""Fetch comments for shortlisted issues + second search sweep + commit lookup."""
import json, time, urllib.request, urllib.parse, os

OUT = r"C:\Users\imrya\flashnext-recipe\docs\gh_phase2.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
HDRS = {"Accept": "application/vnd.github+json", "User-Agent": "hermes-research"}

def get(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

out = {}

# 1) comments for key issues
CMT_ISSUES = [55533, 53480, 40926, 56815, 36826, 18431, 45388, 42371, 53726, 57680, 57562, 55617]
for n in CMT_ISSUES:
    try:
        d = get(f"https://api.github.com/repos/vllm-project/vllm/issues/{n}/comments?per_page=30")
        out[f"cmt_{n}"] = [{"user": c["user"]["login"], "created": c["created_at"],
                            "body": (c.get("body") or "")} for c in d]
        print(f"cmt_{n}: {len(d)}")
    except Exception as e:
        out[f"cmt_{n}"] = {"error": str(e)}
        print(f"cmt_{n} ERROR {e}")
    time.sleep(1.2)

# 2) PLE-related side issues/PRs
for n in [54371, 53896, 54722]:
    try:
        d = get(f"https://api.github.com/repos/vllm-project/vllm/issues/{n}")
        out[n] = {"title": d["title"], "state": d["state"], "created": d["created_at"],
                  "body": (d.get("body") or "")[:8000], "comments": d["comments"],
                  "is_pr": "pull_request" in d, "url": d["html_url"]}
        print(f"#{n} [{d['state']}] :: {d['title']}")
    except Exception as e:
        out[n] = {"error": str(e)}
        print(f"#{n} ERROR {e}")
    time.sleep(1.2)

# 3) commit lookup (may not exist upstream)
try:
    d = get("https://api.github.com/repos/vllm-project/vllm/commits/g76cfe1cd8")
    out["commit_g76cfe1cd8"] = {"sha": d.get("sha"), "date": d.get("commit", {}).get("committer", {}).get("date"),
                                "msg": (d.get("commit", {}).get("message") or "")[:300]}
    print("commit found:", out["commit_g76cfe1cd8"])
except Exception as e:
    out["commit_g76cfe1cd8"] = {"error": str(e)}
    print("commit lookup:", e)
time.sleep(1.2)

# 4) second search sweep
QUERIES = [
    'repo:vllm-project/vllm "no available shared memory broadcast block"',
    'repo:vllm-project/vllm "Qwen3.8-Flash-Next"',
    'repo:vllm-project/vllm ple offload stall OR hang OR deadlock',
    'repo:vllm-project/vllm "VLLM_ENABLE_V1_MULTIPROCESSING"',
    'repo:vllm-project/vllm get_output_async',
    'repo:vllm-project/vllm "output processing time" OR "output processing" backlog',
    'repo:vllm-project/vllm XPU concurrent decode hang OR stall OR freeze',
    'repo:vllm-project/vllm async scheduling regression 0.26 concurrency throughput',
    'repo:vllm-project/vllm "throughput degradation" over time rounds',
    'repo:vllm-project/vllm "num_output_placeholders"',
    'repo:vllm-project/vllm "duplicate PLE"',
    'repo:vllm-project/vllm "PleOffload"',
]
sres = {}
for q in QUERIES:
    try:
        url = "https://api.github.com/search/issues?q=" + urllib.parse.quote(q) + "&per_page=8&sort=updated&order=desc"
        d = get(url)
        items = [{"number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"],
                  "is_pr": "pull_request" in i, "updated": i["updated_at"]} for i in d.get("items", [])]
        sres[q] = {"total": d.get("total_count"), "items": items}
        print(f"[search] total={d.get('total_count')} :: {q}")
        for it in items:
            print(f"    #{it['number']} {'PR ' if it['is_pr'] else '   '} {it['title']}")
    except Exception as e:
        sres[q] = {"error": str(e)}
        print(f"search ERROR {q}: {e}")
    time.sleep(7)

out["searches2"] = sres
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("saved", OUT)
