"""Fetch full bodies of shortlisted vLLM issues/PRs (and PR review comments for #53899)."""
import json, time, urllib.request, urllib.parse, os

OUT = r"C:\Users\imrya\flashnext-recipe\docs\gh_details.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
HDRS = {"Accept": "application/vnd.github+json", "User-Agent": "hermes-research"}

NUMS = [53899, 57562, 36826, 53480, 56815, 55533, 57680, 40926, 46121, 49628, 38079, 18431, 45388, 42371, 57378, 53726, 55617]

def get(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

out = {}
for n in NUMS:
    try:
        d = get(f"https://api.github.com/repos/vllm-project/vllm/issues/{n}")
        body = (d.get("body") or "")[:6000]
        out[n] = {"title": d["title"], "state": d["state"], "created": d["created_at"],
                  "updated": d["updated_at"], "comments": d["comments"], "is_pr": "pull_request" in d,
                  "labels": [l["name"] for l in d.get("labels", [])], "body": body,
                  "url": d["html_url"]}
        print(f"#{n} [{d['state']}] cmt={d['comments']} :: {d['title']}")
        print("  " + body.replace("\n", " ")[:400])
    except Exception as e:
        out[n] = {"error": str(e)}
        print(f"#{n} ERROR {e}")
    time.sleep(1.2)

# PR 53899 review comments + issue comments (most relevant: duplicate PLE request handler)
for url, key in [("https://api.github.com/repos/vllm-project/vllm/pulls/53899/comments", "pr53899_review"),
                 ("https://api.github.com/repos/vllm-project/vllm/issues/53899/comments", "pr53899_issue_cmts")]:
    try:
        d = get(url)
        out[key] = [{"user": c["user"]["login"], "body": (c.get("body") or "")[:1500],
                     "path": c.get("path"), "line": c.get("line"), "created": c["created_at"]}
                    for c in d]
        print(f"{key}: {len(d)} comments")
    except Exception as e:
        out[key] = {"error": str(e)}
        print(f"{key} ERROR {e}")
    time.sleep(1.2)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("saved", OUT)
