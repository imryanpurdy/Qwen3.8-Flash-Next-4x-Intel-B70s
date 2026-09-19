#!/usr/bin/env python3
"""12-way concurrency probe -> result to /tmp/probe12.out (survives plink/heredoc quoting issues)."""
import urllib.request, json, time, concurrent.futures

URL = 'http://localhost:8021/v1/completions'

def one(i):
    d = json.dumps({'model': 'qwen3.8-flash-next',
                    'prompt': 'The history of computing began when',
                    'max_tokens': 256, 'temperature': 0}).encode()
    r = urllib.request.Request(URL, data=d, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    u = json.load(urllib.request.urlopen(r, timeout=300))['usage']
    return u['completion_tokens'], time.time() - t0

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(12) as ex:
    res = list(ex.map(one, range(12)))
tot = sum(r[0] for r in res)
wall = time.time() - t0
out = f'12-way agg={tot} wall={wall:.2f}s rate={tot/wall:.1f}'
print(out, flush=True)
open('/tmp/probe12.out', 'w').write(out + '\n')
