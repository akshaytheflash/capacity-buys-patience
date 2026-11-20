#!/usr/bin/env python3
"""Push per-epoch JSONL results to GitHub using the REST API (no git).

The git-based sync failed twice: first on the 100MB checkpoint limit (GH001),
then on detached-worktree ref resolution. The REST contents API sidesteps both:
we upload only small JSONL files, one at a time.

Usage: python scripts/push_results.py --arm capacity [--token X]
Reads the token from GH_TOKEN or --token, then upserts
  results/raw/<arm>/<run>.jsonl  on branch `results-wip`.
"""
import argparse, base64, json, os, sys, urllib.request, urllib.error
from pathlib import Path

REPO = "akshaytheflash/capacity-buys-patience"
BRANCH = "results-wip"


def api(url, token, method="GET", payload=None):
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if payload is not None:
        body = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    else:
        body = None
    try:
        with urllib.request.urlopen(req, data=body, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"_status": e.code, "_body": e.read()[:200].decode(errors="ignore")}


def ensure_branch(token):
    base = f"https://api.github.com/repos/{REPO}"
    ref = api(f"{base}/git/ref/heads/{BRANCH}", token)
    if ref.get("_status") == 404:
        master = api(f"{base}/git/ref/heads/master", token)
        sha = master.get("object", {}).get("sha")
        if not sha:
            return None
        api(f"{base}/git/refs", token, "POST", {"ref": f"refs/heads/{BRANCH}", "sha": sha})
    return True


def put_file(token, path, content, message):
    base = f"https://api.github.com/repos/{REPO}/contents/{path}"
    cur = api(f"{base}?ref={BRANCH}", token)
    payload = {"message": message, "content": base64.b64encode(content).decode(), "branch": BRANCH}
    if cur.get("sha"):
        payload["sha"] = cur["sha"]
    return api(base, token, "PUT", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--token", default=os.environ.get("GH_TOKEN", ""))
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    if not a.token:
        print("[push] no token", file=sys.stderr); return 1
    ensure_branch(a.token)
    root = Path(a.root)
    files = list(root.glob(f"raw/{a.arm}/*.jsonl"))
    ok = 0
    for f in files:
        if f.stat().st_size > 5_000_000:
            continue
        r = put_file(a.token, f.as_posix(), f.read_bytes(), f"results: {f.name} [skip ci]")
        if r.get("_status") in (200, 201):
            ok += 1
        else:
            print(f"[push] {f.name} failed: {r.get('_status')} {r.get('_body','')[:120]}")
    print(f"[push] {ok}/{len(files)} files -> {BRANCH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
