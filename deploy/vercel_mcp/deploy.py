"""Deploy this folder to Vercel production through the REST API (no CLI needed).

    VERCEL_TOKEN=... python deploy.py [project] [domain]

Uploads every file except tests/, this script and caches, creates a production deployment with the
FastAPI preset (Python runtime), waits for it, then tries to attach the domain (default
<project>.vercel.app) and prints the production aliases. The token is read from the environment only.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

TOKEN = os.environ["VERCEL_TOKEN"]
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = sys.argv[1] if len(sys.argv) > 1 else "stampede-mcp"
DOMAIN = sys.argv[2] if len(sys.argv) > 2 else f"{PROJECT}.vercel.app"
API = "https://api.vercel.com"
SKIP_DIRS = {"tests", "__pycache__", ".vercel", ".git"}
SKIP_FILES = {"deploy.py"}


def call(method, path, body=None, headers=None, raw=None, ok=(200, 201)):
    h = {"Authorization": f"Bearer {TOKEN}"}
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    if body is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(API + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def must(res, what):
    code, body = res
    if code not in (200, 201):
        raise SystemExit(f"{what}: HTTP {code} {json.dumps(body)[:600]}")
    return body


team = must(call("GET", "/v2/user"), "user")["user"].get("defaultTeamId")
q = f"?teamId={team}" if team else "?"
files = []
for root, dirs, names in os.walk(HERE):
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
    for n in sorted(names):
        rel = os.path.relpath(os.path.join(root, n), HERE)
        if n.startswith(".") or n in SKIP_FILES or n.endswith(".pyc"):
            continue
        b = open(os.path.join(root, n), "rb").read()
        sha = hashlib.sha1(b).hexdigest()
        must(call("POST", "/v2/files" + q, raw=b, headers={"x-vercel-digest": sha, "Content-Type": "application/octet-stream"}),
             f"upload {rel}")
        files.append({"file": rel, "sha": sha, "size": len(b)})
print("uploaded", len(files), "files")
t0 = time.time()
d = must(call("POST", "/v13/deployments" + q + "&skipAutoDetectionConfirmation=1",
              {"name": PROJECT, "files": files, "target": "production", "projectSettings": {"framework": "fastapi"}}),
         "create deployment")
did = d["id"]
s = d
for _ in range(200):
    s = must(call("GET", f"/v13/deployments/{did}" + q), "poll")
    if s.get("readyState") in ("READY", "ERROR", "CANCELED"):
        break
    time.sleep(3)
print("state", s.get("readyState"), f"after {time.time() - t0:.0f}s", "deployment", did)
if s.get("readyState") != "READY":
    code, ev = call("GET", f"/v3/deployments/{did}/events" + q + "&limit=-1")
    for e in (ev if isinstance(ev, list) else [])[-60:]:
        txt = (e.get("payload") or {}).get("text") or e.get("text")
        if txt:
            print("  |", txt[:300])
    raise SystemExit(1)
code, doms = call("GET", f"/v9/projects/{PROJECT}/domains" + q)
names = [x.get("name") for x in doms.get("domains", [])] if code == 200 else []
if DOMAIN in names:
    print("domain", DOMAIN, "already attached to", PROJECT)
else:
    code, body = call("POST", f"/v10/projects/{PROJECT}/domains" + q, {"name": DOMAIN})
    print("attach domain", DOMAIN, "->", code, (body.get("error") or {}).get("message", "ok"))
    code, doms = call("GET", f"/v9/projects/{PROJECT}/domains" + q)
    names = [x.get("name") for x in doms.get("domains", [])] if code == 200 else []
print("project domains", names)
proj = must(call("GET", f"/v9/projects/{PROJECT}" + q), "project")
print("protection", json.dumps({k: proj.get(k) for k in ("ssoProtection", "passwordProtection")}))
print("aliases", s.get("alias"))
