"""Pack the Python the browser simulator needs into public/py/stampede-sim.zip.

The archive holds the exchange, the episode runner, the response tables and the headline bank, laid out as in the
repository, so stampede.sim.live and stampede.mcp_server.arena find their files exactly as they do on the server.
Standard library only, so any Python 3 can run it.
"""
import glob
import os
import sys
import zipfile

WEB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(WEB)
OUT = os.path.join(WEB, "public", "py", "stampede-sim.zip")

FILES = [
    "stampede/__init__.py",
    "stampede/agents/__init__.py",
    "stampede/agents/prompting.py",
    "stampede/sim/__init__.py",
    "stampede/sim/episode.py",
    "stampede/sim/surrogate.py",
    "stampede/sim/live.py",
    "stampede/mcp_server/__init__.py",
    "stampede/mcp_server/arena.py",
    "data/headlines.json",
]
GLOBS = ["stampede/exchange/*.py", "results/surrogates/*.json"]


def main() -> int:
    paths = list(FILES)
    for g in GLOBS:
        paths += sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, g)))
    missing = [p for p in paths if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        print("missing:", ", ".join(missing), file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            # Fixed timestamps keep the archive byte-identical between builds.
            info = zipfile.ZipInfo(p, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with open(os.path.join(ROOT, p), "rb") as f:
                z.writestr(info, f.read())
    os.replace(tmp, OUT)
    print(f"{os.path.relpath(OUT, WEB)}: {len(paths)} files, {os.path.getsize(OUT) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
