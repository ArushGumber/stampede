"""Sequential job queue for the single GPU this project may use (GPU 0).

Append shell commands, one per line, to jobs/queue.txt. Each runs with CUDA_VISIBLE_DEVICES=0, its output
goes to logs/jobs/<n>_<slug>.log, and finished lines are recorded in jobs/done.txt with exit code and time.
Lines starting with # are skipped. The runner polls for new lines, so jobs can be added at any time.
"""
import os
import re
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
Q, DONE = os.path.join(ROOT, "jobs", "queue.txt"), os.path.join(ROOT, "jobs", "done.txt")
os.makedirs(os.path.join(ROOT, "jobs"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "logs", "jobs"), exist_ok=True)
open(Q, "a").close()
open(DONE, "a").close()


def done_set():
    return {l.split("\t", 3)[-1].rstrip("\n") for l in open(DONE) if l.strip()}


n = len(done_set())
while True:
    finished = done_set()
    pending = [l.rstrip("\n") for l in open(Q) if l.strip() and not l.startswith("#")]
    todo = [l for l in pending if l not in finished]
    if not todo:
        time.sleep(15)
        continue
    cmd = todo[0]
    n += 1
    slug = re.sub(r"[^A-Za-z0-9]+", "_", cmd.split("scripts/")[-1])[:60]
    log = os.path.join(ROOT, "logs", "jobs", f"{n:03d}_{slug}.log")
    t0 = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    with open(DONE, "a") as f:
        f.write(f"{rc}\t{time.strftime('%H:%M:%S')}\t{round(time.time() - t0)}s\t{cmd}\n")
