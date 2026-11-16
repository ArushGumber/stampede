#!/usr/bin/env bash
# Build a self-contained static copy of the site: the app, the recorded data, and the simulator bundle that runs
# markets in the visitor's browser. Upload web/dist-static/ as is to any static host (Vercel, a static Hugging Face
# Space, GitHub Pages). The demo video is left out; link it with VIDEO_URL in src/config.ts instead.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=dist-static

npm run build
rm -rf "$OUT"
mkdir -p "$OUT/data"
rsync -a --exclude 'data/' dist/ "$OUT/"
rsync -a --exclude '*.webm' ../results/site/ "$OUT/data/"

for f in index.html py/stampede-sim.zip data/meta.json data/episodes/index.json; do
  test -s "$OUT/$f" || { echo "missing $OUT/$f" >&2; exit 1; }
done
echo "static site: web/$OUT ($(du -sh "$OUT" | cut -f1), $(find "$OUT" -type f | wc -l) files)"
