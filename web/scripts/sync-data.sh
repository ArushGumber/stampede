#!/usr/bin/env bash
# Copy the exported site data (results/site) into web/dist/data so the built site works on any static host
# (for example Vercel) without the API. Run after `npm run build`; re-run whenever results/site changes.
#   web/scripts/sync-data.sh [source_dir]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
web="$(dirname "$here")"
src="${1:-${STAMPEDE_SITE_DATA:-$(dirname "$web")/results/site}}"
dst="$web/dist/data"
if [ ! -d "$web/dist" ]; then echo "web/dist does not exist; run npm run build first" >&2; exit 1; fi
if [ ! -f "$src/meta.json" ]; then echo "no meta.json in $src; nothing to sync" >&2; exit 1; fi
rm -rf "$dst.tmp"
mkdir -p "$dst.tmp"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '*.tmp' "$src/" "$dst.tmp/"
else
  cp -R "$src/." "$dst.tmp/"
fi
rm -rf "$dst"
mv "$dst.tmp" "$dst"
n_ep=$(find "$dst/episodes" -name 'ep_*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "synced $src -> $dst ($(ls "$dst/results" 2>/dev/null | wc -l | tr -d ' ') result files, $n_ep episodes, $(du -sh "$dst" | cut -f1))"
