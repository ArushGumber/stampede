#!/usr/bin/env bash
# Keep a public tunnel to the local demo server (127.0.0.1:8765) alive via localhost.run, and write the current
# public URL to docs/PUBLIC_URL.txt. The free domain rotates; the URL file always holds the latest one.
cd "$(dirname "$0")/.."
(
  while true; do
    ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        -R 80:127.0.0.1:8765 nokey@localhost.run > logs/tunnel_raw.log 2>&1
    echo "tunnel closed $(date); reconnecting in 10s" >> logs/tunnel.log
    sleep 10
  done
) &
while true; do
  url=$(sed 's/\x1b\[[0-9;]*m//g' logs/tunnel_raw.log 2>/dev/null | grep -oE "https://[a-z0-9.]+\.lhr\.life" | tail -1)
  if [ -n "$url" ] && ! grep -q "$url" docs/PUBLIC_URL.txt 2>/dev/null; then
    echo "$url  $(date '+%Y-%m-%d %H:%M:%S')" > docs/PUBLIC_URL.txt
    echo "URL $url $(date)" >> logs/tunnel.log
  fi
  sleep 5
done
