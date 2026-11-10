#!/bin/bash
# Build render_ball.so (do not use -static on Linux/WSL; it conflicts with -shared)
set -e
cd "$(dirname "$0")/../tools"
g++ -shared -O3 -fPIC -m64 render_ball.cpp -o render_ball.so
ls -lh render_ball.so
echo "OK: tools/render_ball.so"
