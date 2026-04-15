#!/bin/bash
# Serve the web demo locally
# Usage: ./serve.sh [port]
PORT=${1:-8080}
echo "Open http://localhost:$PORT"
cd "$(dirname "$0")"
python3 -m http.server $PORT
