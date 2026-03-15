#!/usr/bin/env bash
# Build the WASM web app for deployment.
# Outputs optimised assets to dist/ with relative paths so the app works
# when served from any subdirectory (e.g. GitHub Pages at /repo/).
set -euo pipefail

cd "$(dirname "$0")"

echo "Building WASM release..."
trunk build --release --public-url ./

echo ""
echo "Done. Output is in: dist/"
echo "Serve locally with:"
echo "  python3 -m http.server 8080 --directory dist"
