#!/usr/bin/env bash
set -euo pipefail

# publish_web.sh — build WASM and deploy to the web branch (GitHub Pages)

echo "==> Building WASM..."
trunk build --release --public-url ./

NEW_JS=$(ls dist/*.js)
NEW_WASM=$(ls dist/*.wasm)

echo "==> Switching to web branch..."
git checkout web

echo "==> Removing old .js and .wasm files..."
git rm -f *.js *.wasm 2>/dev/null || true

echo "==> Copying new build artifacts..."
cp dist/index.html dist/*.js dist/*.wasm .

NEW_JS_BASE=$(basename "$NEW_JS")
NEW_WASM_BASE=$(basename "$NEW_WASM")

echo "==> Staging files..."
git add index.html "$NEW_JS_BASE" "$NEW_WASM_BASE"

echo "==> Committing..."
git commit -m "deploy WASM build $(date -u '+%Y-%m-%d %H:%M UTC')"

echo "==> Pushing to origin/web..."
git push origin web

echo "==> Switching back to main..."
git checkout main

echo "Done. GitHub Pages will update shortly."
