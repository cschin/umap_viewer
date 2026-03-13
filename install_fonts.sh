#!/usr/bin/env bash
# Copy SF Mono from macOS system fonts into fonts/AndaleMono.ttf.
# Run once before building; AndaleMono.ttf is not included in this repo.
set -euo pipefail

SFMONO="/System/Library/Fonts/SFNSMono.ttf"
OUT="$(dirname "$0")/fonts/SFNSMono.ttf"

if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: macOS only. On other platforms copy any monospace TTF to fonts/AndaleMono.ttf." >&2
    exit 1
fi

if [[ ! -f "$SFMONO" ]]; then
    echo "ERROR: $SFMONO not found." >&2
    exit 1
fi

cp "$SFMONO" "$OUT"
echo "Copied SF Mono -> $OUT"
echo "You can now run: cargo build"
