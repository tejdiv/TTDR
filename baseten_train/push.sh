#!/bin/bash
# Bundle TTDR code into baseten_train/ and push to Baseten.
# Usage: bash push.sh config_download.py
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TTDR_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:?Usage: bash push.sh <config.py>}"

echo "Bundling TTDR code..."
cp -r "$TTDR_ROOT/octo" "$SCRIPT_DIR/octo"
cp -r "$TTDR_ROOT/recap" "$SCRIPT_DIR/recap"
cp -r "$TTDR_ROOT/scripts" "$SCRIPT_DIR/scripts"
cp -r "$TTDR_ROOT/configs" "$SCRIPT_DIR/configs"
cp -r "$TTDR_ROOT/tests" "$SCRIPT_DIR/tests"
cp "$TTDR_ROOT/setup.py" "$SCRIPT_DIR/setup.py"
[ -f "$TTDR_ROOT/setup.cfg" ] && cp "$TTDR_ROOT/setup.cfg" "$SCRIPT_DIR/setup.cfg"
[ -f "$TTDR_ROOT/pyproject.toml" ] && cp "$TTDR_ROOT/pyproject.toml" "$SCRIPT_DIR/pyproject.toml"

echo "Pushing to Baseten..."
cd "$SCRIPT_DIR"
truss train push "$CONFIG"

echo "Cleaning up bundled code..."
rm -rf "$SCRIPT_DIR/octo" "$SCRIPT_DIR/recap" "$SCRIPT_DIR/scripts" "$SCRIPT_DIR/configs" "$SCRIPT_DIR/tests"
rm -f "$SCRIPT_DIR/setup.py" "$SCRIPT_DIR/setup.cfg" "$SCRIPT_DIR/pyproject.toml"

echo "Done."
