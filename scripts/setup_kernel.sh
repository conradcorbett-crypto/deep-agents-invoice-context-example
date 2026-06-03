#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync
uv run python -m ipykernel install --user --name invoice-context-demo --display-name "Python (invoice-context-demo)"
echo "Kernel registered. In Cursor, open the kernel picker and choose Python (invoice-context-demo)."
