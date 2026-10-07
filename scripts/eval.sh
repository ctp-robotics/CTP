#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python tools/eval_open_loop.py \
  --checkpoint "${1:-outputs/your_run/checkpoints/best.pt}" \
  --data-root "${2:-}" \
  --split val \
  --max-batches 20
