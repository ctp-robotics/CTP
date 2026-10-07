#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG=""
GPUS=0
STAGE=""
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/train.sh [--gpus 0,1] [--config path] [--stage pretrain|policy] [other args]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help) usage; exit 0 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpus=*) GPUS="${1#*=}"; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --stage) STAGE="$2"; shift 2 ;;
    --stage=*) STAGE="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPUS"

if [[ -n "$CONFIG" ]]; then ARGS+=(--config "$CONFIG"); fi
if [[ -n "$STAGE" ]]; then ARGS+=(--stage "$STAGE"); fi
exec python train.py "${ARGS[@]}"
