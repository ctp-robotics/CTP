#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-config/policy.yaml}"
GPUS="${2:-1}"

CUDA_VISIBLE_DEVICES="$GPUS" python tools/precompute_policy_latents.py --config "$CONFIG" --split train --overwrite
CUDA_VISIBLE_DEVICES="$GPUS" python tools/precompute_policy_latents.py --config "$CONFIG" --split val --overwrite
