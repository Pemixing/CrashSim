#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_DIR="${1:-${ROOT_DIR}/out/gen/val_full}"
OUT_DIR="${2:-${ROOT_DIR}/out/nusc_crash_val}"
VERSION="${3:-trainval-crash}"

# 原始 nuScenes 根目录（应包含 maps/ 与 v1.0-* 元数据子目录）
SOURCE_NUSC_DIR="/path/to/nuscenes/trainval"

# 选择 python：优先使用 adv conda env（保证 nuscenes-devkit 可用），否则退化到 python3。
PY_BIN="${PY_BIN:-}"
if [[ -z "${PY_BIN}" ]]; then
  if [[ -x "${HOME}/miniforge3/envs/adv/bin/python" ]]; then
    PY_BIN="${HOME}/miniforge3/envs/adv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  else
    echo "[error] cannot find python interpreter (python3/python)." >&2
    exit 1
  fi
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] PY_BIN=${PY_BIN}"
echo "[info] INPUT_DIR=${INPUT_DIR}"
echo "[info] OUT_DIR=${OUT_DIR}"
echo "[info] VERSION=${VERSION}"
echo "[info] SOURCE_NUSC_DIR=${SOURCE_NUSC_DIR}"

ARGS=(
  --input "${INPUT_DIR}"
  --only_adv_success
  --out "${OUT_DIR}"
  --version "${VERSION}"
  --source_nusc_dir "${SOURCE_NUSC_DIR}"
  --link_maps symlink
)

"${PY_BIN}" "${ROOT_DIR}/src/build_nusc_crash.py" \
  "${ARGS[@]}" \
  --verify

