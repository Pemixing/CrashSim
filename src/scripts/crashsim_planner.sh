#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# 兼容两种调用方式：
# 1) 位置参数：crashsim_planner.sh [DATAROOT] [VERSION] [PLANNER] [MAX_SEQS] [OUT_DIR] [EXTRA_ARGS...]
# 2) 直接透传 flags：crashsim_planner.sh --map_flip_singapore ...（此时使用默认 DATAROOT/OUT_DIR 等）
DEFAULT_DATAROOT="${ROOT_DIR}/out/nusc_crash_val"
DEFAULT_VERSION="trainval-crash"
DEFAULT_PLANNER="PDM-Closed"
DEFAULT_MAX_SEQS="-1"

EXTRA_ARGS=()
POSITIONAL_OUT_DIR=""
if [[ "${1:-}" == --* ]]; then
  DATAROOT="${DEFAULT_DATAROOT}"
  VERSION="${DEFAULT_VERSION}"
  PLANNER="${DEFAULT_PLANNER}"   # hardcode | IDM | PDM-Closed | replay | IL | IL-multi | Diffusion
  MAX_SEQS="${DEFAULT_MAX_SEQS}" # -1 means all
  EXTRA_ARGS=( "$@" )
else
  DATAROOT="${1:-${DEFAULT_DATAROOT}}"
  VERSION="${2:-${DEFAULT_VERSION}}"
  PLANNER="${3:-${DEFAULT_PLANNER}}"   # hardcode | IDM | PDM-Closed | replay | IL | IL-multi | Diffusion
  MAX_SEQS="${4:-${DEFAULT_MAX_SEQS}}" # -1 means all
  POSITIONAL_OUT_DIR="${5:-}"
  if [[ "$#" -ge 6 ]]; then
    EXTRA_ARGS=( "${@:6}" )
  fi
fi

# OUT_DIR 须在 PLANNER 确定之后再展开（set -u 下不可提前引用未绑定变量）
# 优先级：位置参数 $5 > 环境变量 OUT_DIR > 按 planner 名生成的默认路径
DEFAULT_OUT_DIR="${ROOT_DIR}/out/crashsim_results/crashsim_planner_${PLANNER}"
if [[ -n "${POSITIONAL_OUT_DIR}" ]]; then
  OUT_DIR="${POSITIONAL_OUT_DIR}"
else
  OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
fi

# 选择 python：优先使用 adv conda env，否则退化到 python3/python。
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

mkdir -p "${OUT_DIR}"

# 可用环境变量覆写（与 crashsim_planner.py 参数同名/同语义）
PAST_LEN="${PAST_LEN:-4}"
FUTURE_LEN="${FUTURE_LEN:-12}"
SEQ_INTERVAL="${SEQ_INTERVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
# 默认保存视频；如需关闭可显式设置 VIZ_VIDEO=0
VIZ_VIDEO="${VIZ_VIDEO:-1}"
# 默认开启 singapore flip（如需关闭可显式设置 MAP_FLIP_SINGAPORE=0）
MAP_FLIP_SINGAPORE="${MAP_FLIP_SINGAPORE:-1}"

# Diffusion planner（DiT + EDM）相关配置，仅当 PLANNER=Diffusion 时生效。
PLANNER_CKPT="/path/to/diffusion_planner_best.pth"
DIFF_DIM="${DIFF_DIM:-128}"
DIFF_DEPTH="${DIFF_DEPTH:-4}"
DIFF_HEADS="${DIFF_HEADS:-4}"
DIFF_SAMPLE_STEPS="${DIFF_SAMPLE_STEPS:-32}"
DIFF_NUM_SAMPLES="${DIFF_NUM_SAMPLES:-1}"

# 评估报告（EvalPlannerAgent）相关开关
EVAL_REPORT="${EVAL_REPORT:-1}"
EVAL_ONLY="${EVAL_ONLY:-1}"
EVAL_RESULTS="${EVAL_RESULTS:-}"
EVAL_USE_LLM="${EVAL_USE_LLM:-1}"
LLM_EVAL_API_BASE="https://your-api-endpoint/v1"
LLM_EVAL_API_KEY=""
LLM_EVAL_MODEL="gpt-4o"
EVAL_MAX_FAILURES_IN_PROMPT="${EVAL_MAX_FAILURES_IN_PROMPT:-120}"
EVAL_MAX_ODD_CELLS_IN_PROMPT="${EVAL_MAX_ODD_CELLS_IN_PROMPT:-30}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] PY_BIN=${PY_BIN}"
echo "[info] DATAROOT=${DATAROOT}"
echo "[info] VERSION=${VERSION}"
echo "[info] PLANNER=${PLANNER}"
echo "[info] MAX_SEQS=${MAX_SEQS}"
echo "[info] OUT_DIR=${OUT_DIR}"

if [[ "${PLANNER}" == "Diffusion" || "${PLANNER}" == "IL" || "${PLANNER}" == "IL-multi" ]]; then
  echo "[info] PLANNER_CKPT=${PLANNER_CKPT}"
  if [[ -n "${PLANNER_CKPT}" && ! -f "${PLANNER_CKPT}" ]]; then
    echo "[warn] PLANNER_CKPT set but file not found: ${PLANNER_CKPT} (planner will run UNTRAINED)."
  fi
fi
if [[ "${PLANNER}" == "Diffusion" ]]; then
  echo "[info] DIFF_DIM=${DIFF_DIM} DIFF_DEPTH=${DIFF_DEPTH} DIFF_HEADS=${DIFF_HEADS} DIFF_SAMPLE_STEPS=${DIFF_SAMPLE_STEPS} DIFF_NUM_SAMPLES=${DIFF_NUM_SAMPLES}"
fi
echo "[info] EVAL_ONLY=${EVAL_ONLY}"
echo "[info] EVAL_REPORT=${EVAL_REPORT}"
echo "[info] EVAL_USE_LLM=${EVAL_USE_LLM}"
echo "[info] LLM_EVAL_API_BASE=${LLM_EVAL_API_BASE}"
echo "[info] LLM_EVAL_MODEL=${LLM_EVAL_MODEL}"
if [[ -n "${LLM_EVAL_API_KEY}" ]]; then
  echo "[info] LLM_EVAL_API_KEY=set"
else
  echo "[info] LLM_EVAL_API_KEY=unset"
fi
if [[ "${EVAL_USE_LLM}" == "1" && -z "${LLM_EVAL_API_KEY}" ]]; then
  echo "[warn] EVAL_USE_LLM=1 but LLM_EVAL_API_KEY is empty; set it in this script or pass --eval_api_key."
fi

ARGS=(
  --dataroot "${DATAROOT}"
  --version "${VERSION}"
  --planner "${PLANNER}"
  --planner_ckpt "${PLANNER_CKPT}"
  --max_seqs "${MAX_SEQS}"
  --out "${OUT_DIR}"
  --past_len "${PAST_LEN}"
  --future_len "${FUTURE_LEN}"
  --seq_interval "${SEQ_INTERVAL}"
  --num_workers "${NUM_WORKERS}"
  --viz
  --eval_api_base "${LLM_EVAL_API_BASE}"
  --eval_model "${LLM_EVAL_MODEL}"
  --eval_max_failures_in_prompt "${EVAL_MAX_FAILURES_IN_PROMPT}"
  --eval_max_odd_cells_in_prompt "${EVAL_MAX_ODD_CELLS_IN_PROMPT}"
  --eval_timeout "${EVAL_TIMEOUT}"
)

if [[ "${VIZ_VIDEO}" == "1" ]]; then
  ARGS+=( --viz_video )
fi

if [[ "${MAP_FLIP_SINGAPORE}" == "1" ]]; then
  ARGS+=( --map_flip_singapore )
fi

if [[ "${EVAL_REPORT}" == "1" ]]; then
  ARGS+=( --eval_report )
else
  ARGS+=( --no_eval_report )
fi

if [[ "${EVAL_USE_LLM}" == "1" ]]; then
  ARGS+=( --eval_use_llm )
else
  ARGS+=( --no_eval_use_llm )
fi

if [[ "${EVAL_ONLY}" == "1" ]]; then
  ARGS+=( --eval_only )
  if [[ -n "${EVAL_RESULTS}" ]]; then
    ARGS+=( --eval_results "${EVAL_RESULTS}" )
  fi
fi

if [[ -n "${LLM_EVAL_API_KEY}" ]]; then
  ARGS+=( --eval_api_key "${LLM_EVAL_API_KEY}" )
fi

# Diffusion planner 专属参数（仅当 PLANNER=Diffusion 时传递）。
if [[ "${PLANNER}" == "Diffusion" ]]; then
  ARGS+=(
    --diff_dim "${DIFF_DIM}"
    --diff_depth "${DIFF_DEPTH}"
    --diff_heads "${DIFF_HEADS}"
    --diff_sample_steps "${DIFF_SAMPLE_STEPS}"
    --diff_num_samples "${DIFF_NUM_SAMPLES}"
  )
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  ARGS+=( "${EXTRA_ARGS[@]}" )
fi

echo "[info] running crashsim_planner.py ..."
"${PY_BIN}" "${ROOT_DIR}/src/crashsim_planner.py" "${ARGS[@]}"

echo "[ok] done. outputs in: ${OUT_DIR}"

