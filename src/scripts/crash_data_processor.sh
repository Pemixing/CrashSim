#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python -m agent.rag.crash_data_processor \
      --data_dir "${ROOT_DIR}/raw_data" \
      --output_dir "${ROOT_DIR}/out/nhtsa_rag" \
