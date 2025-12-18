#!/usr/bin/env bash
set -euo pipefail

# SiInfer 根目录（假设 install.sh 放在 SiInfer 根）
SII_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SII_ROOT}/.." && pwd)"

# 约定：ubi-llm-eval-beta 与 SiInfer 同级
DEFAULT_CONFIG_ROOT="${PARENT_DIR}/ubi-llm-eval-beta"
DEFAULT_TASK_DEFINE="${DEFAULT_CONFIG_ROOT}/workflows/task_define.py"

# ====== 下面这些默认值你可以按需改 ======
DEFAULT_QWENCODER_ROOT="/workspace/generalcodeeval-main-qwencoder-eval/qwencoder-eval/"
DEFAULT_WORKSPACE="/workspace/workspace/"
DEFAULT_VENV_ROOT="/workspace/venv/"
DEFAULT_OUTPUT_ROOT="/workspace/workspace/"
DEFAULT_REQUIREMENTS_MAP="${SII_ROOT}/requirement_map.json"
DEFAULT_SIINFER_REPO_ROOT="${SII_ROOT}"
# =======================================

# 允许外部覆盖（不传就用默认）
# 支持两种命名风格：小写/大写
DEFAULT_UV_BIN=""

config_root="${config_root:-${CONFIG_ROOT:-$DEFAULT_CONFIG_ROOT}}"
qwencoder_root="${qwencoder_root:-${QWENCODER_ROOT:-$DEFAULT_QWENCODER_ROOT}}"
workspace="${workspace:-${WORKSPACE:-$DEFAULT_WORKSPACE}}"
venv_root="${venv_root:-${VENV_ROOT:-$DEFAULT_VENV_ROOT}}"
output_root="${output_root:-${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}}"
task_define="${task_define:-${TASK_DEFINE:-$DEFAULT_TASK_DEFINE}}"
requirements_map="${requirements_map:-${REQUIREMENTS_MAP:-$DEFAULT_REQUIREMENTS_MAP}}"
siinfer_repo_root="${siinfer_repo_root:-${SIINFER_REPO_ROOT:-$DEFAULT_SIINFER_REPO_ROOT}}"
uv_bin="${uv_bin:-${UV_BIN:-$DEFAULT_UV_BIN}}"

PREPARE_PY="${SII_ROOT}/prepare_benchmark.py"

# ---- sanity check ----
if [[ ! -f "${PREPARE_PY}" ]]; then
  echo "[ERR] prepare_benchmark.py not found: ${PREPARE_PY}" >&2
  exit 1
fi
if [[ ! -d "${config_root}" ]]; then
  echo "[ERR] config_root not found: ${config_root}" >&2
  echo "      (expected ubi-llm-eval-beta to be sibling of SiInfer)" >&2
  exit 1
fi
if [[ ! -f "${task_define}" ]]; then
  echo "[ERR] task_define not found: ${task_define}" >&2
  exit 1
fi
if [[ ! -f "${requirements_map}" ]]; then
  echo "[ERR] requirements_map not found: ${requirements_map}" >&2
  exit 1
fi
if [[ ! -d "${siinfer_repo_root}" ]]; then
  echo "[ERR] siinfer_repo_root not found: ${siinfer_repo_root}" >&2
  exit 1
fi

mkdir -p "${workspace}" "${output_root}" "${venv_root}"

# ---- ensure uv in PATH ----
ensure_uv() {
  # 1) 用户显式指定 UV_BIN
  if [[ -n "${uv_bin}" ]]; then
    if [[ -x "${uv_bin}" ]]; then
      export PATH="$(cd "$(dirname "${uv_bin}")" && pwd):${PATH}"
      echo "[install] using uv from UV_BIN=${uv_bin}"
      return 0
    else
      echo "[ERR] UV_BIN set but not executable: ${uv_bin}" >&2
      return 1
    fi
  fi

  # 2) 系统已有 uv
  if command -v uv >/dev/null 2>&1; then
    echo "[install] uv found: $(command -v uv)"
    return 0
  fi

  # 3) 常见路径兜底（~/.local/bin）
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "[install] uv found at ~/.local/bin/uv"
    return 0
  fi

  # 4) 自动安装：装到 venv_root/_tools/uv_env 里，避免污染系统
  local tool_env="${venv_root%/}/_tools/uv_env"
  local tool_bin="${tool_env}/bin"
  local py="${tool_bin}/python"
  local uv="${tool_bin}/uv"

  if [[ ! -x "${uv}" ]]; then
    echo "[install] uv not found; bootstrap into ${tool_env}"

    python3 -m venv "${tool_env}"
    "${py}" -m pip install -U pip >/dev/null
    "${py}" -m pip install -U uv
  fi

  export PATH="${tool_bin}:${PATH}"
  if command -v uv >/dev/null 2>&1; then
    echo "[install] uv bootstrapped: $(command -v uv)"
    return 0
  fi

  echo "[ERR] failed to make uv available" >&2
  return 1
}

ensure_uv

echo "[install] SiInfer root       : ${SII_ROOT}"
echo "[install] config_root        : ${config_root}"
echo "[install] qwencoder_root     : ${qwencoder_root}"
echo "[install] workspace          : ${workspace}"
echo "[install] venv_root          : ${venv_root}"
echo "[install] output_root        : ${output_root}"
echo "[install] task_define        : ${task_define}"
echo "[install] requirements_map   : ${requirements_map}"
echo "[install] siinfer_repo_root  : ${siinfer_repo_root}"
echo "[install] uv                : $(command -v uv)"

python "${PREPARE_PY}" \
  --config-root "${config_root}" \
  --qwencoder-root "${qwencoder_root}" \
  --workspace "${workspace}" \
  --venv-root "${venv_root}" \
  --output-root "${output_root}" \
  --task-define "${task_define}" \
  --requirements-map "${requirements_map}" \
  --siinfer-repo-root "${siinfer_repo_root}"