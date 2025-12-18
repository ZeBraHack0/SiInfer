#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ====== 默认值（不传位置参数时使用） ======
DEFAULT_MODEL_NAME="/volume/bhzhao/Qwen2.5-Coder-32B"
DEFAULT_OUTPUT_DIR="/volume/bhzhao/workspace"
DEFAULT_OPENAI_BASE_URL=""   # 允许为空
DEFAULT_OPENAI_API_KEY=""    # 暂时不用，先读出来
# =========================================

# 位置参数读取（按你要求的形式），并兼容 set -u
MODEL_NAME="${1:-$DEFAULT_MODEL_NAME}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
OPENAI_BASE_URL="${3:-$DEFAULT_OPENAI_BASE_URL}"
OPENAI_API_KEY="${4:-$DEFAULT_OPENAI_API_KEY}"

# 兼容你当前脚本变量命名
workspace="${OUTPUT_DIR}"
model_name="${MODEL_NAME}"
URL="${OPENAI_BASE_URL}"

WATCHER_PY="${ROOT_DIR}/adapter_watcher.py"
BASE_CONFIG="${ROOT_DIR}/base_config.json"
CMD_TEMPLATE='python /volume/bhzhao/SiLLM-OP/bin/bench.py --config {config_path} --override'

# 日志放到 workspace 下
LOG_DIR="${workspace}/orchestrator_logs"
mkdir -p "${LOG_DIR}"

# ---- sanity check ----
if [[ ! -d "${workspace}" ]]; then
  echo "[ERR] workspace not found: ${workspace}" >&2
  exit 1
fi
if [[ ! -f "${WATCHER_PY}" ]]; then
  echo "[ERR] adapter_watcher.py not found: ${WATCHER_PY}" >&2
  exit 1
fi
if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "[ERR] base_config.json not found: ${BASE_CONFIG}" >&2
  exit 1
fi

WATCHER_LOG="${LOG_DIR}/watcher.log"

WATCHER_PID=""
cleanup() {
  set +e
  if [[ -n "${WATCHER_PID}" ]] && kill -0 "${WATCHER_PID}" 2>/dev/null; then
    echo "[launch] killing watcher pid=${WATCHER_PID}"
    pkill -P "${WATCHER_PID}" 2>/dev/null || true
    kill "${WATCHER_PID}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${WATCHER_PID}" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "${WATCHER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

normalize_name() {
  # 去掉可能的 CR，避免奇怪的匹配失败
  printf '%s' "$1" | tr -d '\r'
}

# 解析 bench 名：必须形如 (base|chat|tool)__REST
# 输出：prefix<TAB>rest（保证 read 一次就能读到两个字段）
parse_bench_name() {
  local name
  name="$(normalize_name "$1")"

  local prefix="${name%%__*}"
  if [[ "${prefix}" == "${name}" ]]; then
    return 1  # 没有 "__"
  fi

  case "${prefix}" in
    base|chat|tool) ;;
    *) return 1 ;;
  esac

  local rest="${name#*__}"
  if [[ -z "${rest}" ]]; then
    return 1
  fi

  printf '%s\t%s\n' "${prefix}" "${rest}"
}

# rest: "a__b__c" -> "a/b/c/adapter.py"
rest_to_rel_adapter_path() {
  local rest="$1"
  local rel="${rest//__/\/}"
  echo "${rel}/adapter.py"
}

has_adapter_for_bench() {
  local bench_name="$1"
  local parsed prefix rest rel

  if ! parsed="$(parse_bench_name "${bench_name}")"; then
    return 1
  fi

  IFS=$'\t' read -r prefix rest <<<"${parsed}"
  rel="$(rest_to_rel_adapter_path "${rest}")"

  [[ -f "${ROOT_DIR}/${prefix}/${rel}" ]]
}

echo "[launch] start watcher first..."
echo "[launch] workspace=${workspace}"
echo "[launch] model_name=${model_name}"
echo "[launch] URL=${URL}"
echo "[launch] log_dir=${LOG_DIR}"

python3 "${WATCHER_PY}" \
  --workspace "${workspace}" \
  --base-config "${BASE_CONFIG}" \
  --cmd-template "${CMD_TEMPLATE}" \
  --model-name "${model_name}" \
  --engine-url "${URL}" \
  --verbose \
  >"${WATCHER_LOG}" 2>&1 &

WATCHER_PID="$!"
echo "[launch] watcher pid=${WATCHER_PID}, log=${WATCHER_LOG}"

sleep 1

# ---- discover and start all benches ----
mapfile -t RUN_SCRIPTS < <(find "${workspace}" -maxdepth 2 -type f -name "run.sh" | sort)

if [[ "${#RUN_SCRIPTS[@]}" -eq 0 ]]; then
  echo "[launch] no run.sh found under ${workspace}/*/run.sh"
  exit 0
fi

echo "[launch] found ${#RUN_SCRIPTS[@]} run.sh scripts. filter by adapter existence (base/chat/tool strict)..."

PIDS=()
NAMES=()

for run_sh in "${RUN_SCRIPTS[@]}"; do
  bench_dir="$(dirname "${run_sh}")"
  bench_name="$(basename "${bench_dir}")"
  bench_name="$(normalize_name "${bench_name}")"

  if ! has_adapter_for_bench "${bench_name}"; then
    if parsed="$(parse_bench_name "${bench_name}")"; then
      IFS=$'\t' read -r prefix rest <<<"${parsed}"
      rel="$(rest_to_rel_adapter_path "${rest}")"
      echo "[launch] SKIP bench=${bench_name} (need ${ROOT_DIR}/${prefix}/${rel})"
    else
      echo "[launch] SKIP bench=${bench_name} (bad name format; expect base__/chat__/tool__)"
    fi
    continue
  fi

  bench_log="${LOG_DIR}/${bench_name}.run.log"
  echo "[launch] START bench=${bench_name} run_sh=${run_sh} log=${bench_log}"
  ( cd "${bench_dir}" && bash "./run.sh" ) >"${bench_log}" 2>&1 &

  PIDS+=("$!")
  NAMES+=("${bench_name}")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "[launch] no benchmarks to run after filtering -> kill watcher."
  cleanup
  WATCHER_PID=""
  exit 0
fi

# ---- wait all benches ----
rc_all=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if wait "${pid}"; then
    echo "[launch] DONE  bench=${name} rc=0"
  else
    rc=$?
    echo "[launch] DONE  bench=${name} rc=${rc} (log: ${LOG_DIR}/${name}.run.log)"
    rc_all=1
  fi
done

echo "[launch] all benches finished -> kill watcher now."
cleanup
WATCHER_PID=""

exit "${rc_all}"
