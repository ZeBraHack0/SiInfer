#!/usr/bin/env bash
set -euo pipefail

PY_BIN="${1:-python3}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if ! command -v "${PY_BIN}" >/dev/null 2>&1 && [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERR] PY_BIN not found or not executable: ${PY_BIN}"
  exit 2
fi


if [[ ! -f "${ROOT_DIR}/adapter.py" ]]; then
  echo "[ERR] adapter.py not found under ${ROOT_DIR}"
  exit 1
fi

# 1) create a lightweight package wrapper
PKG_DIR="${ROOT_DIR}/siinfer_adapter"
mkdir -p "${PKG_DIR}"

# symlink preferred (editable dev), fallback to copy
if [[ ! -e "${PKG_DIR}/adapter.py" ]]; then
  if ln -s "../adapter.py" "${PKG_DIR}/adapter.py" 2>/dev/null; then
    :
  else
    cp -f "${ROOT_DIR}/adapter.py" "${PKG_DIR}/adapter.py"
  fi
fi

# 2) expose base + loader via package import
cat > "${PKG_DIR}/__init__.py" <<'PY'
from .adapter import (
    BenchmarkAdapter, REQUEST_READY, RESPONSE_READY,
    find_adapter_file, load_adapter_class_from_file, get_benchmark_adapter,
)

__all__ = [
    "BenchmarkAdapter", "REQUEST_READY", "RESPONSE_READY",
    "find_adapter_file", "load_adapter_class_from_file", "get_benchmark_adapter",
]
PY

# 3) minimal pyproject.toml for editable install
if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
cat > "${ROOT_DIR}/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "siinfer-adapter"
version = "0.1.0"
description = "SiInfer benchmark adapter library"
requires-python = ">=3.8"
dependencies = []

[tool.setuptools]
packages = ["siinfer_adapter"]
TOML
fi

# 4) install (editable)
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${PY_BIN}" -e .
else
  "${PY_BIN}" -m pip install -e .
fi

echo "[OK] Installed. Quick test:"
"${PY_BIN}" -c "import siinfer_adapter; print(siinfer_adapter.get_benchmark_adapter)"

