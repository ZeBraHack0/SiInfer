#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import re
import shutil
import stat
import subprocess
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ----------------------------
# basic utils
# ----------------------------

def _tail_file(path: Path, n_lines: int = 120) -> str:
    try:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception:
        return ""


def run_cmd(
    cmd,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stdout_path: Optional[Path] = None,
    stderr_path: Optional[Path] = None,
    check: bool = True,
    timeout_sec: int = 0,   # 0 means no timeout
) -> int:
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        if timeout_sec and timeout_sec > 0:
            out, err = p.communicate(timeout=timeout_sec)
        else:
            out, err = p.communicate()
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        if stdout_path:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text(out, encoding="utf-8")
        if stderr_path:
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.write_text(err, encoding="utf-8")
        raise TimeoutError(
            f"Command timeout after {timeout_sec}s: {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout={'(see file)' if stdout_path else out[-4000:]}\n"
            f"stderr={'(see file)' if stderr_path else err[-4000:]}\n"
        )

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(out, encoding="utf-8")
    if stderr_path:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.write_text(err, encoding="utf-8")

    if check and p.returncode != 0:
        out_tail = _tail_file(stdout_path) if stdout_path else out[-4000:]
        err_tail = _tail_file(stderr_path) if stderr_path else err[-4000:]
        raise RuntimeError(
            f"Command failed (code={p.returncode}): {' '.join(cmd)}\n"
            f"cwd={cwd}\n"
            f"stdout={'(see file)' if stdout_path else ''}\n"
            f"stderr={'(see file)' if stderr_path else ''}\n"
            f"---- stdout tail ----\n{out_tail}\n"
            f"---- stderr tail ----\n{err_tail}\n"
        )
    return p.returncode


def ensure_executable(path: Path) -> None:
    if not path.exists():
        return
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_qwencoder_path(p: str) -> str:
    p = (p or "").strip().lstrip("./")
    if p.startswith("qwencoder-eval/"):
        return p[len("qwencoder-eval/"):]
    return p


def parse_yaml_cmd_block(yaml_path: Path) -> Optional[str]:
    """
    只解析 example.yaml 里的 `cmd: |` 多行块，不依赖 PyYAML。
    """
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    cmd_idx = None
    cmd_indent = None
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)cmd\s*:\s*(\|.*)?\s*$", ln)
        if m:
            cmd_idx = i
            cmd_indent = len(m.group(1))
            break
    if cmd_idx is None:
        return None

    buf = []
    for j in range(cmd_idx + 1, len(lines)):
        ln = lines[j]
        if ln.strip() == "" and not buf:
            continue
        cur_indent = len(ln) - len(ln.lstrip(" "))
        if cur_indent <= cmd_indent and ln.strip() != "":
            break
        if len(ln) >= cmd_indent + 2:
            buf.append(ln[cmd_indent + 2 :])
        else:
            buf.append(ln.lstrip("\n"))
    cmd = "\n".join(buf).strip("\n")
    return cmd if cmd.strip() else None


def extract_vars_from_cmd(cmd: str) -> Dict[str, str]:
    vars_: Dict[str, str] = {}
    for ln in cmd.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = re.match(
            r'^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^#\s]+)\s*$',
            ln,
        )
        if not m:
            continue
        k = m.group(1)
        v = m.group(2).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        vars_[k] = v
    return vars_


def find_requirements_file(qwencoder_root: Path, benchmark_root_dir: str) -> Optional[Path]:
    """
    默认探测规则：优先 benchmark_root_dir/requirements.txt
    其次 benchmark_root_dir/requirements/requirements.txt
    """
    if not benchmark_root_dir:
        return None
    cand1 = qwencoder_root / benchmark_root_dir / "requirements.txt"
    if cand1.exists():
        return cand1
    cand2 = qwencoder_root / benchmark_root_dir / "requirements" / "requirements.txt"
    if cand2.exists():
        return cand2
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_venv_python_major_minor(venv_py: Path) -> Optional[str]:
    if not venv_py.exists():
        return None
    try:
        out = subprocess.check_output(
            [str(venv_py), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True,
        ).strip()
        return out
    except Exception:
        return None


# ----------------------------
# parse task_define.py safely
# ----------------------------

def _literal_eval_return_list_from_func(tree: ast.AST, func_name: str) -> Optional[List[dict]]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    try:
                        v = ast.literal_eval(stmt.value)
                        if isinstance(v, list):
                            return v
                    except Exception:
                        return None
    return None


def load_tasks_from_task_define(task_define_path: Path, groups: List[str]) -> List[Dict[str, Any]]:
    """
    返回：[{group, task_name, task_id, optional_params}, ...]
    """
    src = task_define_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=str(task_define_path))

    mapping = {
        "base": "get_base_tasks",
        "chat": "get_chat_tasks",
        "tool": "get_tool_calling_tasks",
        "tool_calling": "get_tool_calling_tasks",
    }

    out: List[Dict[str, Any]] = []
    for g in groups:
        if g not in mapping:
            raise ValueError(f"Unknown group: {g} (supported: base, chat, tool)")
        func = mapping[g]
        items = _literal_eval_return_list_from_func(tree, func)
        if not items:
            continue
        for it in items:
            if not isinstance(it, dict) or "task_name" not in it:
                continue
            task_name = str(it["task_name"])
            optional_params = it.get("optional_params", {}) or {}
            if not isinstance(optional_params, dict):
                optional_params = {}
            task_id = f"{g}__{task_name}"
            out.append(
                {
                    "group": g,
                    "task_name": task_name,
                    "task_id": task_id,
                    "optional_params": {str(k): str(v) for k, v in optional_params.items()},
                }
            )
    return out


# ----------------------------
# task-config.json (mapping only)
# ----------------------------

@dataclass
class TaskConfig:
    yaml_template: Optional[str]
    benchmark_root_dir: str
    benchmark_script_path: str
    default_params: Dict[str, str]


def load_task_config(task_config_path: Path) -> Dict[str, TaskConfig]:
    data = json.loads(task_config_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(data, dict):
        raise ValueError("task-config.json should be a dict: task_name -> config")

    out: Dict[str, TaskConfig] = {}
    for task_name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        # 保留 JSON dict 的插入顺序（Python 3.7+）
        dp = cfg.get("default_params", {}) or {}
        if not isinstance(dp, dict):
            dp = {}

        out[str(task_name)] = TaskConfig(
            yaml_template=cfg.get("yaml_template"),
            benchmark_root_dir=str(cfg.get("benchmark_root_dir", "")),
            benchmark_script_path=str(cfg.get("benchmark_script_path", "")),
            default_params={str(k): str(v) for k, v in dp.items()},
        )
    return out


# ----------------------------
# requirements/python override map
# ----------------------------

ReqItem = Union[str, None, Dict[str, Any]]
ReqMapType = Dict[str, Union[ReqItem, Dict[str, ReqItem]]]


def load_requirements_map(path: Optional[str]) -> Optional[ReqMapType]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"--requirements-map not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("--requirements-map must be a JSON dict")
    return obj


def _parse_req_item(item: ReqItem) -> Tuple[Optional[str], Optional[str]]:
    """
    returns (requirements_path_str_or_None, python_spec_or_None)
    """
    if item is None:
        return None, None
    if isinstance(item, str):
        s = item.strip()
        if s == "":
            return None, None
        return s, None
    if isinstance(item, dict):
        req = item.get("requirements", item.get("req", None))
        py = item.get("python", item.get("py", None))
        if isinstance(req, str):
            req = req.strip() or None
        else:
            req = (str(req).strip() if req is not None else None) or None

        if isinstance(py, str):
            py = py.strip() or None
        else:
            py = (str(py).strip() if py is not None else None) or None

        return req, py
    return None, None


def _resolve_path_maybe_relative(
    chosen: str,
    *,
    qwencoder_root: Path,
    config_root: Path,
) -> Path:
    p = Path(chosen).expanduser()
    if p.is_absolute():
        return p
    p1 = (qwencoder_root / chosen).resolve()
    if p1.exists():
        return p1
    return (config_root / chosen).resolve()


def resolve_overrides(
    req_map: Optional[ReqMapType],
    *,
    task_id: str,
    group: str,
    task_name: str,
    qwencoder_root: Path,
    config_root: Path,
) -> Tuple[Optional[Path], str, Optional[str], str]:
    """
    返回：
      (requirements_path_or_None, requirements_source, python_spec_or_None, python_source)

    key 优先级：
      1) task_id
      2) group/task_name
      3) task_name
      4) group dict[group][task_name]
    """
    if not req_map:
        return None, "", None, ""

    keys = [task_id, f"{group}/{task_name}", task_name]

    for k in keys:
        v = req_map.get(k)
        if v is None:
            continue
        req_s, py_s = _parse_req_item(v)
        req_p = _resolve_path_maybe_relative(req_s, qwencoder_root=qwencoder_root, config_root=config_root) if req_s else None
        return req_p, f"override({k})", py_s, f"override({k})"

    gv = req_map.get(group)
    if isinstance(gv, dict) and task_name in gv:
        v = gv.get(task_name)
        req_s, py_s = _parse_req_item(v)
        req_p = _resolve_path_maybe_relative(req_s, qwencoder_root=qwencoder_root, config_root=config_root) if req_s else None
        return req_p, f"override({group}.{task_name})", py_s, f"override({group}.{task_name})"

    return None, "", None, ""


# ----------------------------
# imageVersion helper (match TaskConfigManager)
# ----------------------------

def parse_image_version_from_yaml(yaml_path: Optional[Path]) -> Optional[str]:
    """
    尽量复刻 TaskConfigManager:
      imageVersion = imageUrl.split(":")[-1].split('-')[0]
    """
    if not yaml_path or (not yaml_path.exists()):
        return None
    try:
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    image_url = None
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("imageUrl:"):
            image_url = s.split(":", 1)[1].strip().strip('"').strip("'")
            break

    if image_url:
        try:
            ver = image_url.split(":")[-1].split("-")[0]
            return ver.strip() or None
        except Exception:
            pass

    # fallback: imageVersion:
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("imageVersion:"):
            v = s.split(":", 1)[1].strip().strip('"').strip("'")
            return v.strip() or None
    return None


def version_ge(v: str, base: str = "1.1") -> bool:
    """
    轻量版本比较：提取数字段，按 (major, minor, patch) 比较。
    """
    def to_tuple(x: str) -> Tuple[int, int, int]:
        parts = re.split(r"[^\d]+", x)
        nums = [int(p) for p in parts if p != ""]
        nums = (nums + [0, 0, 0])[:3]
        return (nums[0], nums[1], nums[2])

    try:
        return to_tuple(v) >= to_tuple(base)
    except Exception:
        return False


# ----------------------------
# run.sh generator (match TaskConfigManager.generate_cmd semantics)
# ----------------------------

def generate_run_sh(
    task_id: str,
    task_name: str,
    task_optional_params: Dict[str, str],
    cfg: TaskConfig,
    config_root: Path,
    qwencoder_root: Path,
    venv_dir: Path,
    unified_output_root: Path,
    default_model_name: str,
    default_openai_base_url: str,
    default_openai_api_key: str,
) -> Tuple[str, Dict[str, str]]:

    # 读取 yaml_template：仅用于补全 meta_vars / imageVersion（不依赖 cmd 内容生成）
    cmd_text = None
    meta_vars: Dict[str, str] = {}

    yaml_path = None
    image_ver = None
    if cfg.yaml_template:
        yaml_path = (config_root / cfg.yaml_template).resolve()
        if yaml_path.exists():
            cmd_text = parse_yaml_cmd_block(yaml_path)
            if cmd_text:
                meta_vars = extract_vars_from_cmd(cmd_text)
            image_ver = parse_image_version_from_yaml(yaml_path)

    bench_root = normalize_qwencoder_path(cfg.benchmark_root_dir or meta_vars.get("BENCHMARK_ROOT_DIR", ""))
    bench_script = cfg.benchmark_script_path or meta_vars.get("BENCHMARK_SCRIPT_PATH", "")

    unified_out_dir = (unified_output_root / task_id).resolve()

    # ---- 0) effective default_params：复刻 TaskConfigManager 的 imageVersion>=1.1 注入 EXTRA_* ----
    default_params_eff: Dict[str, str] = dict(cfg.default_params)  # 保留顺序的 copy
    if image_ver and version_ge(image_ver, "1.1"):
        for k in ["EXTRA_HEADERS", "EXTRA_BODY", "EXTRA_QUERY"]:
            if k not in default_params_eff:
                default_params_eff[k] = "None"

    # ---- 1) resolved values：optional 覆盖 default（修复 base/chat 覆盖反了的问题） ----
    # 只处理 default_params_eff 里存在的 key（与线上一致：optional 也必须在 default_params 里才会生效）
    resolved: Dict[str, str] = {}
    for k, default_v in default_params_eff.items():
        v = task_optional_params.get(k, default_v)
        v = "" if v is None else str(v)
        if v != "":
            resolved[k] = v

    # ---- 2) run.sh 里写变量默认值：允许外部 env 覆盖 ----
    # 模拟线上：EXTRA_* 用 export，其它只是变量（但我们也可不 export）
    extra_keys = {"EXTRA_HEADERS", "EXTRA_BODY", "EXTRA_QUERY"}

    param_lines: List[str] = []
    export_lines: List[str] = []
    for k, _ in default_params_eff.items():
        if k not in resolved:
            continue
        v = resolved[k]
        param_lines.append(f': "${{{k}:={json.dumps(v)}}}"')
        if k in extra_keys:
            export_lines.append(f'export {k}')

    # ---- 3) 构造脚本参数：严格复刻 TaskConfigManager.script_args ----
    # 固定 4 个，然后仅当 optional_params 显式提供了 param 且 param 不是 EXTRA_* 时追加
    script_args: List[str] = [
        '"${MODEL_NAME}"',
        '"${OUTPUT_DIR}"',
        '"${OPENAI_BASE_URL}"',
        '"${OPENAI_API_KEY}"',
    ]

    for k in default_params_eff.keys():
        if k in extra_keys:
            continue
        if task_optional_params.get(k):  # 只有 optional 显式非空提供才追加（与线上一致）
            script_args.append(f'"${{{k}}}"')

    # ---- 4) 统一生成启动命令（保证一定 cd） ----
    # 注意：BENCHMARK_SCRIPT_PATH 在 cd 后作为相对路径运行
    launch_lines = [
        'cd "${REPO_ROOT}/${BENCHMARK_ROOT_DIR}"',
        f'bash "${{BENCHMARK_SCRIPT_PATH}}" {" ".join(script_args)} "$@"',
    ]

    run_sh = f"""#!/usr/bin/env bash
set -euo pipefail

# ===== auto-generated for task_id: {task_id} (task_name: {task_name}) =====
REPO_ROOT={json.dumps(str(qwencoder_root.resolve()))}

# uv venv
VENV_DIR={json.dumps(str(venv_dir.resolve()))}
export PATH="${{VENV_DIR}}/bin:${{PATH}}"

# unified output dir (patched)
OUTPUT_DIR={json.dumps(str(unified_out_dir))}
mkdir -p "${{OUTPUT_DIR}}"
export OUTPUT_DIR

# common runtime vars (overridable)
: "${{MODEL_NAME:={json.dumps(default_model_name)}}}"
: "${{OPENAI_BASE_URL:={json.dumps(default_openai_base_url)}}}"
: "${{OPENAI_API_KEY:={json.dumps(default_openai_api_key)}}}"
export MODEL_NAME OPENAI_BASE_URL OPENAI_API_KEY

# aliases for entry scripts expecting API_BASE/API_KEY
: "${{API_BASE:=${{OPENAI_BASE_URL}}}}"
: "${{API_KEY:=${{OPENAI_API_KEY}}}}"
export API_BASE API_KEY

# benchmark entry
BENCHMARK_ROOT_DIR={json.dumps(bench_root)}
BENCHMARK_SCRIPT_PATH={json.dumps(bench_script)}
export BENCHMARK_ROOT_DIR BENCHMARK_SCRIPT_PATH

# ---- default_params (TaskConfigManager semantics: optional overrides default; env-overridable here) ----
{os.linesep.join(param_lines)}
{os.linesep.join(export_lines)}

# imageVersion (from yaml_template): {json.dumps(image_ver or "", ensure_ascii=False)}
# default_params keys (effective): {json.dumps(list(default_params_eff.keys()), ensure_ascii=False)}
# task_define optional_params: {json.dumps(task_optional_params, ensure_ascii=False)}
# resolved values used in this run.sh: {json.dumps(resolved, ensure_ascii=False)}
# script args: {json.dumps(script_args, ensure_ascii=False)}

# ===== launch =====
{os.linesep.join(launch_lines)}
"""

    entry_abs = ""
    if bench_root and bench_script:
        p = (qwencoder_root / bench_root / bench_script).resolve()
        if p.exists():
            entry_abs = str(p)

    meta = {
        "task_id": task_id,
        "task_name": task_name,
        "yaml_template": str(yaml_path) if yaml_path else "",
        "image_version": image_ver or "",
        "BENCHMARK_ROOT_DIR": bench_root,
        "BENCHMARK_SCRIPT_PATH": bench_script,
        "OUTPUT_DIR": str(unified_out_dir),
        "venv_dir": str(venv_dir),
        "entry_abs": entry_abs,
        "default_params_effective_keys": json.dumps(list(default_params_eff.keys()), ensure_ascii=False),
        "resolved_params": json.dumps(resolved, ensure_ascii=False),
        "script_args": json.dumps(script_args, ensure_ascii=False),
    }
    return run_sh, meta


def ensure_pip_in_venv(
    venv_py: Path,
    cwd: Optional[Path],
    stdout_path: Path,
    stderr_path: Path,
    uv_timeout_sec: int = 0,
) -> None:
    """
    Ensure `pip` module exists in this venv.
    Strategy:
      1) python -m pip --version
      2) python -m ensurepip --upgrade
      3) python -m pip --version
    """
    if not venv_py.exists():
        raise FileNotFoundError(f"venv python not found: {venv_py}")

    try:
        run_cmd([str(venv_py), "-m", "pip", "--version"], cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path, check=True)
        return
    except Exception:
        pass

    try:
        print("[INFO] pip missing -> try: python -m ensurepip --upgrade")
        run_cmd(
            [str(venv_py), "-m", "ensurepip", "--upgrade"],
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            check=True,
        )
    except Exception as e:
        print(f"[WARN] ensurepip failed/unavailable: {type(e).__name__}")

    run_cmd([str(venv_py), "-m", "pip", "--version"], cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path, check=True)


# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser("Bootstrap & (optionally) run all benchmarks based on task_define.py")
    ap.add_argument("--task-define", required=True, help="Path to task_define.py")
    ap.add_argument("--groups", default="base,chat,tool", help="Comma-separated: base,chat,tool")
    ap.add_argument("--config-root", required=True, help="Directory containing task-config.json and yaml templates")
    ap.add_argument("--task-config", default="config/task-config.json", help="Path (relative to config-root) to task-config.json")
    ap.add_argument("--qwencoder-root", required=True, help="Path to cloned qwencoder-eval repo root")
    ap.add_argument("--workspace", required=True, help="Where to put per-task folders (run.sh, logs, meta)")
    ap.add_argument("--venv-root", required=True, help="Where to create per-task uv venvs")
    ap.add_argument("--output-root", required=True, help="Unified output root dir; each task uses output-root/<task_id>")
    ap.add_argument("--python", default="", help="Default python for uv venv (can be overridden per-task via requirements-map), e.g. 3.10/python3.10")
    ap.add_argument("--model-name", default="dummy-model", help="Default MODEL_NAME used in run.sh (can override at runtime)")
    ap.add_argument("--openai-base-url", default="http://127.0.0.1:8000/v1", help="Default OPENAI_BASE_URL used in run.sh")
    ap.add_argument("--openai-api-key", default="EMPTY", help="Default OPENAI_API_KEY used in run.sh")
    ap.add_argument("--requirements-map", default="", help="Optional JSON mapping to specify requirements.txt + python per task")
    ap.add_argument("--force-reinstall", action="store_true", help="Destroy old venv and reinstall everything")
    ap.add_argument("--run", action="store_true", help="Actually run each generated run.sh")
    ap.add_argument("--keep-going", action="store_true", help="Continue even if one task fails (only with --run)")
    ap.add_argument("--dry-run", action="store_true", help="Only generate scripts/meta; do not create venv/install/run")
    ap.add_argument(
        "--install-method",
        choices=["auto", "uv", "pip"],
        default="auto",
        help="Dependency install backend: auto(uv->pip fallback), uv(only uv pip), pip(only python -m pip).",
    )
    ap.add_argument(
        "--uv-pip-timeout",
        type=int,
        default=0,
        help="Timeout seconds for uv pip install in auto/uv mode; 0 means no timeout.",
    )
    args = ap.parse_args()

    if shutil.which("uv") is None:
        raise RuntimeError("uv not found in PATH")

    task_define_path = Path(args.task_define).resolve()
    config_root = Path(args.config_root).resolve()
    task_config_path = (config_root / args.task_config).resolve()
    qwencoder_root = Path(args.qwencoder_root).resolve()
    workspace = Path(args.workspace).resolve()
    venv_root = Path(args.venv_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not task_define_path.exists():
        raise FileNotFoundError(f"task_define.py not found: {task_define_path}")
    if not task_config_path.exists():
        raise FileNotFoundError(f"task-config.json not found: {task_config_path}")
    if not qwencoder_root.exists():
        raise FileNotFoundError(f"qwencoder-root not found: {qwencoder_root}")

    req_map = load_requirements_map(args.requirements_map) if args.requirements_map else None

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    tasks = load_tasks_from_task_define(task_define_path, groups)
    task_cfg_map = load_task_config(task_config_path)

    workspace.mkdir(parents=True, exist_ok=True)
    venv_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    total = len(tasks)

    for idx, t in enumerate(tasks, start=1):
        task_id = t["task_id"]
        task_name = t["task_name"]
        group = t["group"]
        optional_params = t["optional_params"]

        print(f"\n[{idx}/{total}] >>> group={group} task_id={task_id} task_name={task_name}")

        task_dir = workspace / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        setup_out = task_dir / "setup.stdout.txt"
        setup_err = task_dir / "setup.stderr.txt"
        run_out = task_dir / "run.stdout.txt"
        run_err = task_dir / "run.stderr.txt"
        req_fingerprint_path = task_dir / "requirements.fingerprint.json"
        python_spec_path = task_dir / "venv.python_spec.txt"

        if task_name not in task_cfg_map:
            msg = f"task-config.json missing entry for {task_name}"
            print(f"[WARN] {msg} -> skip")
            results[task_id] = {"status": "skipped", "reason": msg}
            continue

        cfg = task_cfg_map[task_name]
        bench_root = normalize_qwencoder_path(cfg.benchmark_root_dir)

        # venv paths
        venv_dir = venv_root / task_id
        venv_py = venv_dir / "bin" / "python"

        # overrides: requirements + python
        req_override, req_source, py_override, py_source = resolve_overrides(
            req_map,
            task_id=task_id,
            group=group,
            task_name=task_name,
            qwencoder_root=qwencoder_root,
            config_root=config_root,
        )

        # desired python: per-task override > global --python > None
        desired_python = (py_override or "").strip() or (args.python or "").strip() or None
        if desired_python:
            print(f"[INFO] python({py_source or 'default(--python)'}) = {desired_python}")
        else:
            print(f"[INFO] python = <auto by uv> (no --python and no per-task override)")

        # 1) venv handling
        if not args.dry_run:
            if args.force_reinstall and venv_dir.exists():
                print(f"[INFO] --force-reinstall: destroy venv: {venv_dir}")
                shutil.rmtree(venv_dir, ignore_errors=True)
                if req_fingerprint_path.exists():
                    req_fingerprint_path.unlink(missing_ok=True)
                if python_spec_path.exists():
                    python_spec_path.unlink(missing_ok=True)

            if venv_dir.exists() and not venv_py.exists():
                print(f"[WARN] venv dir exists but python missing -> recreate: {venv_dir}")
                shutil.rmtree(venv_dir, ignore_errors=True)

            if venv_dir.exists() and desired_python and python_spec_path.exists():
                old_spec = python_spec_path.read_text(encoding="utf-8", errors="ignore").strip()
                if old_spec and old_spec != desired_python:
                    print(f"[WARN] python spec changed ({old_spec} -> {desired_python}) -> recreate venv")
                    shutil.rmtree(venv_dir, ignore_errors=True)

            if venv_dir.exists() and desired_python and re.match(r"^\d+\.\d+(\.\d+)?$", desired_python):
                actual_mm = read_venv_python_major_minor(venv_py)
                want_mm = ".".join(desired_python.split(".")[:2])
                if actual_mm and actual_mm != want_mm:
                    print(f"[WARN] venv python mismatch (actual={actual_mm}, want={want_mm}) -> recreate venv")
                    shutil.rmtree(venv_dir, ignore_errors=True)

            if not venv_dir.exists():
                cmd = ["uv", "venv", str(venv_dir)]
                if desired_python:
                    cmd += ["--python", desired_python]
                print(f"[INFO] create venv: {venv_dir}")
                run_cmd(cmd, stdout_path=setup_out, stderr_path=setup_err, check=True)
                if desired_python:
                    write_text(python_spec_path, desired_python)
            else:
                print(f"[INFO] venv exists: {venv_dir} (skip create)")

        # 2) requirements path resolve (override -> auto-detect)
        if req_override is not None:
            req = req_override
            req_source_final = req_source or "override(--requirements-map)"
        else:
            req = find_requirements_file(qwencoder_root, bench_root) if bench_root else None
            req_source_final = "auto-detect"

        if req is None:
            print(f"[WARN] requirements not found for task={task_id} (bench_root={bench_root})")
        else:
            print(f"[INFO] requirements({req_source_final}) = {req}")
            if not req.exists():
                print(f"[WARN] requirements path does not exist: {req}")

        # 2.5) install requirements (idempotent + fingerprint)
        installed = False
        skipped_install = False

        if (not args.dry_run) and (req is not None) and req.exists():
            fp = {
                "requirements_path": str(req.resolve()),
                "sha256": sha256_file(req),
                "venv_python": str(venv_py),
                "python_spec": desired_python or "",
                "install_method": args.install_method,
            }

            if (not args.force_reinstall) and req_fingerprint_path.exists():
                try:
                    old = json.loads(req_fingerprint_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(old, dict)
                        and old.get("requirements_path") == fp["requirements_path"]
                        and old.get("sha256") == fp["sha256"]
                        and (old.get("python_spec", "") == fp["python_spec"])
                        and (old.get("install_method", "auto") == fp["install_method"])
                    ):
                        print("[INFO] requirements unchanged -> skip install")
                        skipped_install = True
                except Exception:
                    pass

            if not skipped_install:
                cwd_install = (qwencoder_root / bench_root) if bench_root else None

                def do_pip():
                    print("[INFO] ensure pip exists in venv ...")
                    ensure_pip_in_venv(
                        Path(venv_py),
                        cwd_install,
                        setup_out,
                        setup_err,
                        uv_timeout_sec=int(args.uv_pip_timeout or 0),
                    )
                    print("[INFO] install deps via: python -m pip install -r ...")
                    run_cmd(
                        [str(venv_py), "-m", "pip", "install", "-r", str(req), "-i", "https://mirrors.aliyun.com/pypi/simple"],
                        cwd=cwd_install,
                        stdout_path=setup_out,
                        stderr_path=setup_err,
                        check=True,
                    )

                def do_uv():
                    print("[INFO] install deps via: uv pip install -r ...")
                    run_cmd(
                        ["uv", "pip", "install", "--python", str(venv_py), "-r", str(req), "-i", "https://mirrors.aliyun.com/pypi/simple"],
                        cwd=cwd_install,
                        stdout_path=setup_out,
                        stderr_path=setup_err,
                        check=True,
                        timeout_sec=int(args.uv_pip_timeout or 0),
                    )

                try:
                    if args.install_method == "pip":
                        do_pip()
                    elif args.install_method == "uv":
                        do_uv()
                    else:
                        try:
                            do_uv()
                        except (TimeoutError, RuntimeError) as e1:
                            print(f"[WARN] uv pip slow/failed -> fallback to pip. reason={type(e1).__name__}")
                            do_pip()

                    installed = True
                finally:
                    if installed:
                        write_text(req_fingerprint_path, json.dumps(fp, indent=2, ensure_ascii=False))

        # 3) run.sh + meta
        run_sh_text, meta = generate_run_sh(
            task_id=task_id,
            task_name=task_name,
            task_optional_params=optional_params,
            cfg=cfg,
            config_root=config_root,
            qwencoder_root=qwencoder_root,
            venv_dir=venv_dir,
            unified_output_root=output_root,
            default_model_name=args.model_name,
            default_openai_base_url=args.openai_base_url,
            default_openai_api_key=args.openai_api_key,
        )
        run_sh_path = task_dir / "run.sh"
        write_text(run_sh_path, run_sh_text)
        ensure_executable(run_sh_path)

        meta_obj = dict(meta)
        meta_obj["requirements_txt"] = str(req) if req else ""
        meta_obj["requirements_source"] = req_source_final if req else ""
        meta_obj["requirements_installed"] = bool(installed)
        meta_obj["requirements_skipped_install"] = bool(skipped_install)
        meta_obj["group"] = group
        meta_obj["optional_params"] = optional_params
        meta_obj["python_spec"] = desired_python or ""
        meta_obj["python_actual_major_minor"] = read_venv_python_major_minor(venv_py) or ""
        write_text(task_dir / "meta.json", json.dumps(meta_obj, indent=2, ensure_ascii=False))

        # 4) run (optional)
        status = "generated"
        exit_code = None
        if args.run and not args.dry_run:
            print(f"[INFO] running: {run_sh_path}")
            try:
                run_cmd(
                    ["bash", str(run_sh_path)],
                    cwd=task_dir,
                    stdout_path=run_out,
                    stderr_path=run_err,
                    check=True,
                )
                status = "ok"
                exit_code = 0
            except Exception as e:
                status = "failed"
                exit_code = 1
                results[task_id] = {"status": status, "exit_code": exit_code, "error": str(e)}
                print(f"[ERROR] task failed: {task_id}")
                if not args.keep_going:
                    break

        results[task_id] = {
            "status": status,
            "exit_code": exit_code,
            "requirements": str(req) if req else "",
            "requirements_source": req_source_final if req else "",
            "installed": bool(installed),
            "skipped_install": bool(skipped_install),
            "python_spec": desired_python or "",
            "python_actual_major_minor": read_venv_python_major_minor(venv_py) or "",
        }

    write_text(workspace / "summary.json", json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[DONE] summary: {workspace / 'summary.json'}")


if __name__ == "__main__":
    main()
