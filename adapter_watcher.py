#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUEST_READY = "REQUEST_READY"
RESPONSE_READY = "RESPONSE_READY"

CLIENT_LOCK = ".client.lock"
CLIENT_PID = ".client.pid"
CLIENT_CMD = ".client.cmd"
CLIENT_SENT = ".client.sent"
CLIENT_DONE = ".client.done"
CLIENT_RC = ".client.rc"
CLIENT_CONFIG = "client_config.json"

ENGINE_URL_FLAG = "--base-url"

META_FORWARD_KEYS = ["max_tokens", "temperature", "top_p"]

META_KEY_TO_FLAG = {
    "max_tokens": "--custom-output-len",
    "temperature": "--temperature",
    "top_p": "--top-p",
    # stop -> --extra-body dict
}


def sh_quote(s: str) -> str:
    return shlex.quote(s)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def atomic_create_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=4), encoding="utf-8")
    os.replace(tmp, path)


def safe_unlink(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def read_first_jsonl(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            return json.loads(ln)
    raise ValueError(f"empty jsonl: {path}")


def count_jsonl_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                n += 1
    return n


def parse_meta_forward_params(meta_path: Path) -> Dict[str, Any]:
    obj = read_first_jsonl(meta_path)
    meta = obj.get("meta", {})
    out: Dict[str, Any] = {}

    for k in META_FORWARD_KEYS:
        if k not in meta:
            continue

        if k == "stop":
            stop_val = meta.get("stop")
            if stop_val is None:
                continue
            if not isinstance(stop_val, list):
                raise TypeError(f"meta.stop must be a list, got {type(stop_val)}")
            out["--extra-body"] = {"stop": stop_val}
            continue

        flag = META_KEY_TO_FLAG.get(k, f"--{k.replace('_', '-')}")
        out[flag] = meta[k]

    return out


def ensure_client_extra_entry(cfg: Dict[str, Any], backend_name: str) -> Dict[str, Any]:
    ce = cfg.get("client_extra")
    if ce is None:
        cfg["client_extra"] = []
        ce = cfg["client_extra"]
    if not isinstance(ce, list):
        raise TypeError(f'client_extra must be a list, got {type(ce)}')

    for item in ce:
        if isinstance(item, dict) and item.get("backend") == backend_name:
            return item

    item = {"backend": backend_name}
    ce.append(item)
    return item


@dataclass
class RunDir:
    run_dir: Path
    requests_path: Path
    meta_path: Path
    responses_path: Path

    @property
    def lock_path(self) -> Path:
        return self.run_dir / CLIENT_LOCK

    @property
    def pid_path(self) -> Path:
        return self.run_dir / CLIENT_PID

    @property
    def cmd_path(self) -> Path:
        return self.run_dir / CLIENT_CMD

    @property
    def sent_path(self) -> Path:
        return self.run_dir / CLIENT_SENT

    @property
    def done_path(self) -> Path:
        return self.run_dir / CLIENT_DONE

    @property
    def rc_path(self) -> Path:
        return self.run_dir / CLIENT_RC

    @property
    def config_path(self) -> Path:
        return self.run_dir / CLIENT_CONFIG

    @property
    def stdout_log(self) -> Path:
        return self.run_dir / "client.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.run_dir / "client.stderr.log"

    @property
    def request_ready(self) -> Path:
        return self.run_dir / REQUEST_READY

    @property
    def response_ready(self) -> Path:
        return self.run_dir / RESPONSE_READY


def discover_run_dirs(workspace: Path) -> List[RunDir]:
    out: List[RunDir] = []
    if not workspace.exists():
        return out

    for run_dir in workspace.iterdir():  # ${RUN_DIR}
        if not run_dir.is_dir():
            continue

        adapter_root = run_dir / "adapter_runs"
        if not adapter_root.is_dir():
            continue

        for run_id_dir in adapter_root.iterdir():  # ${SII_RUN_ID}
            if not run_id_dir.is_dir():
                continue

            for bench_dir in run_id_dir.iterdir():  # ${SII_BENCH_NAME}
                if not bench_dir.is_dir():
                    continue

                if not (bench_dir / REQUEST_READY).exists():
                    continue

                out.append(
                    RunDir(
                        run_dir=bench_dir,
                        requests_path=bench_dir / "requests.jsonl",
                        meta_path=bench_dir / "meta.jsonl",
                        responses_path=bench_dir / "responses.jsonl",
                    )
                )

    return out


def build_client_config(
    base_cfg: Dict[str, Any],
    r: RunDir,
    model_name: str,
    engine_url: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # 覆盖 model
    if isinstance(cfg.get("model"), list):
        cfg["model"] = [model_name]
    else:
        cfg["model"] = model_name

    # 写回目录
    cfg["output"] = str(r.run_dir)
    cfg["datapath"] = str(r.run_dir)

    # num_prompt
    try:
        nreq = count_jsonl_lines(r.requests_path) if r.requests_path.exists() else None
        if nreq is not None:
            if isinstance(cfg.get("num_prompt"), list):
                cfg["num_prompt"] = [nreq]
            else:
                cfg["num_prompt"] = nreq
    except Exception:
        pass

    backend_name = "vllm"
    b = cfg.get("backend")
    if isinstance(b, list) and b:
        backend_name = str(b[0])

    ce_item = ensure_client_extra_entry(cfg, backend_name)

    # 仅当 engine_url 非空时写入
    if isinstance(engine_url, str) and engine_url.strip():
        ce_item[ENGINE_URL_FLAG] = engine_url.strip()

    # meta 参数
    meta_flags = parse_meta_forward_params(r.meta_path)
    for flag, val in meta_flags.items():
        ce_item[flag] = val

    return cfg


def build_cmd(cmd_template: str, r: RunDir) -> str:
    mapping: Dict[str, str] = {
        "run_dir": sh_quote(str(r.run_dir)),
        "requests_path": sh_quote(str(r.requests_path)),
        "meta_path": sh_quote(str(r.meta_path)),
        "responses_path": sh_quote(str(r.responses_path)),
        "config_path": sh_quote(str(r.config_path)),
    }
    return cmd_template.format(**mapping).strip()


def launch_background(r: RunDir, cmd: str) -> int:
    r.run_dir.mkdir(parents=True, exist_ok=True)

    wrapper = f"""
set -euo pipefail
cd {sh_quote(str(r.run_dir))}

set +e
{cmd}
rc=$?
set -e

echo "$rc" > {sh_quote(str(r.rc_path))}

touch {sh_quote(RESPONSE_READY)}
touch {sh_quote(str(r.done_path))}

exit "$rc"
""".strip()

    with open(r.stdout_log, "ab") as out_f, open(r.stderr_log, "ab") as err_f:
        p = subprocess.Popen(
            ["bash", "-lc", wrapper],
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
        )
    return p.pid


def load_pid(r: RunDir) -> Optional[int]:
    try:
        return int(r.pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def count_active(runs: List[RunDir]) -> int:
    n = 0
    for r in runs:
        pid = load_pid(r)
        if pid is not None and pid_alive(pid) and not r.done_path.exists():
            n += 1
    return n


def maybe_recover_prelaunch_stale_lock(r: RunDir, verbose: bool = False) -> None:
    if not r.lock_path.exists():
        return
    if r.sent_path.exists():
        return
    pid = load_pid(r)
    if pid is None or (not pid_alive(pid)):
        if verbose:
            print(f"[watcher] recover stale lock in {r.run_dir}")
        safe_unlink(r.lock_path)
        safe_unlink(r.pid_path)
        safe_unlink(r.cmd_path)


def should_start(r: RunDir) -> Tuple[bool, str]:
    if not r.request_ready.exists():
        return False, "no REQUEST_READY"
    if r.sent_path.exists():
        return False, "already sent (.client.sent exists)"
    if not r.requests_path.exists():
        return False, "requests.jsonl missing"
    if not r.meta_path.exists():
        return False, "meta.jsonl missing"
    if r.response_ready.exists():
        return False, "already RESPONSE_READY"
    if r.done_path.exists():
        return False, "already done"
    if r.lock_path.exists():
        pid = load_pid(r)
        if pid is not None and pid_alive(pid):
            return False, f"locked and running pid={pid}"
        return False, "locked (stale or unknown); delete lock manually if needed"
    return True, "ok"


def load_base_config(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"base config must be a dict JSON, got {type(obj)}")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="workspace root to scan")
    ap.add_argument("--base-config", required=True, help="path to base client config json")
    ap.add_argument("--model-name", required=True, help="model name/path to write into client_config.json")

    # URL 允许为空：默认空串；传空则不写入 client_extra
    ap.add_argument("--engine-url", default="", help="engine url to write into client_extra (optional)")

    ap.add_argument("--cmd-template", required=True,
                    help="client launch command template. "
                         "You can use {config_path} {run_dir} {requests_path} {meta_path} {responses_path}")
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--max-parallel", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    base_cfg = load_base_config(Path(args.base_config).resolve())

    print(f"[watcher] workspace={workspace} poll={args.poll_interval}s max_parallel={args.max_parallel}")

    while True:
        runs = discover_run_dirs(workspace)
        runs.sort(key=lambda x: str(x.run_dir))

        if args.verbose:
            print("[watcher] discover runs:", len(runs))

        for r in runs:
            maybe_recover_prelaunch_stale_lock(r, verbose=args.verbose)

        active = count_active(runs)

        for r in runs:
            if active >= args.max_parallel:
                break

            ok, reason = should_start(r)
            if not ok:
                if args.verbose:
                    print(f"[watcher] skip {r.run_dir} ({reason})")
                continue

            if not atomic_create_lock(r.lock_path):
                continue

            try:
                cfg = build_client_config(
                    base_cfg=base_cfg,
                    r=r,
                    model_name=args.model_name,
                    engine_url=args.engine_url,
                )
                write_json(r.config_path, cfg)
            except Exception as e:
                safe_unlink(r.lock_path)
                if args.verbose:
                    print(f"[watcher] skip {r.run_dir} (build config failed: {e})")
                continue

            try:
                cmd = build_cmd(args.cmd_template, r)
                write_text(r.cmd_path, cmd + "\n")
            except Exception as e:
                safe_unlink(r.lock_path)
                if args.verbose:
                    print(f"[watcher] skip {r.run_dir} (build cmd failed: {e})")
                continue

            if args.dry_run:
                print(f"[watcher] DRYRUN config={r.config_path} cmd={cmd}")
                safe_unlink(r.lock_path)
                continue

            pid = launch_background(r, cmd)
            write_text(r.pid_path, str(pid) + "\n")

            sent_payload = f"time={int(time.time())}\npid={pid}\nconfig={r.config_path}\ncmd={cmd}\n"
            write_text(r.sent_path, sent_payload)

            print(f"[watcher] START {r.run_dir} pid={pid}")
            active += 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
