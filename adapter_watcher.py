#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUEST_READY = "REQUEST_READY"
RESPONSE_READY = "RESPONSE_READY"
RESPONSE_FAILED = "RESPONSE_FAILED"

CLIENT_LOCK = ".client.lock"          # O_EXCL 原子锁，防重复启动
CLIENT_PID = ".client.pid"
CLIENT_CMD = ".client.cmd"
CLIENT_DONE = ".client.done"
CLIENT_FAIL = ".client.fail"


def sh_quote(s: str) -> str:
    return shlex.quote(s)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def atomic_create_lock(lock_path: Path) -> bool:
    """
    原子创建 lock 文件：存在则返回 False；成功创建返回 True
    """
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


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        os.utime(path, None)


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
    def done_path(self) -> Path:
        return self.run_dir / CLIENT_DONE

    @property
    def fail_path(self) -> Path:
        return self.run_dir / CLIENT_FAIL

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

    @property
    def response_failed(self) -> Path:
        return self.run_dir / RESPONSE_FAILED


def discover_run_dirs(root: Path) -> List[RunDir]:
    """
    递归扫描所有 REQUEST_READY 文件，定位 run_dir
    """
    out: List[RunDir] = []
    for ready_file in root.rglob(REQUEST_READY):
        run_dir = ready_file.parent
        req = run_dir / "requests.jsonl"
        meta = run_dir / "meta.jsonl"
        resp = run_dir / "responses.jsonl"
        out.append(RunDir(run_dir=run_dir, requests_path=req, meta_path=meta, responses_path=resp))
    return out


def build_cmd(cmd_template: str, r: RunDir) -> str:
    """
    用占位符填充命令模板，并做 shell quote，避免路径里有空格导致炸掉
    可用占位符：
      {run_dir} {requests_path} {meta_path} {responses_path}
    """
    mapping: Dict[str, str] = {
        "run_dir": sh_quote(str(r.run_dir)),
        "requests_path": sh_quote(str(r.requests_path)),
        "meta_path": sh_quote(str(r.meta_path)),
        "responses_path": sh_quote(str(r.responses_path)),
    }
    # 注意：模板里不要再自己加引号包住占位符（除非你知道你在做什么）
    return cmd_template.format(**mapping)


def launch_background(r: RunDir, cmd: str) -> int:
    """
    启动后台进程：
      - stdout/stderr 重定向到 run_dir
      - 进程结束后：成功则 touch RESPONSE_READY，失败 touch RESPONSE_FAILED
    """
    r.run_dir.mkdir(parents=True, exist_ok=True)

    wrapper = f"""
set -euo pipefail
cd {sh_quote(str(r.run_dir))}
# 运行用户命令
{cmd}
rc=$?
# 约定：命令应当生成 responses.jsonl（路径可通过 {{"responses_path"}} 传给它）
if [ $rc -eq 0 ] && [ -f {sh_quote(str(r.responses_path))} ]; then
  touch {sh_quote(RESPONSE_READY)}
  echo "OK: touched {RESPONSE_READY}"
else
  touch {sh_quote(RESPONSE_FAILED)}
  echo "FAIL: touched {RESPONSE_FAILED}, rc=$rc"
fi
exit $rc
""".strip()

    # 后台运行（新 session），避免 watcher 退出影响子进程
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
        s = r.pid_path.read_text(encoding="utf-8").strip()
        return int(s)
    except Exception:
        return None


def count_active(runs: List[RunDir]) -> int:
    n = 0
    for r in runs:
        pid = load_pid(r)
        if pid is not None and pid_alive(pid) and not r.done_path.exists() and not r.fail_path.exists():
            n += 1
    return n


def should_start(r: RunDir) -> Tuple[bool, str]:
    """
    判断一个 run_dir 是否应该启动：
      - 有 REQUEST_READY
      - requests.jsonl 存在
      - 还没有 RESPONSE_READY/FAILED
      - 还没有 DONE/FAIL 标记
      - lock 不存在（或 lock 存在但 pid 已死，你可以选择回收；这里保守：不回收）
    """
    if not r.request_ready.exists():
        return False, "no REQUEST_READY"
    if not r.requests_path.exists():
        return False, "requests.jsonl missing"
    if r.response_ready.exists():
        return False, "already RESPONSE_READY"
    if r.response_failed.exists():
        return False, "already RESPONSE_FAILED"
    if r.done_path.exists() or r.fail_path.exists():
        return False, "already marked done/fail"
    if r.lock_path.exists():
        pid = load_pid(r)
        if pid is not None and pid_alive(pid):
            return False, f"locked and running pid={pid}"
        return False, "locked (stale or unknown); manual cleanup required"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="root directory to scan, e.g. runs/")
    ap.add_argument("--cmd-template", required=True,
                    help="command template to run when REQUEST_READY is found. "
                         "You can use {run_dir} {requests_path} {meta_path} {responses_path}")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="seconds")
    ap.add_argument("--max-parallel", type=int, default=1, help="max concurrent benchserve processes")
    ap.add_argument("--dry-run", action="store_true", help="print what would run but do not start")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    print(f"[watcher] root={root} poll_interval={args.poll_interval}s max_parallel={args.max_parallel}")

    while True:
        runs = discover_run_dirs(root)
        active = count_active(runs)

        for r in runs:
            if active >= args.max_parallel:
                break

            ok, reason = should_start(r)
            if not ok:
                continue

            # 原子加锁（防重复启动）
            if not atomic_create_lock(r.lock_path):
                continue

            cmd = build_cmd(args.cmd_template, r)
            write_text(r.cmd_path, cmd + "\n")

            if args.dry_run:
                print(f"[watcher] DRYRUN start {r.run_dir} -> {cmd}")
                # dry-run 也不应留下 lock；回收
                try:
                    r.lock_path.unlink(missing_ok=True)  # py3.8+ ok? (3.8 no). 为稳起见用 try/except
                except Exception:
                    pass
                continue

            pid = launch_background(r, cmd)
            write_text(r.pid_path, str(pid) + "\n")
            print(f"[watcher] START {r.run_dir} pid={pid}")
            active += 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()

