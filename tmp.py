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
RESPONSE_FAILED = "RESPONSE_FAILED"

CLIENT_LOCK = ".client.lock"
CLIENT_PID = ".client.pid"
CLIENT_CMD = ".client.cmd"
CLIENT_SENT = ".client.sent"          # 发送记录：存在则永不再处理
CLIENT_DONE = ".client.done"
CLIENT_FAIL = ".client.fail"
CLIENT_RC = ".client.rc"
CLIENT_CONFIG = "client_config.json"  # watcher 生成的 client 配置


# 需要从 meta.jsonl 第一条里提取并透传的参数（写死 list）
META_FORWARD_KEYS = ["max_tokens", "temperature", "top_p"]

# meta key -> client_extra 里的 flag key（你例子里是 "--top-k" 这种风格）
META_KEY_TO_FLAG = {
    "max_tokens": "--max-tokens",
    "temperature": "--temperature",
    "top_p": "--top-p",
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
    """
    只读 meta.jsonl 第一条；默认认为同一数据集采样参数一致。
    返回：{ "--top-p": 1.0, "--temperature": 0.0, ... }
    """
    obj = read_first_jsonl(meta_path)
    meta = obj.get("meta", {})
    out: Dict[str, Any] = {}
    for k in META_FORWARD_KEYS:
        if k not in meta:
            continue
        flag = META_KEY_TO_FLAG.get(k, f"--{k.replace('_', '-')}")
        out[flag] = meta[k]
    return out


def ensure_client_extra_entry(cfg: Dict[str, Any], backend_name: str) -> Dict[str, Any]:
    """
    cfg["client_extra"] 是 list[dict]，找到 backend==backend_name 的那条，没有就创建一条。
    返回那条 dict 的引用。
    """
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
    def fail_path(self) -> Path:
        return self.run_dir / CLIENT_FAIL

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

    @property
    def response_failed(self) -> Path:
        return self.run_dir / RESPONSE_FAILED


def discover_run_dirs(workspace: Path) -> List[RunDir]:
    """
    自动发现 RUN_DIR：workspace 下三层目录
      ${workspace}/${RUN_DIR}/${SII_RUN_ID}/${SII_BENCH_NAME}/REQUEST_READY
    """
    out: List[RunDir] = []
    if not workspace.exists():
        return out

    for run_dir in workspace.iterdir():          # ${RUN_DIR}
        if not run_dir.is_dir():
            continue
        for run_id_dir in run_dir.iterdir():     # ${SII_RUN_ID}
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


def build_client_config(base_cfg: Dict[str, Any], r: RunDir) -> Dict[str, Any]:
    """
    生成最终 client config：
      - 从 base_cfg 深拷贝
      - 覆盖 output/datapath 到当前 run_dir（保证结果写回子目录）
      - 可选：num_prompt 自动设成 requests.jsonl 行数（更贴近真实）
      - 将 meta.jsonl 第一条解析出的参数写入 client_extra
    """
    cfg = copy.deepcopy(base_cfg)

    # 1) 强制写回当前子目录（你需求#3）
    cfg["output"] = str(r.run_dir)
    cfg["datapath"] = str(r.run_dir)

    # 2) num_prompt：如果 base 里是 [..] 列表风格，保持一致
    try:
        nreq = count_jsonl_lines(r.requests_path) if r.requests_path.exists() else None
        if nreq is not None:
            # 兼容两种写法：num_prompt: [512] 或 num_prompt: 512
            if isinstance(cfg.get("num_prompt"), list):
                cfg["num_prompt"] = [nreq]
            else:
                cfg["num_prompt"] = nreq
    except Exception:
        pass

    # 3) 注入 client_extra（所有 meta 参数都塞进去）
    # backend 名：优先取 cfg["backend"][0]，否则默认 "vllm"
    backend_name = "vllm"
    b = cfg.get("backend")
    if isinstance(b, list) and b:
        backend_name = str(b[0])

    meta_flags = parse_meta_forward_params(r.meta_path)
    ce_item = ensure_client_extra_entry(cfg, backend_name)
    for flag, val in meta_flags.items():
        ce_item[flag] = val

    # （可选）你也可以把 requests/meta/response 路径塞进 cfg 里，方便 client 读取
    # 但你给的 schema 里没有这些字段，所以这里不强行加；后续你定 schema 再对齐。

    return cfg


def build_cmd(cmd_template: str, r: RunDir) -> str:
    """
    cmd-template 可用占位符（会自动 quote 路径）：
      {run_dir} {requests_path} {meta_path} {responses_path} {config_path}
    """
    mapping: Dict[str, str] = {
        "run_dir": sh_quote(str(r.run_dir)),
        "requests_path": sh_quote(str(r.requests_path)),
        "meta_path": sh_quote(str(r.meta_path)),
        "responses_path": sh_quote(str(r.responses_path)),
        "config_path": sh_quote(str(r.config_path)),
    }
    return cmd_template.format(**mapping).strip()


def launch_background(r: RunDir, cmd: str) -> int:
    """
    启动后台进程：
      - 这里仍保留“成功且 responses.jsonl 存在才 READY”的约束
      - 你现在还没定 client 命令，建议先用 --dry-run 验证 config 生成
    """
    r.run_dir.mkdir(parents=True, exist_ok=True)

    wrapper = f"""
set -euo pipefail
cd {sh_quote(str(r.run_dir))}

set +e
{cmd}
rc=$?
set -e

echo "$rc" > {sh_quote(str(r.rc_path))}

if [ "$rc" -eq 0 ] && [ -f {sh_quote(str(r.responses_path))} ]; then
  touch {sh_quote(RESPONSE_READY)}
  touch {sh_quote(str(r.done_path))}
else
  touch {sh_quote(RESPONSE_FAILED)}
  touch {sh_quote(str(r.fail_path))}
fi

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
        if pid is not None and pid_alive(pid) and (not r.done_path.exists()) and (not r.fail_path.exists()):
            n += 1
    return n


def maybe_recover_prelaunch_stale_lock(r: RunDir, verbose: bool = False) -> None:
    """
    只回收“安全”的陈旧锁：lock 存在但 sent 不存在且 pid 不存在/已死
    """
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
    if r.response_failed.exists():
        return False, "already RESPONSE_FAILED"
    if r.done_path.exists() or r.fail_path.exists():
        return False, "already marked done/fail"
    if r.lock_path.exists():
        pid = load_pid(r)
        if pid is not None and pid_alive(pid):
            return False, f"locked and running pid={pid}"
        return False, "locked (stale or unknown); delete lock/sent manually if needed"
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

            # 1) 先生成 config（写到该子目录）
            try:
                cfg = build_client_config(base_cfg, r)
                write_json(r.config_path, cfg)
            except Exception as e:
                safe_unlink(r.lock_path)
                if args.verbose:
                    print(f"[watcher] skip {r.run_dir} (build config failed: {e})")
                continue

            # 2) 再构造并记录启动命令（后续你定真实命令即可）
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

            # 发送记录：满足“已发送不再处理”
            sent_payload = f"time={int(time.time())}\npid={pid}\nconfig={r.config_path}\ncmd={cmd}\n"
            write_text(r.sent_path, sent_payload)

            print(f"[watcher] START {r.run_dir} pid={pid}")
            active += 1

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
