from __future__ import annotations
import os, json, time, threading
from typing import Any, Dict, Optional, List

import os
import importlib.util
from pathlib import Path
from typing import Optional, Type, Any

REQUEST_READY = "REQUEST_READY"
RESPONSE_READY = "RESPONSE_READY"

def _touch(path: str) -> None:
    with open(path, "a", encoding="utf-8"):
        os.utime(path, None)

def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

class BenchmarkAdapter:
    name: str = "unknown"
    eval_input_ext: str = "jsonl"

    def __init__(
        self,
        base_dir: str,
        run_id: str,
        bench_name: str,
        chunk_size: int = 4096,
    ):
        self.base_dir = base_dir
        self.run_id = run_id
        self.name = bench_name
        self.chunk_size = int(chunk_size)

        self.run_dir = os.path.join(base_dir, run_id, bench_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._finalized = False
        self._next_id = 0

        self._req_f = None
        self._meta_f = None
        self._req_tmp = os.path.join(self.run_dir, "requests.jsonl.tmp")
        self._meta_tmp = os.path.join(self.run_dir, "meta.jsonl.tmp")

        # 关键：内存缓冲
        self._req_buf: List[Dict[str, Any]] = []
        self._meta_buf: List[Dict[str, Any]] = []

    # 每个 benchmark 实现：把统一 responses+meta 转成 evaluator 输入
    def materialize_eval_input(self, responses_path: str, meta_path: str, out_path: str, cfg: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _ensure_files_open(self) -> None:
        if self._req_f is None:
            self._req_f = open(self._req_tmp, "a", encoding="utf-8")
            self._meta_f = open(self._meta_tmp, "a", encoding="utf-8")

    def _flush_locked(self) -> None:
        """要求已持有 self._lock；把 buffer 一次性写入文件。"""
        if not self._req_buf:
            return
        self._ensure_files_open()
        for r in self._req_buf:
            self._req_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for m in self._meta_buf:
            self._meta_f.write(json.dumps(m, ensure_ascii=False) + "\n")
        self._req_buf.clear()
        self._meta_buf.clear()
        # 离线吞吐优先：这里只 flush，不每次都 fsync（finalize 时再 fsync）
        self._req_f.flush()
        self._meta_f.flush()

    # 对外 API 1：只入队，满 chunk 才落盘
    def iterate_request(
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (prompt is None) == (messages is None):
            raise ValueError("Provide exactly one of prompt or messages.")

        with self._lock:
            if self._finalized:
                raise RuntimeError("Adapter already finalized; cannot iterate_request after pull().")

            rid = f"{self.name}/{self._next_id:08d}"
            self._next_id += 1

            req_row: Dict[str, Any] = {"id": rid}
            if prompt is not None:
                req_row["prompt"] = prompt
            else:
                req_row["messages"] = messages

            meta_row: Dict[str, Any] = {"id": rid, "meta": meta or {}}

            self._req_buf.append(req_row)
            self._meta_buf.append(meta_row)

            if len(self._req_buf) >= self.chunk_size:
                self._flush_locked()

    def _finalize_requests(self) -> None:
        with self._lock:
            if self._finalized:
                return
            # 先 flush 剩余 buffer
            self._flush_locked()

            self._finalized = True
            # close + fsync（保证 tmp 文件落盘）
            if self._req_f is not None:
                self._req_f.flush(); os.fsync(self._req_f.fileno()); self._req_f.close()
                self._meta_f.flush(); os.fsync(self._meta_f.fileno()); self._meta_f.close()
                self._req_f = None
                self._meta_f = None
            else:
                # 从未写过任何数据，也要保证 tmp 文件存在
                open(self._req_tmp, "a", encoding="utf-8").close()
                open(self._meta_tmp, "a", encoding="utf-8").close()

            # 原子发布
            os.replace(self._req_tmp, os.path.join(self.run_dir, "requests.jsonl"))
            os.replace(self._meta_tmp, os.path.join(self.run_dir, "meta.jsonl"))

            manifest = {
                "spec_version": "v1",
                "bench": self.name,
                "run_id": self.run_id,
                "requests_path": "requests.jsonl",
                "meta_path": "meta.jsonl",
                "responses_path": "responses.jsonl",
                "eval_input_path": f"eval_input.{self.eval_input_ext}",
                "num_requests": self._next_id,
            }
            _atomic_write_json(os.path.join(self.run_dir, "manifest.json"), manifest)
            _touch(os.path.join(self.run_dir, REQUEST_READY))

    # 对外 API 2：finalize + 阻塞等待 + 生成 eval 输入
    def pull(
        self,
        cfg: Dict[str, Any],
        poll_interval_s: float = 1.0,
        timeout_s: Optional[float] = None,
    ) -> str:
        self._finalize_requests()

        ready_resp = os.path.join(self.run_dir, RESPONSE_READY)
        start = time.time()
        while not os.path.exists(ready_resp):
            if timeout_s is not None and (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timeout waiting for {ready_resp}")
            time.sleep(poll_interval_s)

        responses_path = os.path.join(self.run_dir, "responses.jsonl")
        meta_path = os.path.join(self.run_dir, "meta.jsonl")
        eval_input_path = os.path.join(self.run_dir, f"eval_input.{self.eval_input_ext}")
        self.materialize_eval_input(responses_path, meta_path, eval_input_path, cfg)
        return eval_input_path


# ============================
# Loader: locate & load per-benchmark adapter subclasses
# ============================


def _find_repo_root() -> Path:
    """
    Resolve SiInfer repo root.
    Priority:
      1) env SII_ADAPTER_REPO_ROOT
      2) walk up from this file to find manifest.json (your repo has SiInfer/manifest.json)
      3) fallback: parent dir of this file
    """
    env_root = os.getenv("SII_ADAPTER_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    # If adapter.py is symlinked into a package, resolve() returns real file location.
    start = here.parent
    for p in [start, *start.parents]:
        if (p / "manifest.json").is_file():
            return p
    return start


def find_adapter_file(bench_name: str, group: Optional[str] = None) -> Optional[Path]:
    """
    Find per-benchmark adapter implementation.

    Expected layout (your plan):
      SiInfer/base/<bench_name>/adapter.py
      SiInfer/chat/<bench_name>/adapter.py
      SiInfer/tool/<bench_name>/adapter.py

    bench_name examples:
      "evalplus-mbpp", "bigcodebench-base", ...
    group: "base" | "chat" | "tool" | None (auto-search)
    """
    root = _find_repo_root()
    groups = [group] if group else ["base", "chat", "tool"]
    for g in groups:
        p = root / g / bench_name / "adapter.py"
        if p.is_file():
            return p
    return None


def load_adapter_class_from_file(py_file: Path) -> Type["BenchmarkAdapter"]:
    """
    Dynamically import adapter subclass from a file path.
    Priority:
      1) symbol named "Adapter"
      2) first subclass of BenchmarkAdapter found in globals
    """
    py_file = py_file.resolve()
    mod_name = f"siinfer_adapter_plugin_{py_file.stem}_{abs(hash(str(py_file)))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(py_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {py_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # 1) Preferred symbol name
    if hasattr(module, "Adapter"):
        cls = getattr(module, "Adapter")
        if isinstance(cls, type) and issubclass(cls, BenchmarkAdapter) and cls is not BenchmarkAdapter:
            return cls

    # 2) Fallback: first subclass
    for v in vars(module).values():
        if isinstance(v, type) and issubclass(v, BenchmarkAdapter) and v is not BenchmarkAdapter:
            return v

    raise ImportError(f"No BenchmarkAdapter subclass found in {py_file}")


def get_benchmark_adapter(
    *,
    base_dir: str,
    run_id: str,
    bench_name: str,
    group: Optional[str] = None,
    chunk_size: int = 4096,
) -> "BenchmarkAdapter":
    """
    Instantiate per-benchmark adapter (subclass) by locating <group>/<bench_name>/adapter.py.

    This intentionally fails hard if no implementation is found,
    to avoid silently producing wrong evaluator output.
    """
    py_file = find_adapter_file(bench_name, group=group)
    if py_file is None:
        raise FileNotFoundError(
            f"Adapter impl not found for bench='{bench_name}'. "
            f"Expected: SiInfer/{{base|chat|tool}}/{bench_name}/adapter.py. "
            f"(repo_root={_find_repo_root()})"
        )
    cls = load_adapter_class_from_file(py_file)
    return cls(base_dir=base_dir, run_id=run_id, bench_name=bench_name, chunk_size=chunk_size)


# Optional: what we want to expose when people do `from adapter import *`
__all__ = [
    # base adapter
    "BenchmarkAdapter",
    "REQUEST_READY",
    "RESPONSE_READY",
    # loader
    "find_adapter_file",
    "load_adapter_class_from_file",
    "get_benchmark_adapter",
]
