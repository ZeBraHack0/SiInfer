import json
import os
from typing import Any, Dict, List, Tuple
import re
from typing import Iterator, Optional
from siinfer_adapter import BenchmarkAdapter


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)

def _parse_rid_num(rid: str) -> int:
    """
    rid 形如: evalplus-mbpp/00000000
    """
    m = re.search(r"/(\d+)$", rid)
    return int(m.group(1)) if m else 0


def _load_vllm_bench_result_texts(path: str) -> Tuple[List[str], List[str]]:
    """
    vllm bench serve --save-result 的 result-filename 文件：
      - 单个 JSON dict
      - generated_texts: list[str]
      - errors: list[str]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    texts = obj.get("generated_texts")
    if not isinstance(texts, list):
        raise ValueError(f"invalid vllm result: missing generated_texts in {path}")

    errs = obj.get("errors", [""] * len(texts))
    if not isinstance(errs, list) or len(errs) != len(texts):
        errs = [""] * len(texts)

    # 规范化为 str
    out_texts: List[str] = []
    for t in texts:
        out_texts.append("" if t is None else str(t))

    out_errs: List[str] = []
    for e in errs:
        out_errs.append("" if e is None else str(e))

    return out_texts, out_errs


class Adapter(BenchmarkAdapter):
    """
    目标：pull 阶段生成的 samples/<p_name>/<sidx>.py 与原 generate.py 逻辑完全一致
    """
    eval_input_ext = "jsonl"

    def materialize_eval_input(
        self,
        responses_path: str,  # 现在传 vllm bench 的 result-filename（JSON，不是 jsonl）
        meta_path: str,
        out_path: str,
        cfg: Dict[str, Any],
    ) -> None:
        workdir = cfg["workdir"]
        os.makedirs(workdir, exist_ok=True)

        # 0) 读 vllm bench result：按数据集顺序排列的输出
        gen_texts, gen_errs = _load_vllm_bench_result_texts(responses_path)

        # 1) 读 meta：收集全局顺序（用 rid_num 排序对齐 vllm 输出顺序）
        # meta.jsonl 行：{"id": rid, "meta": {...}}
        meta_items: List[Tuple[int, str, Dict[str, Any]]] = []
        groups: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}

        for row in _read_jsonl(meta_path):
            rid = row.get("id")
            m = row.get("meta") or {}
            task_id = m.get("task_id")
            if not rid or not task_id:
                continue
            rid_num = _parse_rid_num(rid)
            meta_items.append((rid_num, rid, m))
            groups.setdefault(task_id, []).append((rid_num, rid, m))

        meta_items.sort(key=lambda x: x[0])

        # 2) 对齐：第 i 条 meta <-> 第 i 条 generated_texts
        # 兼容长度不一致：越界当空；errors 非空也当空（算 missing）
        resp_map: Dict[str, str] = {}
        n_pair = min(len(meta_items), len(gen_texts))

        for i, (_, rid, _) in enumerate(meta_items):
            if i < len(gen_texts):
                if i < len(gen_errs) and gen_errs[i]:
                    resp_map[rid] = ""
                else:
                    resp_map[rid] = gen_texts[i]
            else:
                resp_map[rid] = ""

        # 3) 写 samples/<p_name>/<sidx>.py（严格复刻 generate.py 处理逻辑）
        summary: List[Dict[str, Any]] = []

        for task_id, items in groups.items():
            items.sort(key=lambda x: x[0])  # rid_num 递增

            _, _, m0 = items[0]
            p_name = m0["p_name"]
            task_prompt = m0.get("task_prompt", "")
            direct_completion = bool(m0.get("direct_completion", False))
            sidx = int(m0.get("sidx_start", 0))

            task_dir = os.path.join(workdir, p_name)
            os.makedirs(task_dir, exist_ok=True)

            n_total = 0
            n_written = 0
            n_missing_resp = 0

            for _, rid, m in items:
                n_total += 1
                impl = resp_map.get(rid, "")
                if not impl:
                    n_missing_resp += 1

                impl = (impl or "").replace("\t", "    ")

                if "```python" in impl:
                    start = impl.find("```python") + 9
                    end = impl.find("```", start)
                    if end != -1:
                        impl = impl[start:end].strip()
                    else:
                        impl = impl[start:].strip()
                elif "```" in impl:
                    impl = impl.split("```")[0]

                # 关键：如果 direct_completion=True，但模型输出已经包含 task_prompt 前缀，避免重复拼接
                if direct_completion:
                    if task_prompt and impl.startswith(task_prompt):
                        content = impl
                    else:
                        content = task_prompt + impl
                else:
                    content = impl

                try:
                    with open(os.path.join(task_dir, f"{sidx}.py"), "w", encoding="utf-8") as f:
                        f.write(content)
                except UnicodeEncodeError:
                    continue

                sidx += 1
                n_written += 1

            summary.append({
                "task_id": task_id,
                "p_name": p_name,
                "direct_completion": direct_completion,
                "sidx_start": int(m0.get("sidx_start", 0)),
                "num_requests": n_total,
                "num_written": n_written,
                "num_missing_resp": n_missing_resp,
                "workdir": workdir,
                "vllm_num_prompts": len(gen_texts),
                "vllm_num_paired": n_pair,
            })

        # 4) debug jsonl
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in summary:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
