# eval_plus/api/evalplus_adapter.py
import os
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from benchmark_adapter import BenchmarkAdapter  # <- 你那份基类放哪就改哪

def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _parse_rid_num(rid: str) -> int:
    # rid 形如 "evalplus-mbpp/00000012"
    try:
        return int(rid.split("/")[-1])
    except Exception:
        return 10**18

def _extract_text(obj: Any) -> str:
    """
    尽量兼容多种 responses.jsonl 结构：
    - {"id":..., "text": "..."}
    - {"id":..., "completion": "..."}
    - {"id":..., "output": "..."}
    - {"id":..., "choices":[{"text":...}]}
    - {"id":..., "choices":[{"message":{"content":...}}]}
    - {"id":..., "response": {...}}  (递归)
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("text", "completion", "output", "generated_text", "content"):
            v = obj.get(k)
            if isinstance(v, str):
                return v
        if isinstance(obj.get("message"), dict) and isinstance(obj["message"].get("content"), str):
            return obj["message"]["content"]
        if isinstance(obj.get("choices"), list) and obj["choices"]:
            c0 = obj["choices"][0]
            if isinstance(c0, dict):
                if isinstance(c0.get("text"), str):
                    return c0["text"]
                if isinstance(c0.get("message"), dict) and isinstance(c0["message"].get("content"), str):
                    return c0["message"]["content"]
        if isinstance(obj.get("response"), dict):
            return _extract_text(obj["response"])
    return ""

class EvalPlusAdapter(BenchmarkAdapter):
    """
    目标：pull 阶段生成的 samples/<p_name>/<sidx>.py 与原 generate.py 逻辑完全一致
    """
    eval_input_ext = "jsonl"  # pull() 会返回 run_dir/eval_input.jsonl；内容用于 debug

    def materialize_eval_input(
        self,
        responses_path: str,
        meta_path: str,
        out_path: str,
        cfg: Dict[str, Any],
    ) -> None:
        # cfg 里必须给 workdir（即 generate.py 里的 --root，通常是 .../run_xxx/samples）
        workdir = cfg["workdir"]
        os.makedirs(workdir, exist_ok=True)

        # 1) 读 responses：id -> response_obj
        resp_map: Dict[str, Any] = {}
        for row in _read_jsonl(responses_path):
            rid = row.get("id")
            if not rid:
                continue
            resp_map[rid] = row

        # 2) 读 meta：按 task_id 分组，并按 rid 的全局递增顺序排列（复刻“生成顺序”）
        # meta.jsonl 行：{"id": rid, "meta": {...}}
        groups: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = {}
        for row in _read_jsonl(meta_path):
            rid = row.get("id")
            m = row.get("meta") or {}
            task_id = m.get("task_id")
            if not rid or not task_id:
                continue
            groups.setdefault(task_id, []).append((_parse_rid_num(rid), rid, m))

        # 3) 对每个 task：严格复刻 generate 写文件逻辑
        summary: List[Dict[str, Any]] = []

        for task_id, items in groups.items():
            items.sort(key=lambda x: x[0])  # 按 rid_num 排序，保证与 iterate_request 发出的顺序一致

            # 从首条 meta 取固定信息
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
                r = resp_map.get(rid)
                if r is None:
                    n_missing_resp += 1
                    impl = ""
                else:
                    impl = _extract_text(r)

                # === 复刻 decoder._post_process_generation：tab -> spaces ===
                impl = (impl or "").replace("\t", "    ")

                # === 下面这段：逐行逐句复刻 generate.py 里的处理逻辑（你贴的那段） ===
                if "```python" in impl:
                    start = impl.find("```python") + 9
                    end = impl.find("```", start)
                    if end != -1:
                        impl = impl[start:end].strip()
                    else:
                        impl = impl[start:].strip()
                    # print("Extracted code from python block.")  # 这里不强制 print，避免刷屏

                elif "```" in impl:
                    impl = impl.split("```")[0]
                    # print("``` exist in generation. Please check the generation results.")

                try:
                    with open(os.path.join(task_dir, f"{sidx}.py"), "w", encoding="utf-8") as f:
                        if direct_completion:
                            f.write(task_prompt + impl)
                        else:
                            f.write(impl)
                except UnicodeEncodeError:
                    # === 复刻 generate：continue 且不递增 sidx ===
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
            })

        # 4) out_path 写一份 debug 用 jsonl（不影响 evaluator；evaluator 继续读 samples/ 目录）
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in summary:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

