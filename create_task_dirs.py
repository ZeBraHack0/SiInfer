#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def sanitize(name: str) -> str:
    """Make a safe directory name (cross-platform)."""
    name = name.strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "unnamed"


def extract_return_list_from_func(tree: ast.AST, func_name: str) -> Optional[List[dict]]:
    """
    Parse `def func_name(): return [ ... ]` and literal-eval the returned list.
    (No imports / execution; purely AST + literal_eval.)
    """
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    try:
                        v = ast.literal_eval(stmt.value)
                    except Exception:
                        return None
                    if isinstance(v, list):
                        # keep only dict items with task_name
                        out = []
                        for it in v:
                            if isinstance(it, dict) and "task_name" in it:
                                out.append(it)
                        return out
    return None


def load_tasks(task_define_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      {
        "base": [{"task_name": ..., "optional_params": {...}}, ...],
        "chat": [...],
        "tool": [...],
      }
    """
    src = task_define_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(task_define_path))

    mapping = {
        "base": "get_base_tasks",
        "chat": "get_chat_tasks",
        "tool": "get_tool_calling_tasks",
    }

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for g, fn in mapping.items():
        items = extract_return_list_from_func(tree, fn)
        if items is None:
            raise RuntimeError(f"Failed to parse {fn}() from {task_define_path}")
        groups[g] = items
    return groups


def main():
    ap = argparse.ArgumentParser(description="Create per-task folders based on task_define.py (base/chat/tool).")
    ap.add_argument("--task-define", required=True, help="Path to task_define.py")
    ap.add_argument("--outdir", required=True, help="Output root directory")
    ap.add_argument("--layout", choices=["group/task", "flat"], default="group/task",
                    help="Directory layout: 'group/task' (default) or 'flat' (group__task)")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be created")
    args = ap.parse_args()

    task_define_path = Path(args.task_define).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    groups = load_tasks(task_define_path)

    # Enforce expected counts (your constraint: base 11 / chat 20 / tool 3)
    expected = {"base": 11, "chat": 20, "tool": 3}
    actual = {k: len(v) for k, v in groups.items()}
    for g in ["base", "chat", "tool"]:
        if actual.get(g, 0) != expected[g]:
            print(f"[WARN] {g} count mismatch: expected {expected[g]}, got {actual.get(g, 0)}")

    created: List[Dict[str, str]] = []
    total = 0

    for group_name in ["base", "chat", "tool"]:
        tasks = groups[group_name]
        for item in tasks:
            task_name = str(item.get("task_name", "")).strip()
            if not task_name:
                continue

            g = sanitize(group_name)
            t = sanitize(task_name)

            if args.layout == "group/task":
                path = outdir / g / t
            else:
                path = outdir / f"{g}__{t}"

            if args.dry_run:
                print(f"[dry-run] mkdir -p {path}")
            else:
                path.mkdir(parents=True, exist_ok=True)

                # 写一个 meta.json，方便后续脚本复用（可删）
                meta = {
                    "group": group_name,
                    "task_name": task_name,
                    "optional_params": item.get("optional_params", {}) or {},
                }
                (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            created.append({
                "group": group_name,
                "task_name": task_name,
                "dir": str(path),
            })
            total += 1

    # 写清单
    manifest = outdir / "manifest.json"
    if args.dry_run:
        print(f"[dry-run] would write manifest: {manifest} (items={len(created)})")
    else:
        manifest.write_text(json.dumps(created, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. total={total}, outdir={outdir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
