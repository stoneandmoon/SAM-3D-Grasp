#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

ROOT = "/root/SAM-3D-Grasp"

PATTERNS = [
    r"results\[['\"]rotation['\"]\]",
    r"results\[['\"]translation['\"]\]",
    r"results\[['\"]scale['\"]\]",
    r"results\[['\"]translation_scale['\"]\]",
    r"results\[['\"]pointmap['\"]\]",
    r"results\[['\"]pointmap_colors['\"]\]",
    r"results\[['\"]6drotation_normalized['\"]\]",
    r"results\[['\"]coords['\"]\]",
    r"results\[['\"]coords_original['\"]\]",
]

TEXT_EXTS = {".py"}
IGNORE_DIRS = {
    ".git", "__pycache__", "build", "dist", ".mypy_cache",
    ".pytest_cache", "node_modules"
}


def is_py(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in TEXT_EXTS


def main():
    compiled = [re.compile(p) for p in PATTERNS]
    hits = []

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for fn in files:
            path = os.path.join(root, fn)
            if not is_py(path):
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for i, line in enumerate(lines, start=1):
                for cp in compiled:
                    if cp.search(line):
                        start = max(0, i - 4)
                        end = min(len(lines), i + 4)

                        context = "".join(
                            f"{k+1:04d}: {lines[k]}"
                            for k in range(start, end)
                        )
                        hits.append((path, i, cp.pattern, context))
                        break

    out_path = os.path.join(ROOT, "results_flow_trace.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for path, lineno, pattern, context in hits:
            out.write("=" * 100 + "\n")
            out.write(f"FILE: {path}\n")
            out.write(f"LINE: {lineno}\n")
            out.write(f"PATTERN: {pattern}\n")
            out.write("-" * 100 + "\n")
            out.write(context)
            out.write("\n")

    print(f"追踪完成，共命中 {len(hits)} 处。")
    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    main()