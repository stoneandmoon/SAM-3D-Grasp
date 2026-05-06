#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

ROOT = "/root/SAM-3D-Grasp"
TEXT_EXTS = {".py"}
IGNORE_DIRS = {
    ".git", "__pycache__", "build", "dist", ".mypy_cache",
    ".pytest_cache", "node_modules"
}

PATTERNS = [
    r"\bresults\s*=\s*{",
    r"\breturn\s+results\b",
    r"\bdef\s+run\s*\(",
    r"\bclass\s+.*Pipeline",
    r"\bpointmap\b",
    r"\b6drotation_normalized\b",
    r"\brotation\b",
    r"\btranslation\b",
    r"\bscale\b",
]

def main():
    cps = [re.compile(p) for p in PATTERNS]
    out_path = os.path.join(ROOT, "results_builder_hits.txt")

    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fn in files:
                path = os.path.join(root, fn)
                if os.path.splitext(path)[1].lower() not in TEXT_EXTS:
                    continue

                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                except Exception:
                    continue

                for i, line in enumerate(lines, start=1):
                    if any(cp.search(line) for cp in cps):
                        total += 1
                        out.write(f"{path}:{i}\n")
                        for k in range(max(0, i - 3), min(len(lines), i + 3)):
                            out.write(f"{k+1:04d}: {lines[k]}")
                        out.write("\n" + "-" * 80 + "\n\n")

    print(f"完成，共命中 {total} 处。")
    print(f"输出文件: {out_path}")


if __name__ == "__main__":
    main()