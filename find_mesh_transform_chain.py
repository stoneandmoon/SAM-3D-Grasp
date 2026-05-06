#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

ROOT = "/root/SAM-3D-Grasp"
IGNORE_DIRS = {
    ".git", "__pycache__", "build", "dist", ".mypy_cache",
    ".pytest_cache", "node_modules"
}
TEXT_EXTS = {".py"}

PATTERNS = [
    r"\bdef\s+decode_slat\s*\(",
    r"\bdef\s+postprocess_slat_output\s*\(",
    r"\bglb\b",
    r"\bgaussian\b",
    r"\bmesh\b",
    r"\bTrimesh\b",
    r"\bo3d\b",
    r"\bcompose_transform\b",
    r"\bdecompose_transform\b",
    r"\btransform_points\b",
    r"\btransform\b",
    r"\bapply_transform\b",
    r"\bscale\b",
    r"\brotation\b",
    r"\btranslation\b",
    r"\bcoords\b",
    r"\bshape\b",
]

def main():
    cps = [re.compile(p) for p in PATTERNS]
    out_path = os.path.join(ROOT, "mesh_transform_chain.txt")
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
                        out.write("=" * 100 + "\n")
                        out.write(f"{path}:{i}\n")
                        start = max(0, i - 5)
                        end = min(len(lines), i + 20)
                        for k in range(start, end):
                            out.write(f"{k+1:04d}: {lines[k]}")
                        out.write("\n")

    print(f"完成，共命中 {total} 处。")
    print(f"输出文件: {out_path}")

if __name__ == "__main__":
    main()