#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

ROOT = "/root/SAM-3D-Grasp"

KEYWORDS = [
    "rotation",
    "translation",
    "scale",
    "translation_scale",
    "6drotation_normalized",
    "pointmap",
    "coords",
    "coords_original",
    "quaternion",
    "quat",
    "canonical",
    "normalize",
    "denormalize",
    "transform",
    "extrinsic",
    "intrinsic",
]

TEXT_EXTS = {
    ".py", ".yaml", ".yml", ".json", ".txt", ".md"
}

IGNORE_DIRS = {
    ".git", "__pycache__", "build", "dist", ".mypy_cache",
    ".pytest_cache", "node_modules"
}


def is_text_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in TEXT_EXTS


def main():
    results = []

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for fn in files:
            path = os.path.join(root, fn)
            if not is_text_file(path):
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for i, line in enumerate(lines, start=1):
                hit_keys = [k for k in KEYWORDS if k in line]
                if hit_keys:
                    results.append((path, i, line.rstrip("\n"), hit_keys))

    # 输出
    out_path = os.path.join(ROOT, "pose_symbol_hits.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for path, lineno, line, keys in results:
            out.write(f"{path}:{lineno} | keys={keys}\n")
            out.write(f"    {line}\n\n")

    print(f"搜索完成，共命中 {len(results)} 处。")
    print(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
    