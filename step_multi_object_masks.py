#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step_multi_object_masks.py

功能：
  使用你现有的 SAM3Segmenter，对一张 RGB 图像中的多个物体分别生成掩码。

输出：
  output_multi_masks/
    ├── yellow_rubber_duck_mask.png
    ├── yellow_rubber_duck_mask.npy
    ├── yellow_rubber_duck_overlay.png
    ├── white_ceramic_pitcher_mask.png
    ├── white_ceramic_pitcher_mask.npy
    ├── white_ceramic_pitcher_overlay.png
    ├── ...
    ├── all_masks_overlay.png
    └── summary.json

默认分割的 5 个目标：
  1. yellow rubber duck
  2. white ceramic pitcher
  3. red toy car
  4. small white bowl
  5. blue cup

运行示例：
  python step_multi_object_masks.py

或指定路径：
  python step_multi_object_masks.py \
      --image /root/SAM-3D-Grasp/data/test/000002/rgb/000000.png \
      --out-dir ./output_multi_masks

或自定义目标：
  python step_multi_object_masks.py \
      --targets "yellow duck,white mug,red toy car,white bowl,green tennis ball"
"""

import os
import sys
import json
import argparse
from pathlib import Path

# =========================================================
# 环境设置
# =========================================================
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

# =========================================================
# 导入你的 SAM3 分割器
# =========================================================
try:
    from modules.sam3_segment_pipeline4 import SAM3Segmenter
except Exception:
    from modules.sam3_segment import SAM3Segmenter


# =========================================================
# 路径工具
# =========================================================
def resolve_path(path_like, must_exist=False):
    """
    兼容：
      data/...
      ./data/...
      SAM-3D-Grasp/data/...
      /root/SAM-3D-Grasp/data/...
    """
    p = Path(path_like)

    if p.is_absolute():
        q = p
    else:
        raw = str(p).replace("\\", "/")
        repo_prefix = REPO_ROOT.name + "/"

        if raw.startswith(repo_prefix):
            q = REPO_ROOT / raw[len(repo_prefix):]
        else:
            q = REPO_ROOT / p

    if must_exist and not q.exists():
        raise FileNotFoundError(f"找不到文件: {q}")

    return q


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# =========================================================
# 参数解析
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="使用 SAM3 对图中多个物体生成掩码")

    parser.add_argument(
        "--image",
        type=str,
        default="data/test/000002/rgb/000000.png",
        help="输入 RGB 图像路径（默认使用 pipeline4 的测试图）"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="./output_multi_masks",
        help="输出目录"
    )

    parser.add_argument(
        "--targets",
        type=str,
        default="yellow rubber duck,white ceramic pitcher,red toy car,small white bowl,blue cup",
        help="要分割的目标列表，逗号分隔"
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="最小有效 mask 像素面积，小于此值认为分割失败"
    )

    return parser.parse_args()


# =========================================================
# 通用工具
# =========================================================
def slugify(text):
    text = text.lower().strip().replace(" ", "_")
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["_", "-"]:
            keep.append(ch)
    return "".join(keep)


def parse_targets(targets_text):
    items = [x.strip() for x in targets_text.split(",")]
    items = [x for x in items if len(x) > 0]
    return items


def normalize_mask(mask):
    """
    统一输出为 uint8 的 HxW 二值 mask，取值 0/255
    """
    if mask is None:
        return None

    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = np.squeeze(mask)

    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)

    mask = mask * 255
    return mask


def save_mask(mask, png_path, npy_path):
    cv2.imwrite(str(png_path), mask)
    np.save(str(npy_path), (mask > 0).astype(np.uint8))


def save_overlay(image_bgr, mask, save_path, label):
    """
    单个目标可视化
    """
    vis = image_bgr.copy()
    mask_bool = mask > 0

    # 绿色半透明覆盖
    overlay = vis.copy()
    overlay[mask_bool] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)

    # 红色轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    # 画框 + 标签
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 100:
            continue

        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)

        y_text_top = max(0, y - 28)
        cv2.rectangle(vis, (x, y_text_top), (min(x + w, vis.shape[1] - 1), y), (255, 255, 0), -1)
        cv2.putText(
            vis,
            label,
            (x + 4, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(save_path), vis)


def save_combined_overlay(image_bgr, masks_dict, save_path):
    """
    所有目标综合可视化
    """
    vis = image_bgr.copy()

    colors = [
        (0, 255, 0),    # green
        (255, 0, 0),    # blue
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (255, 255, 0),  # cyan
        (0, 128, 255),
        (128, 0, 255),
        (255, 128, 0),
    ]

    alpha = 0.30

    for idx, (label, mask) in enumerate(masks_dict.items()):
        color = colors[idx % len(colors)]
        mask_bool = mask > 0

        color_layer = np.zeros_like(vis)
        color_layer[:, :] = color

        vis[mask_bool] = (
            vis[mask_bool].astype(np.float32) * (1 - alpha)
            + color_layer[mask_bool].astype(np.float32) * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)

        ys, xs = np.where(mask_bool)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            cv2.putText(
                vis,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(save_path), vis)


# =========================================================
# 主函数
# =========================================================
def main():
    args = parse_args()

    image_path = resolve_path(args.image, must_exist=True)
    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)

    targets = parse_targets(args.targets)
    if len(targets) == 0:
        raise ValueError("targets 为空，请至少提供一个目标")

    print("=" * 100)
    print("🚀 多目标 SAM3 掩码生成开始")
    print(f"[Image]   {image_path}")
    print(f"[OutDir]  {out_dir}")
    print(f"[Targets] {targets}")
    print("=" * 100)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"读不到图像: {image_path}")

    sam = SAM3Segmenter()

    all_masks = {}
    summary = {
        "image": str(image_path),
        "out_dir": str(out_dir),
        "targets_requested": targets,
        "targets_succeeded": [],
        "targets_failed": [],
        "items": []
    }

    try:
        for idx, target in enumerate(targets, start=1):
            print("\n" + "-" * 100)
            print(f"[{idx}/{len(targets)}] 正在分割目标: {target}")
            print("-" * 100)

            mask = sam.segment_by_text(str(image_path), target)

            if mask is None:
                print(f"❌ 未找到目标: {target}")
                summary["targets_failed"].append(target)
                summary["items"].append({
                    "target": target,
                    "success": False,
                    "reason": "segment_by_text returned None"
                })
                continue

            mask = normalize_mask(mask)
            area = int((mask > 0).sum())

            if area < args.min_area:
                print(f"❌ 目标 {target} 的 mask 面积过小: {area} < {args.min_area}")
                summary["targets_failed"].append(target)
                summary["items"].append({
                    "target": target,
                    "success": False,
                    "reason": f"mask area too small: {area}",
                    "mask_area": area
                })
                continue

            stem = slugify(target)

            mask_png = out_dir / f"{stem}_mask.png"
            mask_npy = out_dir / f"{stem}_mask.npy"
            overlay_png = out_dir / f"{stem}_overlay.png"

            save_mask(mask, mask_png, mask_npy)
            save_overlay(image_bgr, mask, overlay_png, target)

            all_masks[target] = mask
            summary["targets_succeeded"].append(target)
            summary["items"].append({
                "target": target,
                "success": True,
                "mask_area": area,
                "mask_png": str(mask_png),
                "mask_npy": str(mask_npy),
                "overlay_png": str(overlay_png),
            })

            print(f"✅ 分割成功: {target}")
            print(f"   mask area : {area}")
            print(f"   mask png  : {mask_png}")
            print(f"   mask npy  : {mask_npy}")
            print(f"   overlay   : {overlay_png}")

        # 综合可视化
        if len(all_masks) > 0:
            combined_overlay = out_dir / "all_masks_overlay.png"
            save_combined_overlay(image_bgr, all_masks, combined_overlay)
            summary["all_masks_overlay"] = str(combined_overlay)
            print("\n✅ 综合可视化已保存:")
            print(f"   {combined_overlay}")
        else:
            print("\n⚠️ 没有任何目标分割成功。")

    finally:
        print("\n[Memory] 正在卸载 SAM3 模型...")
        try:
            sam._unload_model()
            print("✅ SAM3 模型已卸载。")
        except Exception as e:
            print(f"⚠️ 卸载失败，但不影响结果保存: {e}")

    # 保存 summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print("🎉 全部完成")
    print(f"[Summary] {summary_path}")
    print(f"[Success] {len(summary['targets_succeeded'])} / {len(targets)}")
    print("=" * 100)


if __name__ == "__main__":
    main()