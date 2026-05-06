#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ho3d_hand_mask_grabcut_depth_demo.py

在原来的 GrabCut 基础上，利用深度图对手部 mask 做细化：
- 只保留候选区域中「深度最近」的一层（手更靠近相机）
- 只保留面积最大的连通块（避免整块物体被吃进去）
- 最后做一点形态学闭运算，让 mask 更连贯

使用示例（注意：不再需要显式传 meta，自动从 rgb 推导 .pkl）：
  python3 ho3d_hand_mask_grabcut_depth_demo.py \
    --rgb /home/zhn/下载/ho3dmini/HO3D_v3/train/ABF10/rgb/1020.jpg \
    --depth /home/zhn/下载/ho3dmini/HO3D_v3/train/ABF10/depth/1020.png \
    --out-prefix /home/zhn/下载/ho3dmini_hand_demo_depth/ABF10_1020 \
    --debug
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def refine_hand_mask_with_depth(raw_mask,
                                depth,
                                near_percent=25,
                                z_margin_mm=20,
                                min_area=300,
                                debug=False):
    """
    基于深度和面积约束，对 GrabCut 的手部掩膜进行细化，尽量去掉物体部分。

    参数
    ----
    raw_mask : np.uint8, 0/1
        GrabCut 输出的手部候选 mask（可能包含物体）。
    depth : np.ndarray
        对应的深度图，HO3D 中通常是以毫米存的 uint16。
    near_percent : int
        取 hand 区域中最近 near_percent% 的像素作为“手的深度基准”。
    z_margin_mm : float
        允许的深度余量（毫米）。比这个更远的像素会被当成背景。
    min_area : int
        过滤连通域时最小面积，小于这个面积的噪点会被去掉。
    debug : bool
        如果为 True，会打印一些调试信息。

    返回
    ----
    refined_mask : np.uint8, 0/1
        细化后的手部 mask。
    """
    raw_mask = (raw_mask > 0).astype(np.uint8)
    depth = depth.astype(np.float32)

    # 只看 raw_mask==1 且 深度>0 的像素
    valid = (raw_mask == 1) & (depth > 0)
    if np.count_nonzero(valid) == 0:
        if debug:
            print("[DEBUG] refine: no valid depth pixels inside raw mask, "
                  "return raw mask.")
        return raw_mask

    hand_depths = depth[valid]

    # 取比较靠近相机的一部分深度，作为“手的参考深度”
    z_near = np.percentile(hand_depths, near_percent)

    # 允许一点余量（比如 20mm）
    z_thresh = z_near + z_margin_mm

    # 比这个更远的像素当成背景（通常是物体或背景）
    depth_refined = np.zeros_like(raw_mask, dtype=np.uint8)
    depth_refined[valid & (depth <= z_thresh)] = 1

    # 连通域：只保留面积最大的那块（避免零碎粘物体）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        depth_refined, connectivity=8
    )

    if num_labels <= 1:
        refined = depth_refined
    else:
        # stats[0] 是背景，从 1 开始才是前景区域
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_idx = np.argmax(areas) + 1
        if areas[max_idx - 1] < min_area:
            # 最大的那块也太小了，就直接用 depth_refined 原图
            refined = depth_refined
        else:
            refined = np.zeros_like(depth_refined, dtype=np.uint8)
            refined[labels == max_idx] = 1

    # 适当闭运算一下，让 mask 不至于太瘦或有小洞
    kernel = np.ones((3, 3), np.uint8)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=1)

    if debug:
        print(f"[DEBUG] raw mask area: {np.count_nonzero(raw_mask)}, "
              f"refined area: {np.count_nonzero(refined)}, "
              f"z_near: {z_near:.1f} mm, z_thresh: {z_thresh:.1f} mm")

    return refined


def load_ho3d_sample(rgb_path, depth_path, meta_path=None):
    """读取一帧 HO3D 数据（RGB、深度、手 bbox），meta 支持 .pkl / .json。"""
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)

    # 1) 读 RGB / Depth
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"RGB not found: {rgb_path}")

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")

    # 2) 自动推 meta 路径（如果没显式传入）
    if meta_path is None:
        # 你的 ho3dmini 下 meta 是 .pkl：
        #   /rgb/1020.jpg -> /meta/1020.pkl
        meta_path = Path(
            str(rgb_path)
            .replace("/rgb/", "/meta/")
            .replace(".jpg", ".pkl")
            .replace(".png", ".pkl")
        )
    else:
        meta_path = Path(meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    # 3) 读取 meta（支持 .pkl / .json）
    if meta_path.suffix == ".json":
        with open(meta_path, "r") as f:
            meta = json.load(f)
    elif meta_path.suffix == ".pkl":
        with open(meta_path, "rb") as f:
            # HO3D v3 的 pkl 一般是 Python2/3 混用的，这里加 encoding 保险一点
            meta = pickle.load(f, encoding="latin1")
    else:
        raise ValueError(
            f"Unsupported meta file extension: {meta_path.suffix}"
        )

    # 4) 取 hand bbox
    if "handBoundingBox" in meta:
        x, y, w, h = meta["handBoundingBox"]
    elif "hand_bounding_box" in meta:
        x, y, w, h = meta["hand_bounding_box"]
    else:
        raise KeyError(
            f"meta file {meta_path} does not contain handBoundingBox "
            f"or hand_bounding_box key"
        )

    rect = (int(x), int(y), int(w), int(h))
    return rgb, depth, rect


def run_one(rgb_path, depth_path, out_prefix, meta_path=None, debug=False):
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    out_prefix = Path(out_prefix)

    print("[INFO] RGB  :", rgb_path)
    print("[INFO] DEPTH:", depth_path)

    if meta_path is not None:
        print("[INFO] META (explicit):", meta_path)

    rgb, depth, rect = load_ho3d_sample(rgb_path, depth_path, meta_path)

    h, w = rgb.shape[:2]
    print(f"[INFO] Image size: {w}x{h}, rect={rect}")

    # --- 1. 初始化 GrabCut ---
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(rgb, mask, rect, bgdModel, fgdModel,
                5, cv2.GC_INIT_WITH_RECT)

    # --- 2. 得到原始前景候选 mask（可能包含物体） ---
    raw_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        1, 0
    ).astype("uint8")

    # --- 3. 用深度细化 hand mask，去掉远处物体 ---
    hand_mask = refine_hand_mask_with_depth(
        raw_mask,
        depth,
        near_percent=25,   # 越小越“贴近相机”，手会更薄一点
        z_margin_mm=20,    # 允许多 20mm 的深度余量
        min_area=300,      # 面积太小就不做最大连通块过滤
        debug=debug,
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- 4. 可视化 & 保存 ---
    # 原始 GrabCut 结果
    overlay_raw = rgb.copy()
    overlay_raw[raw_mask == 1] = (0, 0, 255)
    cv2.imwrite(str(out_prefix) + "_hand_grabcut_raw_overlay.png",
                overlay_raw)

    # 深度细化后的结果
    overlay_refined = rgb.copy()
    overlay_refined[hand_mask == 1] = (0, 0, 255)
    cv2.imwrite(str(out_prefix) + "_hand_grabcut_depth_overlay.png",
                overlay_refined)

    # 方便调试：把 mask 纯图也存一下
    cv2.imwrite(str(out_prefix) + "_hand_raw_mask.png",
                raw_mask * 255)
    cv2.imwrite(str(out_prefix) + "_hand_refined_mask.png",
                hand_mask * 255)

    print("[INFO] Saved:")
    print("  ", str(out_prefix) + "_hand_grabcut_raw_overlay.png")
    print("  ", str(out_prefix) + "_hand_grabcut_depth_overlay.png")
    print("  ", str(out_prefix) + "_hand_raw_mask.png")
    print("  ", str(out_prefix) + "_hand_refined_mask.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="HO3D hand mask via GrabCut + depth refinement"
    )
    parser.add_argument("--rgb", required=True,
                        help="路径: HO3D rgb/*.jpg")
    parser.add_argument("--depth", required=True,
                        help="路径: HO3D depth/*.png")
    parser.add_argument(
        "--meta",
        required=False,
        help="路径: HO3D meta/*.pkl 或 *.json（可选，不写则由 rgb 路径推断）"
    )
    parser.add_argument("--out-prefix", required=True,
                        help="输出文件名前缀，不带扩展名")
    parser.add_argument("--debug", action="store_true",
                        help="是否打印调试信息")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_one(
        args.rgb,
        args.depth,
        args.out_prefix,
        meta_path=args.meta,
        debug=args.debug,
    )
