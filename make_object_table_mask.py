#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_object_table_mask.py

功能：
  根据：
    1. RGB 图
    2. 深度图
    3. 原始物体 mask
    4. table_plane.json
    5. 相机内参

  自动生成：
    object + local table patch 的联合 mask

用途：
  给 SAM3D 输入这个联合 mask，让它同时生成“鸭子 + 局部桌面支撑面”。

典型命令：

python make_object_table_mask.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --depth ./data/test/000002/depth/000000.png \
  --object-mask ./output_table_axis_probe_xy/object_mask.png \
  --table-plane ./output_table_axis_probe_xy/table_plane.json \
  --fx 607.0 --fy 607.0 --cx 320.0 --cy 240.0 \
  --out-dir ./output_object_table_mask

如果你的 mask 路径不是 object_mask.png，把 --object-mask 换成你 pipeline3 生成的目标 mask。
"""

import os
import json
import argparse
import numpy as np
import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 object mask: {path}")
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"读取 mask 失败: {path}")
    return (m > 127).astype(np.uint8)


def load_depth(path, depth_scale):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 depth: {path}")

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 depth 失败: {path}")

    depth = depth.astype(np.float32)

    # 常见 16-bit depth: mm -> m
    if depth.max() > 20:
        depth = depth / float(depth_scale)

    return depth


def load_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 rgb: {path}")
    rgb = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"读取 rgb 失败: {path}")
    return rgb


def load_table_plane(path):
    """
    兼容几种可能的 table_plane.json 格式。
    需要读出：
      normal: [nx, ny, nz]
      d: float
    平面形式：
      normal · x + d = 0
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 table_plane.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normal = None
    d = None

    # 常见格式 1
    if "normal" in data:
        normal = data["normal"]

    if "d" in data:
        d = data["d"]

    # 常见格式 2
    if normal is None and "plane_normal" in data:
        normal = data["plane_normal"]

    if d is None and "plane_d" in data:
        d = data["plane_d"]

    # 常见格式 3: plane_model = [a,b,c,d]
    if normal is None and "plane_model" in data:
        pm = data["plane_model"]
        normal = pm[:3]
        d = pm[3]

    # 常见格式 4: plane = [a,b,c,d]
    if normal is None and "plane" in data and isinstance(data["plane"], list):
        pm = data["plane"]
        normal = pm[:3]
        d = pm[3]

    if normal is None or d is None:
        print("[DEBUG] table_plane.json 内容：")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        raise RuntimeError("table_plane.json 中没有找到 normal/d 或 plane_model")

    normal = np.asarray(normal, dtype=np.float64)
    d = float(d)

    norm = np.linalg.norm(normal) + 1e-12
    normal = normal / norm
    d = d / norm

    return normal, d


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("object mask 是空的")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1


def expand_bbox(x0, y0, x1, y1, w, h, pad):
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    return x0, y0, x1, y1


def make_local_table_mask(
    depth,
    object_mask,
    normal,
    d,
    fx,
    fy,
    cx,
    cy,
    bbox_pad=60,
    plane_thresh=0.012,
    bottom_bias=True,
):
    """
    生成局部桌面 mask。

    逻辑：
      1. 用 depth + intrinsics 把像素反投影到相机 3D
      2. 用 table plane 筛选属于桌面的像素
      3. 只保留 object bbox 周围的一小块区域
      4. 排除 object mask 本身
      5. 可选：更偏向物体底部附近的桌面
    """
    h, w = depth.shape[:2]

    x0, y0, x1, y1 = bbox_from_mask(object_mask)
    ex0, ey0, ex1, ey1 = expand_bbox(x0, y0, x1, y1, w, h, bbox_pad)

    valid = depth > 1e-6

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    z = depth
    x = (xx.astype(np.float32) - cx) * z / fx
    y = (yy.astype(np.float32) - cy) * z / fy

    # plane distance in camera frame
    dist = normal[0] * x + normal[1] * y + normal[2] * z + d

    table_mask = (np.abs(dist) < plane_thresh) & valid

    # 只保留物体周围局部区域
    local = np.zeros_like(table_mask, dtype=bool)
    local[ey0:ey1 + 1, ex0:ex1 + 1] = True

    table_mask = table_mask & local

    # 不要把物体自身也当成桌面
    table_mask = table_mask & (object_mask == 0)

    if bottom_bias:
        # 进一步偏向物体下半部分附近的支撑区域
        # 但不完全只取下方，因为视角下桌面可能在物体四周可见
        obj_h = y1 - y0 + 1
        y_bottom_start = int(y0 + obj_h * 0.35)

        bottom_region = np.zeros_like(table_mask, dtype=bool)
        by0 = max(0, y_bottom_start - bbox_pad // 3)
        by1 = min(h - 1, y1 + bbox_pad)
        bottom_region[by0:by1 + 1, ex0:ex1 + 1] = True

        table_mask = table_mask & bottom_region

    # 形态学清理
    table_mask_u8 = table_mask.astype(np.uint8) * 255

    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)

    table_mask_u8 = cv2.morphologyEx(table_mask_u8, cv2.MORPH_OPEN, kernel3)
    table_mask_u8 = cv2.morphologyEx(table_mask_u8, cv2.MORPH_CLOSE, kernel5)

    table_mask = (table_mask_u8 > 127).astype(np.uint8)

    return table_mask


def overlay_debug(rgb, object_mask, table_mask, alpha=0.45):
    """
    object: 红色
    table : 绿色
    combined overlap: 黄色
    """
    out = rgb.copy()

    red = np.zeros_like(out)
    red[:, :, 2] = 255

    green = np.zeros_like(out)
    green[:, :, 1] = 255

    obj = object_mask > 0
    tab = table_mask > 0

    out[obj] = (out[obj] * (1 - alpha) + red[obj] * alpha).astype(np.uint8)
    out[tab] = (out[tab] * (1 - alpha) + green[tab] * alpha).astype(np.uint8)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--object-mask", required=True)
    parser.add_argument("--table-plane", required=True)

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--bbox-pad", type=int, default=60)
    parser.add_argument("--plane-thresh", type=float, default=0.012)

    parser.add_argument("--out-dir", default="./output_object_table_mask")
    parser.add_argument("--no-bottom-bias", action="store_true")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth, args.depth_scale)
    object_mask = load_mask(args.object_mask)
    normal, d = load_table_plane(args.table_plane)

    h, w = depth.shape[:2]
    if object_mask.shape[:2] != (h, w):
        raise RuntimeError(
            f"mask 尺寸 {object_mask.shape[:2]} 和 depth 尺寸 {(h, w)} 不一致，"
            f"请确认 object mask 是同一张图上的 mask"
        )

    print("=" * 80)
    print("[Start] 生成 object + local table patch mask")
    print("=" * 80)
    print(f"[RGB]          {args.rgb}")
    print(f"[Depth]        {args.depth}")
    print(f"[Object mask]  {args.object_mask}")
    print(f"[Table plane]  {args.table_plane}")
    print(f"[Plane] normal={normal.tolist()}, d={d:.8f}")
    print(f"[Intrinsics] fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")

    table_mask = make_local_table_mask(
        depth=depth,
        object_mask=object_mask,
        normal=normal,
        d=d,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        bbox_pad=args.bbox_pad,
        plane_thresh=args.plane_thresh,
        bottom_bias=(not args.no_bottom_bias),
    )

    combined = ((object_mask > 0) | (table_mask > 0)).astype(np.uint8)

    obj_pixels = int(object_mask.sum())
    table_pixels = int(table_mask.sum())
    combined_pixels = int(combined.sum())

    print("\n[Mask stats]")
    print(f"  object pixels   = {obj_pixels}")
    print(f"  table pixels    = {table_pixels}")
    print(f"  combined pixels = {combined_pixels}")
    if obj_pixels > 0:
        print(f"  table/object ratio = {table_pixels / obj_pixels:.3f}")

    out_object = os.path.join(args.out_dir, "object_mask_input.png")
    out_table = os.path.join(args.out_dir, "local_table_patch_mask.png")
    out_combined = os.path.join(args.out_dir, "object_plus_local_table_mask.png")
    out_overlay = os.path.join(args.out_dir, "debug_overlay_object_red_table_green.png")

    cv2.imwrite(out_object, object_mask * 255)
    cv2.imwrite(out_table, table_mask * 255)
    cv2.imwrite(out_combined, combined * 255)

    overlay = overlay_debug(rgb, object_mask, table_mask)
    cv2.imwrite(out_overlay, overlay)

    print("\n[Save]")
    print(f"  原始物体 mask        : {os.path.abspath(out_object)}")
    print(f"  局部桌面 patch mask  : {os.path.abspath(out_table)}")
    print(f"  联合 mask            : {os.path.abspath(out_combined)}")
    print(f"  debug overlay        : {os.path.abspath(out_overlay)}")

    print("\n颜色说明 debug_overlay：")
    print("  红色 = 原始鸭子 mask")
    print("  绿色 = 局部桌面 patch")
    print("  最终给 SAM3D 的是 object_plus_local_table_mask.png")

    print("\n✅ 完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
