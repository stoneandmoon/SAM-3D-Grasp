#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_real_duck_raw_camera.py

作用：
  从 RGB-D + duck mask 直接提取真实鸭子残缺点云。
  输出点云保持在原始相机坐标系，不做居中、不做归一化、不做旋转。

输出：
  out_dir/duck_partial_raw_camera.ply
  out_dir/duck_partial_raw_camera_colored.ply
  out_dir/debug_mask_overlay.png
  out_dir/extract_info.json

典型运行：
python extract_real_duck_raw_camera.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --depth ./data/test/000002/depth/000000.png \
  --mask ./pure_duck_mask.png \
  --fx 607.0 --fy 607.0 --cx 320.0 --cy 240.0 \
  --table-plane ./output_table_axis_probe_xy/table_plane.json \
  --out-dir ./output_real_duck_raw
"""

import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def load_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 RGB: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取 RGB 失败: {path}")
    return img


def load_depth(path, depth_scale=1000.0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 depth: {path}")

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 depth 失败: {path}")

    depth = depth.astype(np.float32)

    # 常见 16-bit depth 单位是 mm
    if np.nanmax(depth) > 20:
        depth = depth / float(depth_scale)

    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_mask(path, target_hw):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 mask: {path}")

    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"读取 mask 失败: {path}")

    H, W = target_hw
    if mask.shape[:2] != (H, W):
        print(f"[Resize mask] {mask.shape[:2]} -> {(H, W)}")
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return (mask > 127).astype(np.uint8)


def project_depth(depth, fx, fy, cx, cy):
    H, W = depth.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    z = depth.astype(np.float64)
    x = (xx.astype(np.float64) - float(cx)) * z / float(fx)
    y = (yy.astype(np.float64) - float(cy)) * z / float(fy)

    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def make_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def write_pcd(path, points, colors=None):
    pcd = make_pcd(points, colors)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay_mask(rgb, mask, out_path):
    vis = rgb.copy().astype(np.float32)
    idx = mask > 0
    # 红色区域 = duck mask，注意 OpenCV 是 BGR
    vis[idx] = vis[idx] * 0.35 + np.array([0, 0, 255], dtype=np.float32) * 0.65
    cv2.imwrite(out_path, np.clip(vis, 0, 255).astype(np.uint8))
    print(f"[Save] {out_path}")


def as_float_list(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and all(isinstance(v, (int, float)) for v in x):
        return [float(v) for v in x]
    return None


def recursive_items(obj, prefix=""):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            items.append((p, v))
            items.extend(recursive_items(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            items.append((p, v))
            items.extend(recursive_items(v, p))
    return items


def load_table_plane(path):
    if path is None or path.strip() == "":
        return None, None

    if not os.path.exists(path):
        print(f"[WARN] 找不到 table plane: {path}")
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["plane_model", "plane", "coefficients"]:
        if key in data:
            arr = as_float_list(data[key])
            if arr is not None and len(arr) == 4:
                n = normalize(arr[:3])
                d = float(arr[3]) / max(np.linalg.norm(arr[:3]), 1e-12)
                return n, d

    for p, v in recursive_items(data):
        arr = as_float_list(v)
        if arr is not None and len(arr) == 4:
            lower = p.lower()
            if "plane" in lower or "table" in lower:
                n = normalize(arr[:3])
                d = float(arr[3]) / max(np.linalg.norm(arr[:3]), 1e-12)
                return n, d

    normal = None
    d = None
    for p, v in recursive_items(data):
        arr = as_float_list(v)
        if arr is not None and len(arr) == 3 and "normal" in p.lower():
            normal = arr
        if isinstance(v, (int, float)) and (p.lower().endswith(".d") or p.lower() == "d"):
            d = float(v)

    if normal is not None and d is not None:
        n = normalize(normal)
        d = d / max(np.linalg.norm(normal), 1e-12)
        return n, d

    print("[WARN] 无法解析 table plane")
    return None, None


def orient_plane_to_object_side(points, n, d):
    if n is None:
        return n, d

    h = points @ n + float(d)
    p05 = float(np.percentile(h, 5))
    p95 = float(np.percentile(h, 95))

    if abs(p05) > abs(p95):
        n = -n
        d = -d
        h = -h
        print("[Table] 法向已翻转，使鸭子位于正侧。")

    return n, d


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--table-plane", default="")
    parser.add_argument("--out-dir", default="./output_real_duck_raw")

    # 可选过滤
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--erode-mask", type=int, default=0, help="可选：腐蚀 mask 去边缘噪声，默认 0")
    parser.add_argument("--remove-outlier", action="store_true", help="可选：统计离群点过滤，不建议第一遍开启")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 从 RGB-D 直接提取真实鸭子 raw camera partial")
    print("=" * 80)

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth, args.depth_scale)

    H, W = depth.shape[:2]

    if rgb.shape[:2] != (H, W):
        print(f"[WARN] RGB size {rgb.shape[:2]} != depth size {(H, W)}")

    mask = load_mask(args.mask, (H, W))

    if args.erode_mask > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, k, iterations=int(args.erode_mask))
        print(f"[Mask] erode iterations = {args.erode_mask}")

    valid = (mask > 0) & (depth > args.min_depth) & (depth < args.max_depth)

    xyz_img = project_depth(depth, args.fx, args.fy, args.cx, args.cy)

    points = xyz_img[valid]

    # OpenCV BGR -> RGB
    rgb_as_rgb = rgb[..., ::-1]
    colors = rgb_as_rgb[valid]

    if len(points) == 0:
        raise RuntimeError("提取结果为空，请检查 mask/depth/intrinsics")

    if args.remove_outlier:
        pcd_tmp = make_pcd(points, colors)
        pcd_tmp, ind = pcd_tmp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        points = np.asarray(pcd_tmp.points)
        colors = np.asarray(pcd_tmp.colors)
        print(f"[Outlier] after filter points={len(points)}")

    out_raw = os.path.join(args.out_dir, "duck_partial_raw_camera.ply")
    out_colored = os.path.join(args.out_dir, "duck_partial_raw_camera_colored.ply")
    out_overlay = os.path.join(args.out_dir, "debug_mask_overlay.png")
    out_info = os.path.join(args.out_dir, "extract_info.json")

    write_pcd(out_raw, points, None)
    write_pcd(out_colored, points, colors)
    overlay_mask(rgb, mask, out_overlay)

    n, d = load_table_plane(args.table_plane)
    table_stats = None

    if n is not None:
        n, d = orient_plane_to_object_side(points, n, d)
        h = points @ n + float(d)
        table_stats = {
            "normal_oriented": n.tolist(),
            "d_oriented": float(d),
            "height_p01": float(np.percentile(h, 1)),
            "height_p05": float(np.percentile(h, 5)),
            "height_p50": float(np.percentile(h, 50)),
            "height_p95": float(np.percentile(h, 95)),
            "height_p99": float(np.percentile(h, 99)),
        }

        print("\n[Height to table]")
        for k, v in table_stats.items():
            print(f"  {k}: {v}")

    bbox = bbox_from_mask(mask)
    mn = points.min(axis=0)
    mx = points.max(axis=0)

    info = {
        "rgb": args.rgb,
        "depth": args.depth,
        "mask": args.mask,
        "intrinsics": {
            "fx": args.fx,
            "fy": args.fy,
            "cx": args.cx,
            "cy": args.cy,
        },
        "depth_scale": args.depth_scale,
        "image_size": [int(W), int(H)],
        "mask_bbox_xyxy": bbox,
        "num_points": int(len(points)),
        "xyz_min": mn.tolist(),
        "xyz_max": mx.tolist(),
        "xyz_extent": (mx - mn).tolist(),
        "table_stats": table_stats,
        "outputs": {
            "raw": out_raw,
            "colored": out_colored,
            "overlay": out_overlay,
        },
        "important": "This point cloud is raw camera-frame projection. It is not centered, normalized, rotated, or densified.",
    }

    with open(out_info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"[Save] {out_info}")

    print("\n✅ 完成。后续配准请优先用：")
    print(f"  {os.path.abspath(out_raw)}")
    print("=" * 80)


if __name__ == "__main__":
    main()