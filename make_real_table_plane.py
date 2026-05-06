#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_real_table_plane.py

作用：
  从真实 RGB-D 中重新拟合桌面平面，生成：
    output_table_axis_probe_xy/table_plane.json
    output_table_axis_probe_xy/table_points.ply
    output_table_axis_probe_xy/non_table_points.ply
    output_table_axis_probe_xy/debug_info.json

平面形式：
  normal · x + d = 0

运行示例：
python make_real_table_plane.py \
  --rgb ./data/test/000002/rgb/000000.png \
  --depth ./data/test/000002/depth/000000.png \
  --duck-mask ./pure_duck_mask.png \
  --fx 607.0 \
  --fy 607.0 \
  --cx 320.0 \
  --cy 240.0 \
  --out-dir ./output_table_axis_probe_xy
"""

import os
import json
import argparse
import numpy as np
import cv2
import open3d as o3d


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_depth(path, depth_scale=1000.0):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 depth: {path}")

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"读取 depth 失败: {path}")

    depth = depth.astype(np.float32)

    # 常见 16-bit depth 是 mm
    if np.nanmax(depth) > 20:
        depth = depth / float(depth_scale)

    depth[~np.isfinite(depth)] = 0.0
    return depth


def load_rgb(path):
    if path is None or path.strip() == "":
        return None

    if not os.path.exists(path):
        print(f"[WARN] 找不到 RGB: {path}，跳过 overlay。")
        return None

    rgb = cv2.imread(path, cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"[WARN] 读取 RGB 失败: {path}，跳过 overlay。")
        return None

    return rgb


def load_mask(path, shape_hw):
    if path is None or path.strip() == "":
        return None

    if not os.path.exists(path):
        print(f"[WARN] 找不到 duck mask: {path}，将不排除鸭子区域。")
        return None

    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        print(f"[WARN] 读取 duck mask 失败: {path}，将不排除鸭子区域。")
        return None

    H, W = shape_hw
    if m.shape[:2] != (H, W):
        print(f"[Resize mask] {m.shape[:2]} -> {(H, W)}")
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    return (m > 127).astype(np.uint8)


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def project_depth_to_points(depth, fx, fy, cx, cy):
    H, W = depth.shape[:2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    z = depth.astype(np.float64)
    x = (xx.astype(np.float64) - float(cx)) * z / float(fx)
    y = (yy.astype(np.float64) - float(cy)) * z / float(fy)

    pts = np.stack([x, y, z], axis=-1)
    return pts


def make_candidate_mask(depth, duck_mask, bbox_pad, use_roi=True):
    H, W = depth.shape[:2]
    valid = depth > 1e-6

    candidate = valid.copy()

    if duck_mask is not None:
        # 排除鸭子本体，避免 RANSAC 拟合到鸭子表面
        kernel = np.ones((9, 9), np.uint8)
        duck_dilate = cv2.dilate((duck_mask > 0).astype(np.uint8), kernel, iterations=1) > 0
        candidate = candidate & (~duck_dilate)

        if use_roi:
            box = bbox_from_mask(duck_mask)
            if box is not None:
                x0, y0, x1, y1 = box
                x0 = max(0, x0 - bbox_pad)
                y0 = max(0, y0 - bbox_pad)
                x1 = min(W - 1, x1 + bbox_pad)
                y1 = min(H - 1, y1 + bbox_pad)

                roi = np.zeros((H, W), dtype=bool)
                roi[y0:y1 + 1, x0:x1 + 1] = True
                candidate = candidate & roi

                print(f"[ROI] 使用 duck bbox 周围区域: x[{x0},{x1}], y[{y0},{y1}]")

    return candidate


def pcd_from_points(points, color=None):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None and len(points) > 0:
        colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def normalize_plane(model):
    a, b, c, d = [float(v) for v in model]
    n = np.asarray([a, b, c], dtype=np.float64)
    norm = float(np.linalg.norm(n))
    if norm < 1e-12:
        raise RuntimeError("RANSAC 得到的 plane normal 长度为 0")
    n = n / norm
    d = d / norm
    return n, float(d)


def orient_plane_to_duck_side(normal, d, all_points, duck_points=None):
    """
    法向有正负歧义。
    如果有 duck_points，让鸭子位于 signed distance 正侧。
    否则让大部分非平面点在正侧。
    """
    normal = np.asarray(normal, dtype=np.float64)
    d = float(d)

    if duck_points is not None and len(duck_points) > 20:
        h = duck_points @ normal + d
        med = float(np.median(h))
        if med < 0:
            normal = -normal
            d = -d
            print("[Orient] 已翻转法向：让鸭子位于桌面正侧。")
        else:
            print("[Orient] 法向保持：鸭子已位于桌面正侧。")
        return normal, d

    h = all_points @ normal + d
    p01 = float(np.percentile(h, 1))
    p99 = float(np.percentile(h, 99))
    if abs(p01) > abs(p99):
        normal = -normal
        d = -d
        print("[Orient] 已翻转法向：让主体点云位于正侧。")
    else:
        print("[Orient] 法向保持。")

    return normal, d


def save_overlay(rgb, candidate_mask, table_pixel_mask, duck_mask, out_path):
    if rgb is None:
        return

    vis = rgb.copy().astype(np.float32)

    # candidate 蓝色
    idx = candidate_mask > 0
    vis[idx] = vis[idx] * 0.5 + np.array([255, 120, 0], dtype=np.float32) * 0.5

    # table 绿色
    idx = table_pixel_mask > 0
    vis[idx] = vis[idx] * 0.35 + np.array([0, 255, 0], dtype=np.float32) * 0.65

    # duck 红色
    if duck_mask is not None:
        idx = duck_mask > 0
        vis[idx] = vis[idx] * 0.35 + np.array([0, 0, 255], dtype=np.float32) * 0.65

    cv2.imwrite(out_path, np.clip(vis, 0, 255).astype(np.uint8))
    print(f"[Save] {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rgb", default="", help="RGB 图像路径，可选，用于 debug overlay")
    parser.add_argument("--depth", required=True, help="depth 图像路径")
    parser.add_argument("--duck-mask", default="", help="鸭子 mask，用于排除鸭子和确定 ROI")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--depth-scale", type=float, default=1000.0)

    parser.add_argument("--bbox-pad", type=int, default=120, help="围绕鸭子 bbox 扩展多少像素找桌面")
    parser.add_argument("--no-roi", action="store_true", help="不用鸭子 bbox ROI，直接在整张图里拟合最大平面")
    parser.add_argument("--ransac-dist", type=float, default=0.010, help="RANSAC 平面距离阈值，单位米")
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3000)

    parser.add_argument("--min-candidate-points", type=int, default=500)
    parser.add_argument("--out-dir", default="./output_table_axis_probe_xy")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 重新拟合真实 RGB-D 桌面平面")
    print("=" * 80)

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    rgb = load_rgb(args.rgb)

    H, W = depth.shape[:2]
    duck_mask = load_mask(args.duck_mask, (H, W))

    xyz_img = project_depth_to_points(depth, args.fx, args.fy, args.cx, args.cy)

    candidate_mask = make_candidate_mask(
        depth=depth,
        duck_mask=duck_mask,
        bbox_pad=args.bbox_pad,
        use_roi=(not args.no_roi),
    )

    candidate_points = xyz_img[candidate_mask]
    all_valid_points = xyz_img[depth > 1e-6]

    if duck_mask is not None:
        duck_valid = (duck_mask > 0) & (depth > 1e-6)
        duck_points = xyz_img[duck_valid]
    else:
        duck_points = None

    print(f"[Depth] size={W}x{H}")
    print(f"[Points] all_valid={len(all_valid_points)}")
    print(f"[Points] candidate_for_table={len(candidate_points)}")

    if len(candidate_points) < args.min_candidate_points:
        raise RuntimeError(
            f"候选桌面点太少: {len(candidate_points)}。"
            f"可以尝试：增大 --bbox-pad，或者加 --no-roi。"
        )

    pcd_candidate = pcd_from_points(candidate_points)

    print("\n[RANSAC] 拟合桌面平面...")
    plane_model, inliers = pcd_candidate.segment_plane(
        distance_threshold=float(args.ransac_dist),
        ransac_n=int(args.ransac_n),
        num_iterations=int(args.iters),
    )

    inliers = np.asarray(inliers, dtype=np.int64)
    n, d = normalize_plane(plane_model)

    n, d = orient_plane_to_duck_side(
        normal=n,
        d=d,
        all_points=all_valid_points,
        duck_points=duck_points,
    )

    # 在整张 depth 上重新计算哪些点属于桌面
    signed = xyz_img.reshape(-1, 3) @ n + d
    signed = signed.reshape(H, W)

    valid = depth > 1e-6
    table_pixel_mask = (np.abs(signed) < float(args.ransac_dist)) & valid

    # 如果有 duck mask，排除鸭子
    if duck_mask is not None:
        table_pixel_mask = table_pixel_mask & (duck_mask == 0)

    table_points = xyz_img[table_pixel_mask]
    non_table_points = xyz_img[valid & (~table_pixel_mask)]

    print("\n[Plane Result]")
    print(f"  normal = {n.tolist()}")
    print(f"  d      = {d:.8f}")
    print(f"  plane  = normal · x + d = 0")
    print(f"  candidate inliers = {len(inliers)} / {len(candidate_points)}")
    print(f"  full table points = {len(table_points)}")

    # 保存点云
    table_ply = os.path.join(args.out_dir, "table_points.ply")
    non_table_ply = os.path.join(args.out_dir, "non_table_points.ply")
    candidate_ply = os.path.join(args.out_dir, "candidate_points_for_ransac.ply")
    plane_json = os.path.join(args.out_dir, "table_plane.json")
    debug_json = os.path.join(args.out_dir, "debug_info.json")
    overlay_path = os.path.join(args.out_dir, "debug_overlay.png")

    o3d.io.write_point_cloud(table_ply, pcd_from_points(table_points, color=[0.1, 0.9, 0.2]))
    o3d.io.write_point_cloud(non_table_ply, pcd_from_points(non_table_points, color=[0.8, 0.8, 0.8]))
    o3d.io.write_point_cloud(candidate_ply, pcd_from_points(candidate_points, color=[0.1, 0.3, 1.0]))

    print(f"[Save] {table_ply}")
    print(f"[Save] {non_table_ply}")
    print(f"[Save] {candidate_ply}")

    plane_info = {
        "plane_form": "normal dot x + d = 0",
        "normal": n.tolist(),
        "d": float(d),
        "plane_model": [float(n[0]), float(n[1]), float(n[2]), float(d)],
        "source": {
            "rgb": args.rgb,
            "depth": args.depth,
            "duck_mask": args.duck_mask,
        },
        "intrinsics": {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
        },
        "params": {
            "depth_scale": float(args.depth_scale),
            "bbox_pad": int(args.bbox_pad),
            "use_roi": not args.no_roi,
            "ransac_dist": float(args.ransac_dist),
            "ransac_n": int(args.ransac_n),
            "iters": int(args.iters),
        },
        "stats": {
            "all_valid_points": int(len(all_valid_points)),
            "candidate_points": int(len(candidate_points)),
            "candidate_inliers": int(len(inliers)),
            "table_points": int(len(table_points)),
            "non_table_points": int(len(non_table_points)),
        },
        "outputs": {
            "table_points": table_ply,
            "non_table_points": non_table_ply,
            "candidate_points_for_ransac": candidate_ply,
            "debug_overlay": overlay_path,
        },
    }

    with open(plane_json, "w", encoding="utf-8") as f:
        json.dump(plane_info, f, indent=2, ensure_ascii=False)

    with open(debug_json, "w", encoding="utf-8") as f:
        json.dump(plane_info, f, indent=2, ensure_ascii=False)

    print(f"[Save] {plane_json}")
    print(f"[Save] {debug_json}")

    save_overlay(
        rgb=rgb,
        candidate_mask=candidate_mask.astype(np.uint8),
        table_pixel_mask=table_pixel_mask.astype(np.uint8),
        duck_mask=duck_mask,
        out_path=overlay_path,
    )

    print("\n✅ 完成。后续配准用这个参数：")
    print(f"  --real-table-plane {os.path.abspath(plane_json)}")


if __name__ == "__main__":
    main()