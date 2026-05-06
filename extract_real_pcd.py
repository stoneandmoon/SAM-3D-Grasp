#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPEN3D_CPU_RENDERING"] = "true"

import argparse
import faulthandler
faulthandler.enable()

import cv2
import numpy as np
import open3d as o3d


def keep_largest_component(mask_u8):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    out = np.zeros_like(mask_u8)
    out[labels == largest_label] = 1
    return out


def refine_mask(mask_gray, erode_iter=1):
    mask = (mask_gray > 127).astype(np.uint8)
    mask = keep_largest_component(mask)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)

    if erode_iter > 0:
        mask = cv2.erode(mask, k3, iterations=erode_iter)

    return mask.astype(bool)


def depth_consistency_filter(
    depth_mm,
    obj_mask,
    median_ksize=5,
    abs_dev_mm=20.0,
    rel_dev=0.02,
    edge_thresh_mm=35.0,
    min_depth_mm=50,
    max_depth_mm=5000
):
    depth_mm = depth_mm.copy()
    valid = (depth_mm >= min_depth_mm) & (depth_mm <= max_depth_mm) & obj_mask

    med = cv2.medianBlur(depth_mm, median_ksize)

    diff = np.abs(depth_mm.astype(np.int32) - med.astype(np.int32)).astype(np.float32)
    adaptive_thresh = np.maximum(abs_dev_mm, med.astype(np.float32) * rel_dev)

    depth_f = depth_mm.astype(np.float32)
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    good = valid & (diff <= adaptive_thresh) & (grad <= edge_thresh_mm)

    depth_clean = np.zeros_like(depth_mm)
    depth_clean[good] = depth_mm[good]

    return depth_clean, good, med, grad


def fill_small_holes(depth_mm, obj_mask, max_iters=2, min_valid_neighbors=5):
    d = depth_mm.astype(np.float32).copy()
    valid = d > 0
    kernel = np.ones((3, 3), np.float32)

    for _ in range(max_iters):
        holes = obj_mask & (~valid)
        if not np.any(holes):
            break

        sum_neighbors = cv2.filter2D(d, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        cnt_neighbors = cv2.filter2D(valid.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)

        fillable = holes & (cnt_neighbors >= float(min_valid_neighbors))
        if not np.any(fillable):
            break

        d[fillable] = sum_neighbors[fillable] / cnt_neighbors[fillable]
        valid = d > 0

    return d.astype(np.uint16)


def uvz_to_xyz(u, v, z_m, intrinsics):
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x = (u - float(cx)) * z_m / float(fx)
    y = (v - float(cy)) * z_m / float(fy)
    pts = np.stack([x, y, z_m], axis=1)
    return pts


def build_original_points(rgb, depth_mm, intrinsics, depth_scale, valid_mask):
    valid = valid_mask & (depth_mm > 0)
    v, u = np.where(valid)

    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    z_m = depth_mm[v, u].astype(np.float64) * float(depth_scale)
    pts = uvz_to_xyz(u.astype(np.float64), v.astype(np.float64), z_m, intrinsics)
    cols = rgb[v, u].astype(np.float64) / 255.0
    return pts, cols


def build_interpolated_points(
    rgb,
    depth_mm,
    intrinsics,
    depth_scale,
    valid_mask,
    interp_edge_diff_mm=20.0,
    interp_center_span_mm=30.0,
    enable_h=True,
    enable_v=True,
    enable_center=True
):
    """
    局部受限插值：
    1) 横向相邻像素深度接近 -> 补边中点
    2) 纵向相邻像素深度接近 -> 补边中点
    3) 2x2 小块都有效且深度跨度不大 -> 补中心点

    这不是整图 resize，而是只在局部平滑区域加点。
    """
    d = depth_mm.astype(np.float64)
    valid = valid_mask & (depth_mm > 0)

    pts_list = []
    col_list = []

    # ---------- 横向边中点 ----------
    if enable_h:
        vh = valid[:, :-1] & valid[:, 1:]
        vh &= (np.abs(d[:, :-1] - d[:, 1:]) <= float(interp_edge_diff_mm))

        yy, xx = np.where(vh)
        if len(xx) > 0:
            u_h = xx.astype(np.float64) + 0.5
            v_h = yy.astype(np.float64)
            z_h = 0.5 * (d[yy, xx] + d[yy, xx + 1]) * float(depth_scale)
            pts_h = uvz_to_xyz(u_h, v_h, z_h, intrinsics)

            col_h = 0.5 * (
                rgb[yy, xx].astype(np.float64) +
                rgb[yy, xx + 1].astype(np.float64)
            ) / 255.0

            pts_list.append(pts_h)
            col_list.append(col_h)
            print(f"[DEBUG] interp horizontal points: {len(pts_h)}", flush=True)

    # ---------- 纵向边中点 ----------
    if enable_v:
        vv = valid[:-1, :] & valid[1:, :]
        vv &= (np.abs(d[:-1, :] - d[1:, :]) <= float(interp_edge_diff_mm))

        yy, xx = np.where(vv)
        if len(xx) > 0:
            u_v = xx.astype(np.float64)
            v_v = yy.astype(np.float64) + 0.5
            z_v = 0.5 * (d[yy, xx] + d[yy + 1, xx]) * float(depth_scale)
            pts_v = uvz_to_xyz(u_v, v_v, z_v, intrinsics)

            col_v = 0.5 * (
                rgb[yy, xx].astype(np.float64) +
                rgb[yy + 1, xx].astype(np.float64)
            ) / 255.0

            pts_list.append(pts_v)
            col_list.append(col_v)
            print(f"[DEBUG] interp vertical points: {len(pts_v)}", flush=True)

    # ---------- 2x2 小块中心点 ----------
    if enable_center:
        v00 = valid[:-1, :-1]
        v01 = valid[:-1, 1:]
        v10 = valid[1:, :-1]
        v11 = valid[1:, 1:]

        cell = v00 & v01 & v10 & v11

        d00 = d[:-1, :-1]
        d01 = d[:-1, 1:]
        d10 = d[1:, :-1]
        d11 = d[1:, 1:]

        dmax = np.maximum.reduce([d00, d01, d10, d11])
        dmin = np.minimum.reduce([d00, d01, d10, d11])
        span = dmax - dmin

        diag1 = np.abs(d00 - d11)
        diag2 = np.abs(d01 - d10)

        cell &= (span <= float(interp_center_span_mm))
        cell &= (diag1 <= float(interp_center_span_mm))
        cell &= (diag2 <= float(interp_center_span_mm))

        yy, xx = np.where(cell)
        if len(xx) > 0:
            u_c = xx.astype(np.float64) + 0.5
            v_c = yy.astype(np.float64) + 0.5
            z_c = 0.25 * (
                d[yy, xx] +
                d[yy, xx + 1] +
                d[yy + 1, xx] +
                d[yy + 1, xx + 1]
            ) * float(depth_scale)

            pts_c = uvz_to_xyz(u_c, v_c, z_c, intrinsics)

            col_c = 0.25 * (
                rgb[yy, xx].astype(np.float64) +
                rgb[yy, xx + 1].astype(np.float64) +
                rgb[yy + 1, xx].astype(np.float64) +
                rgb[yy + 1, xx + 1].astype(np.float64)
            ) / 255.0

            pts_list.append(pts_c)
            col_list.append(col_c)
            print(f"[DEBUG] interp center points: {len(pts_c)}", flush=True)

    if len(pts_list) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    pts_interp = np.concatenate(pts_list, axis=0)
    col_interp = np.concatenate(col_list, axis=0)

    return pts_interp, col_interp


def backproject_to_pointcloud(
    rgb,
    depth_mm,
    intrinsics,
    depth_scale,
    valid_mask,
    enable_interp=True,
    interp_edge_diff_mm=20.0,
    interp_center_span_mm=30.0
):
    pts_ori, col_ori = build_original_points(
        rgb=rgb,
        depth_mm=depth_mm,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        valid_mask=valid_mask
    )

    print(f"[DEBUG] original points: {len(pts_ori)}", flush=True)

    if enable_interp:
        pts_itp, col_itp = build_interpolated_points(
            rgb=rgb,
            depth_mm=depth_mm,
            intrinsics=intrinsics,
            depth_scale=depth_scale,
            valid_mask=valid_mask,
            interp_edge_diff_mm=interp_edge_diff_mm,
            interp_center_span_mm=interp_center_span_mm,
            enable_h=True,
            enable_v=True,
            enable_center=True
        )
        print(f"[DEBUG] interpolated points: {len(pts_itp)}", flush=True)

        points = np.concatenate([pts_ori, pts_itp], axis=0)
        colors = np.concatenate([col_ori, col_itp], axis=0)
    else:
        points = pts_ori
        colors = col_ori

    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    points = points[finite_mask]
    colors = colors[finite_mask]

    print(f"[DEBUG] total points shape = {points.shape}, dtype = {points.dtype}", flush=True)
    print(f"[DEBUG] total colors shape = {colors.shape}, dtype = {colors.dtype}", flush=True)

    if len(points) == 0:
        return o3d.geometry.PointCloud()

    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def clean_pointcloud(
    pcd,
    voxel_size=0.0012,
    radius_nb_points=8,
    radius_multiplier=2.2,
    stat_nb_neighbors=16,
    stat_std_ratio=1.2,
):
    if len(pcd.points) == 0:
        return pcd

    print("[DEBUG] start voxel_down_sample", flush=True)
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    if len(pcd.points) == 0:
        return pcd

    radius = max(voxel_size * radius_multiplier, 0.0035)

    print(f"[DEBUG] start remove_radius_outlier, radius={radius:.6f}", flush=True)
    _, ind = pcd.remove_radius_outlier(nb_points=radius_nb_points, radius=radius)
    pcd = pcd.select_by_index(ind)

    if len(pcd.points) == 0:
        return pcd

    print("[DEBUG] start remove_statistical_outlier", flush=True)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=stat_nb_neighbors,
        std_ratio=stat_std_ratio
    )
    pcd = pcd.select_by_index(ind)

    return pcd


def generate_clean_pointcloud(
    rgb_path,
    depth_path,
    mask_path,
    out_file,
    intrinsics,
    depth_scale=0.001,
    mask_erode_iter=1,
    median_ksize=5,
    abs_dev_mm=20.0,
    rel_dev=0.02,
    edge_thresh_mm=35.0,
    voxel_size=0.0012,
    fill_holes=True,
    enable_interp=True,
    interp_edge_diff_mm=20.0,
    interp_center_span_mm=30.0
):
    print("🚀 开始提取更稳健的残缺点云...", flush=True)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if rgb is None or depth is None or mask is None:
        raise FileNotFoundError("❌ 图像加载失败，请检查 rgb/depth/mask 路径。")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)

    print(f"RGB shape   : {rgb.shape}", flush=True)
    print(f"Depth shape : {depth.shape}, dtype={depth.dtype}", flush=True)
    print(f"Mask shape  : {mask.shape}", flush=True)

    mask_refined = refine_mask(mask, erode_iter=mask_erode_iter)

    depth_clean, valid_2d, med, grad = depth_consistency_filter(
        depth_mm=depth,
        obj_mask=mask_refined,
        median_ksize=median_ksize,
        abs_dev_mm=abs_dev_mm,
        rel_dev=rel_dev,
        edge_thresh_mm=edge_thresh_mm
    )

    if fill_holes:
        depth_filled = fill_small_holes(depth_clean, mask_refined, max_iters=2, min_valid_neighbors=5)
    else:
        depth_filled = depth_clean.copy()

    valid_final = mask_refined & (depth_filled > 0)

    print(f"🟡 mask 内像素数: {int(mask_refined.sum())}", flush=True)
    print(f"🟡 初筛后有效深度像素数: {int((depth_clean > 0).sum())}", flush=True)
    print(f"🟢 补小洞后有效深度像素数: {int(valid_final.sum())}", flush=True)

    raw_pcd = backproject_to_pointcloud(
        rgb=rgb,
        depth_mm=depth_filled,
        intrinsics=intrinsics,
        depth_scale=depth_scale,
        valid_mask=valid_final,
        enable_interp=enable_interp,
        interp_edge_diff_mm=interp_edge_diff_mm,
        interp_center_span_mm=interp_center_span_mm
    )
    print(f"💡 回投影后原始/插值点数: {len(raw_pcd.points)}", flush=True)

    clean_pcd = clean_pointcloud(
        raw_pcd,
        voxel_size=voxel_size,
        radius_nb_points=8,
        radius_multiplier=2.2,
        stat_nb_neighbors=16,
        stat_std_ratio=1.2,
    )
    print(f"✨ 清理后点数: {len(clean_pcd.points)}", flush=True)

    raw_out = out_file.replace(".ply", "_raw_interp.ply")
    clean_out = out_file.replace(".ply", "_interp_clean.ply")

    o3d.io.write_point_cloud(raw_out, raw_pcd)
    o3d.io.write_point_cloud(clean_out, clean_pcd)

    print(f"📁 插值后的原始点云已保存: {raw_out}", flush=True)
    print(f"📁 插值+清理后的点云已保存: {clean_out}", flush=True)
    print("✅ 已跳过可视化，避免服务器环境下 Open3D 崩溃", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="根据 pipeline 传入的 RGB / depth / mask 提取真实残缺点云"
    )

    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="深度图路径")
    parser.add_argument("--mask", required=True, help="pipeline 分割得到的目标 mask 路径")
    parser.add_argument("--out", required=True, help="输出点云路径，例如 output_real_partial_from_depth/real_partial_from_depth.ply")

    parser.add_argument("--fx", type=float, default=572.4114)
    parser.add_argument("--fy", type=float, default=573.2265)
    parser.add_argument("--cx", type=float, default=325.2611)
    parser.add_argument("--cy", type=float, default=242.0489)
    parser.add_argument("--depth-scale", type=float, default=0.001)

    args = parser.parse_args()

    CAMERA_INTRINSICS = {
        "fx": args.fx,
        "fy": args.fy,
        "cx": args.cx,
        "cy": args.cy,
    }

    generate_clean_pointcloud(
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_path=args.mask,
        out_file=args.out,
        intrinsics=CAMERA_INTRINSICS,
        depth_scale=args.depth_scale,

        # 原来这部分完全不动
        mask_erode_iter=1,
        median_ksize=5,
        abs_dev_mm=20.0,
        rel_dev=0.02,
        edge_thresh_mm=35.0,
        fill_holes=True,

        # 原来这部分完全不动
        voxel_size=0.0012,

        # 原来这部分完全不动
        enable_interp=True,
        interp_edge_diff_mm=20.0,
        interp_center_span_mm=30.0
    )