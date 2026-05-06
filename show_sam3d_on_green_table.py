#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
show_sam3d_on_green_table.py

目标：
  把 SAM3D 生成的完整点云放到已经恢复为 XY 平面的绿色桌面上显示。

典型运行：
python show_sam3d_on_green_table.py \
  --sam3d ./output_3d/reconstructed_mesh.ply \
  --table ./output_table_axis_probe_xy/table_plane_inliers_table_xy.ply \
  --partial ./output_table_axis_probe_xy/object_points_from_mask_table_xy.ply \
  --out-dir ./output_sam3d_on_green_table \
  --show

如果你要看鸭子：
python show_sam3d_on_green_table.py \
  --sam3d ./sam3d_duck_clean.ply \
  --table ./output_table_axis_probe_xy/table_plane_inliers_table_xy.ply \
  --partial ./output_table_axis_probe_xy/object_points_from_mask_table_xy.ply \
  --out-dir ./output_duck_on_green_table \
  --show
"""

import os
import argparse
import itertools
import numpy as np
import open3d as o3d


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pcd(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] 找不到 {name}: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"[ERROR] {name} 点云为空: {path}")
    print(f"[Load] {name}: {path}, points={len(pcd.points)}")
    return pcd


def np_points(pcd):
    return np.asarray(pcd.points).astype(np.float64)


def make_pcd(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def remove_outliers_and_keep_main_cluster(
    pcd,
    voxel_size=0.003,
    nb_neighbors=20,
    std_ratio=2.0,
    dbscan_eps=0.018,
    dbscan_min_points=20,
):
    """
    清理 SAM3D 点云：
    1. voxel downsample
    2. statistical outlier removal
    3. DBSCAN 只保留最大主体
    """
    print("\n[Clean] 清理 SAM3D 点云浮空碎片...")

    before = len(pcd.points)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"  voxel_down_sample: {before} -> {len(pcd.points)}")
    else:
        print(f"  skip voxel_down_sample, points={len(pcd.points)}")

    if len(pcd.points) > nb_neighbors:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        print(f"  statistical_outlier_removal -> {len(pcd.points)}")

    if len(pcd.points) == 0:
        raise RuntimeError("[ERROR] 清理后 SAM3D 点云为空，请调小清理强度。")

    labels = np.array(
        pcd.cluster_dbscan(
            eps=dbscan_eps,
            min_points=dbscan_min_points,
            print_progress=False,
        )
    )

    if labels.size == 0 or labels.max() < 0:
        print("  [WARN] DBSCAN 没找到有效簇，保留当前点云。")
        return pcd

    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)
    main_label = unique[np.argmax(counts)]
    keep = np.where(labels == main_label)[0]

    cleaned = pcd.select_by_index(keep)
    print(f"  dbscan keep main cluster label={main_label}, points={len(cleaned.points)}")

    return cleaned


def generate_axis_rotations():
    """
    生成 24 个正交轴向旋转矩阵。
    作用：
      SAM3D 点云可能 canonical 轴不稳定，
      这里枚举 xyz 轴排列和正负方向，找一个最适合放到桌面的姿态。
    """
    mats = []
    axes = np.eye(3)

    for perm in itertools.permutations([0, 1, 2]):
        P = axes[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            R = P @ np.diag(signs)
            if np.linalg.det(R) > 0.5:
                mats.append(R)

    # 去重
    unique = []
    for R in mats:
        if not any(np.allclose(R, Q) for Q in unique):
            unique.append(R)

    return unique


def robust_bbox_xy(points, q_low=2, q_high=98):
    lo = np.percentile(points[:, :2], q_low, axis=0)
    hi = np.percentile(points[:, :2], q_high, axis=0)
    return lo, hi, hi - lo


def robust_height(points, q_low=1, q_high=99):
    zlo = np.percentile(points[:, 2], q_low)
    zhi = np.percentile(points[:, 2], q_high)
    return zlo, zhi, zhi - zlo


def evaluate_candidate(sam_pts_rot, partial_pts, scale):
    """
    候选姿态打分。
    越小越好。

    主要约束：
    1. 放大/缩小后底部贴近桌面 z=0
    2. 不允许大量点在桌面以下
    3. 高度不要过分离谱
    4. XY 尺寸要接近红色 partial
    """
    pts = sam_pts_rot.copy() * scale

    z_bottom = np.percentile(pts[:, 2], 1)
    pts[:, 2] -= z_bottom

    below_ratio = np.mean(pts[:, 2] < -0.003)
    zlo, zhi, h = robust_height(pts)

    _, _, sam_xy_size = robust_bbox_xy(pts)
    _, _, part_xy_size = robust_bbox_xy(partial_pts)

    xy_size_err = np.linalg.norm(sam_xy_size - part_xy_size) / (np.linalg.norm(part_xy_size) + 1e-8)

    # partial 是可见壳，z 高度通常比完整物体小，所以这里只做软约束
    part_h = robust_height(partial_pts)[2]
    height_ratio = h / (part_h + 1e-8)

    if height_ratio < 0.6:
        height_penalty = (0.6 - height_ratio) * 2.0
    elif height_ratio > 5.0:
        height_penalty = (height_ratio - 5.0) * 0.5
    else:
        height_penalty = 0.0

    score = xy_size_err + 10.0 * below_ratio + height_penalty

    return score, {
        "xy_size_err": float(xy_size_err),
        "below_ratio": float(below_ratio),
        "height": float(h),
        "part_height": float(part_h),
        "height_ratio": float(height_ratio),
    }


def fit_sam3d_to_partial_on_table(sam_pcd, partial_pcd):
    """
    把 SAM3D 点云放到桌面坐标系：
      - 尝试 24 种轴向
      - 按 partial XY 尺寸估计 scale
      - 底部贴 z=0
      - XY 中心对齐到 partial 中心
    """
    print("\n[Fit] 正在把 SAM3D 点云放到桌面 XY 坐标系...")

    sam_pts0 = np_points(sam_pcd)
    partial_pts = np_points(partial_pcd)

    # SAM3D 先中心化
    sam_center0 = np.median(sam_pts0, axis=0)
    sam_pts0 = sam_pts0 - sam_center0

    part_xy_lo, part_xy_hi, part_xy_size = robust_bbox_xy(partial_pts)
    part_xy_center = (part_xy_lo + part_xy_hi) / 2.0

    candidates = generate_axis_rotations()

    best = None
    logs = []

    for i, R in enumerate(candidates):
        pts_r = sam_pts0 @ R.T

        _, _, sam_xy_size = robust_bbox_xy(pts_r)

        # 用 XY 尺寸估计尺度；为了避免某个轴过小，用中位数比例
        valid = sam_xy_size > 1e-8
        if valid.sum() == 0:
            continue

        ratios = part_xy_size[valid] / sam_xy_size[valid]
        scale = float(np.median(ratios))

        # 防止异常尺度
        if not np.isfinite(scale) or scale <= 0:
            continue

        score, info = evaluate_candidate(pts_r, partial_pts, scale)

        logs.append((score, i, scale, info))

        if best is None or score < best[0]:
            best = (score, i, R, scale, info)

    if best is None:
        raise RuntimeError("[ERROR] 没有找到可用的 SAM3D 轴向候选。")

    logs = sorted(logs, key=lambda x: x[0])
    print("\n[Fit] Top candidates:")
    for rank, item in enumerate(logs[:10], 1):
        score, i, scale, info = item
        print(
            f"  #{rank:02d} A[{i:02d}] "
            f"score={score:.4f}, scale={scale:.5f}, "
            f"xy_err={info['xy_size_err']:.4f}, "
            f"height={info['height']:.4f}, "
            f"height_ratio={info['height_ratio']:.2f}, "
            f"below={info['below_ratio']:.4f}"
        )

    score, best_i, best_R, best_scale, best_info = best
    print(
        f"\n[Fit] 选择 A[{best_i:02d}], "
        f"score={score:.4f}, scale={best_scale:.6f}"
    )

    pts = sam_pts0 @ best_R.T
    pts = pts * best_scale

    # 底部贴桌面 z=0
    bottom_z = np.percentile(pts[:, 2], 1)
    pts[:, 2] -= bottom_z

    # XY 中心对齐 partial
    sam_xy_lo, sam_xy_hi, _ = robust_bbox_xy(pts)
    sam_xy_center = (sam_xy_lo + sam_xy_hi) / 2.0
    delta_xy = part_xy_center - sam_xy_center
    pts[:, 0] += delta_xy[0]
    pts[:, 1] += delta_xy[1]

    fitted = make_pcd(pts)

    transform_info = {
        "best_axis_index": int(best_i),
        "scale": float(best_scale),
        "score": float(score),
        "delta_xy": [float(delta_xy[0]), float(delta_xy[1])],
        "bottom_z_before_shift": float(bottom_z),
        "info": best_info,
        "R_axis": best_R.tolist(),
    }

    return fitted, transform_info


def make_table_mesh_from_points(table_pcd, z=0.0, margin=0.10):
    """
    用桌面点云范围生成一块半透明感的绿色桌面矩形。
    Open3D PLY 不支持真正透明，这里用绿色薄平面表示。
    """
    pts = np_points(table_pcd)
    xy_min = np.percentile(pts[:, :2], 1, axis=0) - margin
    xy_max = np.percentile(pts[:, :2], 99, axis=0) + margin

    x0, y0 = xy_min
    x1, y1 = xy_max

    vertices = np.array([
        [x0, y0, z],
        [x1, y0, z],
        [x1, y1, z],
        [x0, y1, z],
    ], dtype=np.float64)

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.0, 0.75, 0.0])
    return mesh


def save_transform_json(path, info):
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam3d", required=True, help="SAM3D 生成的完整点云 ply")
    parser.add_argument("--table", required=True, help="桌面点云，建议 table_plane_inliers_table_xy.ply")
    parser.add_argument("--partial", required=True, help="目标 mask 回投影点云，建议 object_points_from_mask_table_xy.ply")
    parser.add_argument("--out-dir", default="./output_sam3d_on_green_table")

    parser.add_argument("--voxel", type=float, default=0.003)
    parser.add_argument("--dbscan-eps", type=float, default=0.018)
    parser.add_argument("--dbscan-min-points", type=int, default=20)

    parser.add_argument("--show", action="store_true", help="弹窗显示 Open3D 可视化")
    parser.add_argument("--no-clean", action="store_true", help="不清理 SAM3D 浮空碎片")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 显示 SAM3D 点云在绿色桌面 XY 平面上")
    print("=" * 80)

    sam3d_raw = load_pcd(args.sam3d, "SAM3D raw")
    table_pcd = load_pcd(args.table, "green table")
    partial_pcd = load_pcd(args.partial, "red partial object")

    # 颜色
    table_pcd.paint_uniform_color([0.0, 0.85, 0.0])     # 绿色桌面点
    partial_pcd.paint_uniform_color([1.0, 0.0, 0.0])    # 红色 partial

    if args.no_clean:
        sam3d_clean = sam3d_raw
        print("\n[Clean] 跳过 SAM3D 清理。")
    else:
        sam3d_clean = remove_outliers_and_keep_main_cluster(
            sam3d_raw,
            voxel_size=args.voxel,
            dbscan_eps=args.dbscan_eps,
            dbscan_min_points=args.dbscan_min_points,
        )

    o3d.io.write_point_cloud(
        os.path.join(args.out_dir, "sam3d_clean_main_cluster.ply"),
        sam3d_clean,
    )

    sam3d_on_table, info = fit_sam3d_to_partial_on_table(sam3d_clean, partial_pcd)
    sam3d_on_table.paint_uniform_color([0.1, 0.35, 1.0])  # 蓝色 SAM3D

    table_mesh = make_table_mesh_from_points(table_pcd, z=0.0, margin=0.10)

    # 保存结果
    out_sam = os.path.join(args.out_dir, "sam3d_on_green_table_blue.ply")
    out_table = os.path.join(args.out_dir, "green_table_points.ply")
    out_partial = os.path.join(args.out_dir, "partial_red_table_xy.ply")
    out_transform = os.path.join(args.out_dir, "sam3d_on_table_transform.json")

    o3d.io.write_point_cloud(out_sam, sam3d_on_table)
    o3d.io.write_point_cloud(out_table, table_pcd)
    o3d.io.write_point_cloud(out_partial, partial_pcd)
    save_transform_json(out_transform, info)

    # 合成一个点云版本，方便你直接 CloudCompare / Open3D 打开
    merged = table_pcd + partial_pcd + sam3d_on_table
    out_merged = os.path.join(args.out_dir, "scene_table_green_partial_red_sam3d_blue.ply")
    o3d.io.write_point_cloud(out_merged, merged)

    print("\n[Save]")
    print(f"  cleaned SAM3D        : {os.path.abspath(os.path.join(args.out_dir, 'sam3d_clean_main_cluster.ply'))}")
    print(f"  SAM3D on table       : {os.path.abspath(out_sam)}")
    print(f"  merged scene         : {os.path.abspath(out_merged)}")
    print(f"  transform json       : {os.path.abspath(out_transform)}")
    print("\n颜色说明：")
    print("  绿色 = 桌面 XY 平面")
    print("  红色 = 真实深度/Mask 回投影 partial 点云")
    print("  蓝色 = SAM3D 生成点云，已贴到桌面上")

    if args.show:
        print("\n[Show] 打开 Open3D 可视化窗口...")
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])

        # 注意：table_mesh 是绿色面，table_pcd 是绿色桌面点。
        # 如果面遮挡点云，可以按 H 隐藏/显示几何，或者把 table_mesh 从列表里删掉。
        o3d.visualization.draw_geometries(
            [
                table_mesh,
                table_pcd,
                partial_pcd,
                sam3d_on_table,
                coord,
            ],
            window_name="SAM3D on Green Table: green=table, red=partial, blue=SAM3D",
            width=1280,
            height=900,
            point_show_normal=False,
        )

    print("\n✅ 完成。优先打开这个文件看效果：")
    print(f"  {os.path.abspath(out_merged)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
