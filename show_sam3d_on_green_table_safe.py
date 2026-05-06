#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
show_sam3d_on_green_table_safe.py

安全版：
  完全绕开 Open3D 的显示、paint、DBSCAN、写文件操作，避免 Segmentation fault。

功能：
  1. 读取 SAM3D 点云
  2. 读取已经变成 XY 平面的桌面点云
  3. 读取真实深度 partial 点云
  4. 清理 SAM3D 浮空离群点
  5. 自动把 SAM3D 放到绿色桌面 z=0 上
  6. 保存彩色 PLY:
       绿色 = 桌面
       红色 = partial
       蓝色 = SAM3D
  7. 保存 PNG 预览图

典型运行：
python show_sam3d_on_green_table_safe.py \
  --sam3d ./sam3d_duck_clean.ply \
  --table ./output_table_axis_probe_xy/table_plane_inliers_table_xy.ply \
  --partial ./output_table_axis_probe_xy/object_points_from_mask_table_xy.ply \
  --out-dir ./output_duck_on_green_table_safe
"""

import os
import json
import argparse
import itertools
import numpy as np

from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# 基础工具
# -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_ply_xyz(path):
    """
    轻量 PLY 读取器，只读取 vertex 的 x y z。
    支持：
      - ascii
      - binary_little_endian

    不依赖 Open3D，避免 segfault。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"PLY header 不完整: {path}")
            header_lines.append(line.decode("utf-8", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break

        if header_lines[0] != "ply":
            raise RuntimeError(f"不是 PLY 文件: {path}")

        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False

        for line in header_lines:
            parts = line.split()
            if len(parts) == 0:
                continue

            if parts[0] == "format":
                fmt = parts[1]

            elif parts[0] == "element":
                if parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    in_vertex = True
                else:
                    in_vertex = False

            elif parts[0] == "property" and in_vertex:
                # property float x
                # property uchar red
                if len(parts) >= 3:
                    properties.append((parts[1], parts[2]))

        if fmt is None or vertex_count is None:
            raise RuntimeError(f"PLY header 缺少 format 或 vertex 数量: {path}")

        prop_names = [p[1] for p in properties]
        if not all(k in prop_names for k in ["x", "y", "z"]):
            raise RuntimeError(f"PLY 不包含 x/y/z 属性: {path}")

        x_idx = prop_names.index("x")
        y_idx = prop_names.index("y")
        z_idx = prop_names.index("z")

        if fmt == "ascii":
            pts = []
            for _ in range(vertex_count):
                line = f.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                vals = line.split()
                pts.append([
                    float(vals[x_idx]),
                    float(vals[y_idx]),
                    float(vals[z_idx]),
                ])
            pts = np.asarray(pts, dtype=np.float64)

        elif fmt == "binary_little_endian":
            dtype_map = {
                "char": "i1",
                "uchar": "u1",
                "int8": "i1",
                "uint8": "u1",
                "short": "<i2",
                "ushort": "<u2",
                "int16": "<i2",
                "uint16": "<u2",
                "int": "<i4",
                "uint": "<u4",
                "int32": "<i4",
                "uint32": "<u4",
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "float64": "<f8",
            }

            dtype_fields = []
            for ptype, pname in properties:
                if ptype not in dtype_map:
                    raise RuntimeError(f"暂不支持 PLY 属性类型 {ptype}: {path}")
                dtype_fields.append((pname, dtype_map[ptype]))

            dtype = np.dtype(dtype_fields)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)

            pts = np.stack([
                data["x"].astype(np.float64),
                data["y"].astype(np.float64),
                data["z"].astype(np.float64),
            ], axis=1)

        else:
            raise RuntimeError(f"暂不支持 PLY format: {fmt}")

    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


def write_colored_ply_ascii(path, xyz, rgb):
    """
    写 ASCII 彩色 PLY。
    rgb 范围 0-255。
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.uint8)

    assert xyz.shape[0] == rgb.shape[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(xyz, rgb):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def voxel_downsample_np(points, voxel_size):
    if voxel_size <= 0:
        return points

    q = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    idx = np.sort(idx)
    return points[idx]


def robust_bbox_xy(points, q_low=2.0, q_high=98.0):
    lo = np.percentile(points[:, :2], q_low, axis=0)
    hi = np.percentile(points[:, :2], q_high, axis=0)
    size = hi - lo
    center = 0.5 * (lo + hi)
    return lo, hi, size, center


def robust_z_range(points, q_low=1.0, q_high=99.0):
    zlo = float(np.percentile(points[:, 2], q_low))
    zhi = float(np.percentile(points[:, 2], q_high))
    return zlo, zhi, zhi - zlo


# -----------------------------
# SAM3D 清理
# -----------------------------

def remove_statistical_outliers_np(points, k=20, std_ratio=2.0):
    """
    NumPy + scipy 版本统计离群点去除。
    """
    if points.shape[0] <= k + 1:
        return points

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)

    # 第 0 个是自己，距离为 0
    mean_dists = dists[:, 1:].mean(axis=1)

    mu = mean_dists.mean()
    sigma = mean_dists.std() + 1e-12
    keep = mean_dists <= mu + std_ratio * sigma

    return points[keep]


def keep_largest_component_radius(points, radius=0.025, min_component_points=30):
    """
    用半径邻接图保留最大连通块。
    适合 7000 点级别。
    """
    n = points.shape[0]
    if n == 0:
        return points

    tree = cKDTree(points)

    visited = np.zeros(n, dtype=bool)
    best_comp = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        comp = []

        while queue:
            u = queue.pop()
            comp.append(u)

            neigh = tree.query_ball_point(points[u], r=radius)
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

        if len(comp) > len(best_comp):
            best_comp = comp

    if len(best_comp) < min_component_points:
        print(f"[Clean][WARN] 最大连通块点数太少 {len(best_comp)}，跳过连通块过滤")
        return points

    return points[np.asarray(best_comp, dtype=np.int64)]


def clean_sam3d_np(points, voxel=0.003, sor_k=20, sor_std=2.0, comp_radius=0.025):
    print("\n[Clean] 安全清理 SAM3D 点云，不使用 Open3D")
    print(f"  raw points = {points.shape[0]}")

    pts = voxel_downsample_np(points, voxel)
    print(f"  voxel downsample {voxel} -> {pts.shape[0]}")

    pts2 = remove_statistical_outliers_np(pts, k=sor_k, std_ratio=sor_std)
    print(f"  statistical outlier -> {pts2.shape[0]}")

    pts3 = keep_largest_component_radius(pts2, radius=comp_radius)
    print(f"  largest component radius={comp_radius} -> {pts3.shape[0]}")

    return pts3


# -----------------------------
# 24 种轴向候选
# -----------------------------

def generate_axis_rotations():
    mats = []
    I = np.eye(3)

    for perm in itertools.permutations([0, 1, 2]):
        P = I[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            R = P @ np.diag(signs)
            if np.linalg.det(R) > 0.5:
                mats.append(R)

    unique = []
    for R in mats:
        if not any(np.allclose(R, Q) for Q in unique):
            unique.append(R)

    return unique


def candidate_score(pts, partial_pts):
    """
    越小越好。

    这里用几个简单稳定的约束：
      1. SAM3D 的 XY 尺寸要接近 partial
      2. SAM3D 不能大量在桌面以下
      3. SAM3D 高度不能离谱
    """
    _, _, sam_xy_size, _ = robust_bbox_xy(pts)
    _, _, part_xy_size, _ = robust_bbox_xy(partial_pts)

    xy_err = np.linalg.norm(sam_xy_size - part_xy_size) / (np.linalg.norm(part_xy_size) + 1e-8)

    below_ratio = float(np.mean(pts[:, 2] < -0.005))

    _, _, sam_h = robust_z_range(pts)
    _, _, part_h = robust_z_range(partial_pts)

    height_ratio = sam_h / (part_h + 1e-8)

    if height_ratio < 0.5:
        height_penalty = 2.0 * (0.5 - height_ratio)
    elif height_ratio > 6.0:
        height_penalty = 0.4 * (height_ratio - 6.0)
    else:
        height_penalty = 0.0

    score = xy_err + 10.0 * below_ratio + height_penalty

    info = {
        "xy_err": float(xy_err),
        "below_ratio": float(below_ratio),
        "sam_height": float(sam_h),
        "partial_height": float(part_h),
        "height_ratio": float(height_ratio),
        "score": float(score),
    }

    return score, info


def fit_sam3d_to_table_partial(sam_pts, partial_pts):
    """
    把 SAM3D 放到桌面坐标系中。

    步骤：
      1. SAM3D 自身中心化
      2. 枚举 24 种轴向旋转
      3. 用 XY 尺寸估计 scale
      4. 底部贴 z=0
      5. XY 中心对齐 partial
    """
    print("\n[Fit] 开始把 SAM3D 放到绿色桌面上")

    sam0 = sam_pts.copy()
    sam0 = sam0 - np.median(sam0, axis=0, keepdims=True)

    _, _, part_xy_size, part_xy_center = robust_bbox_xy(partial_pts)

    rotations = generate_axis_rotations()
    logs = []
    best = None

    for idx, R in enumerate(rotations):
        pts_r = sam0 @ R.T

        _, _, sam_xy_size, _ = robust_bbox_xy(pts_r)

        valid = sam_xy_size > 1e-8
        if valid.sum() == 0:
            continue

        ratios = part_xy_size[valid] / sam_xy_size[valid]
        scale = float(np.median(ratios))

        if not np.isfinite(scale) or scale <= 0:
            continue

        pts = pts_r * scale

        # 底部贴桌面 z=0
        bottom_z = np.percentile(pts[:, 2], 1.0)
        pts[:, 2] -= bottom_z

        # XY 对齐 partial
        _, _, _, sam_xy_center = robust_bbox_xy(pts)
        delta_xy = part_xy_center - sam_xy_center
        pts[:, 0] += delta_xy[0]
        pts[:, 1] += delta_xy[1]

        score, info = candidate_score(pts, partial_pts)

        logs.append((score, idx, scale, delta_xy, bottom_z, info, R))

        if best is None or score < best[0]:
            best = (score, idx, scale, delta_xy, bottom_z, info, R, pts)

    logs = sorted(logs, key=lambda x: x[0])

    print("\n[Fit] Top candidates:")
    for rank, item in enumerate(logs[:10], 1):
        score, idx, scale, delta_xy, bottom_z, info, R = item
        print(
            f"  #{rank:02d} A[{idx:02d}] "
            f"score={score:.4f}, scale={scale:.6f}, "
            f"xy_err={info['xy_err']:.4f}, "
            f"height={info['sam_height']:.4f}, "
            f"height_ratio={info['height_ratio']:.2f}, "
            f"below={info['below_ratio']:.4f}"
        )

    if best is None:
        raise RuntimeError("没有找到可用候选")

    score, idx, scale, delta_xy, bottom_z, info, R, fitted_pts = best

    print("\n[Fit] 最终选择:")
    print(f"  axis_index = A[{idx:02d}]")
    print(f"  scale      = {scale:.8f}")
    print(f"  score      = {score:.6f}")
    print(f"  delta_xy   = [{delta_xy[0]:.6f}, {delta_xy[1]:.6f}]")
    print(f"  bottom_z   = {bottom_z:.6f}")
    print(f"  info       = {info}")

    transform = {
        "axis_index": int(idx),
        "scale": float(scale),
        "score": float(score),
        "delta_xy": [float(delta_xy[0]), float(delta_xy[1])],
        "bottom_z_before_shift": float(bottom_z),
        "R_axis": R.tolist(),
        "info": info,
    }

    return fitted_pts, transform


# -----------------------------
# 输出绿色桌面图
# -----------------------------

def build_colored_scene(table_pts, partial_pts, sam_pts):
    """
    绿色 = 桌面
    红色 = partial
    蓝色 = SAM3D
    """
    table_rgb = np.tile(np.array([[0, 220, 0]], dtype=np.uint8), (table_pts.shape[0], 1))
    partial_rgb = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (partial_pts.shape[0], 1))
    sam_rgb = np.tile(np.array([[30, 90, 255]], dtype=np.uint8), (sam_pts.shape[0], 1))

    xyz = np.concatenate([table_pts, partial_pts, sam_pts], axis=0)
    rgb = np.concatenate([table_rgb, partial_rgb, sam_rgb], axis=0)

    return xyz, rgb


def save_preview_png(path, table_pts, partial_pts, sam_pts, elev=22, azim=-55):
    """
    保存一个 3D 预览图，不打开窗口。
    智星云不能弹窗时，直接下载这个 PNG 看。
    """
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")

    # 点太多会慢，抽样显示
    def sample(points, max_n):
        if points.shape[0] <= max_n:
            return points
        idx = np.random.default_rng(0).choice(points.shape[0], size=max_n, replace=False)
        return points[idx]

    t = sample(table_pts, 2500)
    p = sample(partial_pts, 2500)
    s = sample(sam_pts, 5000)

    ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=2, c="green", label="table")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=3, c="red", label="partial")
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=2, c="blue", label="SAM3D")

    ax.set_title("SAM3D on Green Table\ngreen=table, red=partial, blue=SAM3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    all_pts = np.concatenate([t, p, s], axis=0)
    mins = np.percentile(all_pts, 1, axis=0)
    maxs = np.percentile(all_pts, 99, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.55)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(centers[2] - radius * 0.2, -0.03), centers[2] + radius * 0.8)

    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_top_png(path, table_pts, partial_pts, sam_pts):
    """
    保存俯视 XY 图。
    用来检查 SAM3D 是否放到了绿色桌面目标区域。
    """
    fig = plt.figure(figsize=(8, 8), dpi=160)
    ax = fig.add_subplot(111)

    ax.scatter(table_pts[:, 0], table_pts[:, 1], s=2, c="green", label="table")
    ax.scatter(partial_pts[:, 0], partial_pts[:, 1], s=4, c="red", label="partial")
    ax.scatter(sam_pts[:, 0], sam_pts[:, 1], s=2, c="blue", label="SAM3D")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top View XY\ngreen=table, red=partial, blue=SAM3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out-dir", default="./output_duck_on_green_table_safe")

    parser.add_argument("--voxel", type=float, default=0.003)
    parser.add_argument("--sor-k", type=int, default=20)
    parser.add_argument("--sor-std", type=float, default=2.0)
    parser.add_argument("--comp-radius", type=float, default=0.025)

    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=-55)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] Safe version: SAM3D on green table")
    print("=" * 80)

    sam_raw = read_ply_xyz(args.sam3d)
    table_pts = read_ply_xyz(args.table)
    partial_pts = read_ply_xyz(args.partial)

    print(f"[Load] SAM3D raw : {args.sam3d}, points={sam_raw.shape[0]}")
    print(f"[Load] table     : {args.table}, points={table_pts.shape[0]}")
    print(f"[Load] partial   : {args.partial}, points={partial_pts.shape[0]}")

    # 保存一点统计信息
    print("\n[Stats] table z:")
    print(
        f"  min={table_pts[:,2].min():.6f}, "
        f"p50={np.percentile(table_pts[:,2],50):.6f}, "
        f"max={table_pts[:,2].max():.6f}"
    )

    print("[Stats] partial z:")
    print(
        f"  min={partial_pts[:,2].min():.6f}, "
        f"p1={np.percentile(partial_pts[:,2],1):.6f}, "
        f"p50={np.percentile(partial_pts[:,2],50):.6f}, "
        f"p99={np.percentile(partial_pts[:,2],99):.6f}"
    )

    if args.no_clean:
        sam_clean = sam_raw.copy()
        print("\n[Clean] 跳过清理")
    else:
        sam_clean = clean_sam3d_np(
            sam_raw,
            voxel=args.voxel,
            sor_k=args.sor_k,
            sor_std=args.sor_std,
            comp_radius=args.comp_radius,
        )

    sam_on_table, transform = fit_sam3d_to_table_partial(sam_clean, partial_pts)

    # 输出
    scene_xyz, scene_rgb = build_colored_scene(table_pts, partial_pts, sam_on_table)

    out_scene = os.path.join(args.out_dir, "scene_green_table_red_partial_blue_sam3d_SAFE.ply")
    out_sam = os.path.join(args.out_dir, "sam3d_blue_on_table_SAFE.ply")
    out_png = os.path.join(args.out_dir, "preview_3d_green_table_red_partial_blue_sam3d.png")
    out_top = os.path.join(args.out_dir, "preview_top_xy.png")
    out_json = os.path.join(args.out_dir, "sam3d_on_table_transform_SAFE.json")

    write_colored_ply_ascii(out_scene, scene_xyz, scene_rgb)

    sam_rgb = np.tile(np.array([[30, 90, 255]], dtype=np.uint8), (sam_on_table.shape[0], 1))
    write_colored_ply_ascii(out_sam, sam_on_table, sam_rgb)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(transform, f, indent=2, ensure_ascii=False)

    save_preview_png(out_png, table_pts, partial_pts, sam_on_table, elev=args.elev, azim=args.azim)
    save_top_png(out_top, table_pts, partial_pts, sam_on_table)

    print("\n[Save]")
    print(f"  彩色总场景 PLY : {os.path.abspath(out_scene)}")
    print(f"  蓝色 SAM3D PLY : {os.path.abspath(out_sam)}")
    print(f"  3D 预览 PNG    : {os.path.abspath(out_png)}")
    print(f"  俯视 XY PNG    : {os.path.abspath(out_top)}")
    print(f"  变换参数 JSON  : {os.path.abspath(out_json)}")

    print("\n颜色说明：")
    print("  绿色 = 桌面 XY 平面")
    print("  红色 = 真实 partial 点云")
    print("  蓝色 = SAM3D 生成点云，已贴到桌面上")

    print("\n✅ 完成。优先看这两个文件：")
    print(f"  {os.path.abspath(out_png)}")
    print(f"  {os.path.abspath(out_scene)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
