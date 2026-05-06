#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step_table_axis_visible_shell_align.py

作用：
  在“桌面竖直轴约束配准”的基础上，进一步使用可见壳 visible shell 做精配准。

为什么需要这个脚本：
  旧脚本让真实 partial 匹配完整 SAM3D 鸭子的全部表面，容易出现红色 partial
  钻进蓝色完整模型内部的问题。

核心改法：
  每个候选位姿下：
    1. 把 SAM3D 完整鸭子变换到真实相机坐标系
    2. 用相机内参投影到图像平面
    3. z-buffer 提取相机可见外壳 visible shell
    4. 用真实 partial 匹配 visible shell，而不是完整 full surface

输入：
  --sam3d-single        单独生成的 SAM3D 鸭子
  --real-partial        真实残缺鸭子点云
  --real-table-plane    真实桌面平面 table_plane.json
  --init-transform      上一步桌面轴约束结果 table_axis_constrained_transform.json
  --fx --fy --cx --cy   相机内参
  --width --height      图像尺寸

输出：
  out_dir/
    aligned_sam3d_duck_visible_refined.ply
    visible_shell_refined.ply
    compare_visible_shell_real.ply
    compare_full_duck_real_table.ply
    visible_shell_transform.json
"""

import os
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def rotation_about_axis(axis, angle_rad):
    axis = normalize(axis)
    K = skew(axis)
    I = np.eye(3)
    return I + np.sin(angle_rad) * K + (1.0 - np.cos(angle_rad)) * (K @ K)


def project_to_plane_vec(v, n):
    n = normalize(n)
    return v - np.dot(v, n) * n


def robust_bbox_diag(points, q0=2, q1=98):
    points = np.asarray(points, dtype=np.float64)
    lo = np.percentile(points, q0, axis=0)
    hi = np.percentile(points, q1, axis=0)
    return float(np.linalg.norm(hi - lo))


def downsample_points(points, max_points=60000, seed=0):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=int(max_points), replace=False)
    return points[idx]


def geometry_to_points(path, sample_points=180000):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"[Load] {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"  读取为 mesh: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        return np.asarray(pcd.points, dtype=np.float64)

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        print(f"  读取为 point cloud: points={len(pcd.points)}")
        return np.asarray(pcd.points, dtype=np.float64)

    raise RuntimeError(f"无法读取有效点云/mesh: {path}")


def np_to_pcd(points, color=None):
    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None and len(points) > 0:
        colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_pcd(points, path, color=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    points = np.asarray(points, dtype=np.float64)

    if len(points) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 0\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
        print(f"[Save empty] {path}")
        return

    pcd = np_to_pcd(points, color=color)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def apply_transform(points, scale, R, t):
    points = np.asarray(points, dtype=np.float64)
    return float(scale) * (points @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


# ============================================================
# JSON 读取
# ============================================================

def is_num(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def as_float_list(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)) and all(is_num(v) for v in x):
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


def normalize_plane(normal, d):
    normal = np.asarray(normal, dtype=np.float64)
    d = float(d)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise RuntimeError("plane normal 长度接近 0")
    return normal / norm, d / norm


def load_table_plane(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 优先 plane_model
    for key in ["plane_model", "plane", "coefficients"]:
        if key in data:
            arr = as_float_list(data[key])
            if arr is not None and len(arr) == 4:
                n, d = normalize_plane(arr[:3], arr[3])
                print(f"[Real Table] 使用 key: {key}")
                print(f"[Real Table] normal={n.tolist()}, d={d:.8f}")
                return n, d

    normal_keys = ["normal", "plane_normal", "normal_camera", "table_normal", "table_plane_normal", "n"]
    d_keys = ["d", "plane_d", "d_camera", "table_d", "table_plane_d", "offset"]

    for path, value in recursive_items(data):
        if not isinstance(value, dict):
            continue

        n_arr = None
        d_val = None

        for nk in normal_keys:
            if nk in value:
                arr = as_float_list(value[nk])
                if arr is not None and len(arr) == 3:
                    n_arr = arr
                    break

        for dk in d_keys:
            if dk in value and is_num(value[dk]):
                d_val = float(value[dk])
                break

        if n_arr is not None and d_val is not None:
            n, d = normalize_plane(n_arr, d_val)
            print(f"[Real Table] 使用来源: {path}")
            print(f"[Real Table] normal={n.tolist()}, d={d:.8f}")
            return n, d

    # 递归找 4 长数组
    candidates = []
    for path, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 4:
            score = 0
            lower = path.lower()
            if "plane" in lower:
                score += 10
            if "table" in lower:
                score += 5
            candidates.append((score, path, arr))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        _, path, arr = candidates[0]
        n, d = normalize_plane(arr[:3], arr[3])
        print(f"[Real Table] 使用来源: {path}")
        print(f"[Real Table] normal={n.tolist()}, d={d:.8f}")
        return n, d

    raise RuntimeError(f"无法读取 table plane: {json_path}")


def orient_table_normal_to_object_side(points, n, d):
    points = np.asarray(points, dtype=np.float64)
    h = points @ n + float(d)
    p05 = float(np.percentile(h, 5))
    p95 = float(np.percentile(h, 95))

    if abs(p05) > abs(p95):
        n = -n
        d = -d
        h = -h
        print("[Real Table] 法向已翻转，使物体位于正侧。")
    else:
        print("[Real Table] 法向保持不变。")

    print(
        f"[Real Table] signed height: "
        f"p05={np.percentile(h,5):.6f}, "
        f"p50={np.percentile(h,50):.6f}, "
        f"p95={np.percentile(h,95):.6f}"
    )

    return normalize(n), float(d)


def load_init_transform(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scale = float(data["scale"])

    if "R_total_single_to_real" in data:
        R = np.asarray(data["R_total_single_to_real"], dtype=np.float64)
    elif "R_total" in data:
        R = np.asarray(data["R_total"], dtype=np.float64)
    else:
        raise RuntimeError("init transform json 中找不到 R_total_single_to_real")

    if "t_total_single_to_real" in data:
        t = np.asarray(data["t_total_single_to_real"], dtype=np.float64)
    elif "t_total" in data:
        t = np.asarray(data["t_total"], dtype=np.float64)
    else:
        raise RuntimeError("init transform json 中找不到 t_total_single_to_real")

    yaw = float(data.get("yaw_deg", 0.0))

    print("[Init Transform]")
    print(f"  scale = {scale:.8f}")
    print(f"  yaw   = {yaw:.3f}")
    print(f"  t     = {t.tolist()}")

    return scale, R, t, data


# ============================================================
# 真实 partial 过滤
# ============================================================

def split_real_duck_and_table(points, n_real, d_real, above_thresh=0.008, table_thresh=0.006, no_filter=False):
    points = np.asarray(points, dtype=np.float64)
    h = points @ n_real + float(d_real)

    table_mask = np.abs(h) < float(table_thresh)
    table_pts = points[table_mask]

    if no_filter:
        duck_pts = points
    else:
        duck_mask = h > float(above_thresh)
        duck_pts = points[duck_mask]
        if len(duck_pts) < 200:
            print("[WARN] 真实鸭子候选太少，退回使用原始 real partial。")
            duck_pts = points

    print("\n[Real Split]")
    print(f"  all real points = {len(points)}")
    print(f"  table points    = {len(table_pts)}")
    print(f"  duck candidate  = {len(duck_pts)}")
    print(f"  above_thresh    = {above_thresh}")
    print(f"  table_thresh    = {table_thresh}")

    return duck_pts, table_pts


# ============================================================
# 可见壳提取
# ============================================================

def extract_visible_shell(
    points_cam,
    fx,
    fy,
    cx,
    cy,
    width,
    height,
    depth_margin=0.004,
    z_mode="min",
    max_points=None,
    seed=0,
):
    """
    使用 z-buffer 提取相机可见壳。

    默认相机坐标：
      z > 0
      u = fx*x/z + cx
      v = fy*y/z + cy

    z_mode:
      min: 每个像素取最小 z，常见 RGB-D 相机坐标用这个
      max: 如果发现取到背面，可以切换到 max
    """
    pts = np.asarray(points_cam, dtype=np.float64)

    valid_z = pts[:, 2] > 1e-6
    pts_valid = pts[valid_z]
    if len(pts_valid) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = pts_valid[:, 2]
    u = np.round(fx * pts_valid[:, 0] / z + cx).astype(np.int64)
    v = np.round(fy * pts_valid[:, 1] / z + cy).astype(np.int64)

    inside = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    pts_in = pts_valid[inside]
    u = u[inside]
    v = v[inside]

    if len(pts_in) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = pts_in[:, 2]
    pix = v * int(width) + u

    order = np.lexsort((z, pix)) if z_mode == "min" else np.lexsort((-z, pix))

    pix_sorted = pix[order]
    z_sorted = z[order]

    unique_pix, first_idx = np.unique(pix_sorted, return_index=True)
    z_ref = z_sorted[first_idx]

    # 构造 pixel -> z_ref 的数组
    ref = np.full(int(width) * int(height), np.nan, dtype=np.float64)
    ref[unique_pix] = z_ref

    z_front = ref[pix]

    if z_mode == "min":
        visible_mask = z <= (z_front + float(depth_margin))
    else:
        visible_mask = z >= (z_front - float(depth_margin))

    shell = pts_in[visible_mask]

    if max_points is not None and len(shell) > max_points:
        shell = downsample_points(shell, max_points=max_points, seed=seed)

    return shell


# ============================================================
# 打分和微调
# ============================================================

def trimmed_rmse(dists, trim_ratio):
    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]
    if len(dists) == 0:
        return 1e9
    keep_n = max(10, int(float(trim_ratio) * len(dists)))
    keep_n = min(keep_n, len(dists))
    keep = np.partition(dists, keep_n - 1)[:keep_n]
    return float(np.sqrt(np.mean(keep ** 2)))


def score_visible_shell(shell, real_duck, full_points, n_real, d_real, coverage_thresh, p2v_trim=0.90, v2p_trim=0.30):
    if len(shell) < 50:
        return {
            "objective": 1e9,
            "rmse_p2v": 1e9,
            "rmse_v2p": 1e9,
            "coverage": 0.0,
            "bottom_h": 0.0,
            "penetration": 1e9,
        }

    tree_shell = cKDTree(shell)
    d_p2v, _ = tree_shell.query(real_duck, k=1, workers=-1)

    tree_real = cKDTree(real_duck)
    d_v2p, _ = tree_real.query(shell, k=1, workers=-1)

    rmse_p2v = trimmed_rmse(d_p2v, p2v_trim)
    rmse_v2p = trimmed_rmse(d_v2p, v2p_trim)

    coverage = float(np.mean(d_p2v < float(coverage_thresh)))

    h = full_points @ n_real + float(d_real)
    bottom_h = float(np.percentile(h, 1.0))
    penetration = max(0.0, -bottom_h)

    objective = rmse_p2v + 0.20 * rmse_v2p + 2.0 * penetration

    return {
        "objective": float(objective),
        "rmse_p2v": float(rmse_p2v),
        "rmse_v2p": float(rmse_v2p),
        "coverage": float(coverage),
        "bottom_h": float(bottom_h),
        "penetration": float(penetration),
    }


def refine_inplane_translation_visible(
    src_points_single,
    real_duck,
    scale,
    R,
    t_init,
    n_real,
    d_real,
    args,
    iters=8,
):
    """
    固定 scale 和 R，只优化桌面平面内平移。
    每次都用 visible shell 来匹配 real partial。
    """
    t = np.asarray(t_init, dtype=np.float64).reshape(3).copy()

    src_rot_scaled = float(scale) * (src_points_single @ R.T)

    for it in range(int(iters)):
        full = src_rot_scaled + t.reshape(1, 3)

        shell = extract_visible_shell(
            full,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            width=args.width,
            height=args.height,
            depth_margin=args.depth_margin,
            z_mode=args.camera_z_mode,
            max_points=args.visible_max_points,
            seed=args.seed + it,
        )

        if len(shell) < 50:
            break

        tree = cKDTree(shell)
        dists, idx = tree.query(real_duck, k=1, workers=-1)

        keep_n = max(50, int(args.translation_refine_trim * len(real_duck)))
        keep_n = min(keep_n, len(real_duck))
        order = np.argsort(dists)
        keep = order[:keep_n]

        matched_shell = shell[idx[keep]]
        target = real_duck[keep]

        delta = target - matched_shell
        step = np.median(delta, axis=0)
        step = project_to_plane_vec(step, n_real)

        if np.linalg.norm(step) < 1e-7:
            break

        t = t + step

    return t


def make_initial_translation_for_candidate(src_points_single, real_duck, scale, R, n_real, d_real, table_offset=0.0, bottom_percentile=1.0):
    """
    先让 SAM3D 鸭子底部贴到真实桌面；
    再用 centroid 初始化平面内平移。
    """
    src_rs = float(scale) * (src_points_single @ R.T)

    bottom_dot = float(np.percentile(src_rs @ n_real, float(bottom_percentile)))
    t_normal = (-float(d_real) + float(table_offset) - bottom_dot) * n_real

    src_h = src_rs + t_normal.reshape(1, 3)

    diff = real_duck.mean(axis=0) - src_h.mean(axis=0)
    t_plane = project_to_plane_vec(diff, n_real)

    return t_normal + t_plane


# ============================================================
# 可视化
# ============================================================

def make_table_plane_vis(n, d, center_hint, size=0.35, grid=70):
    n = normalize(n)
    center_hint = np.asarray(center_hint, dtype=np.float64)
    center = center_hint - (np.dot(n, center_hint) + float(d)) * n

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = normalize(np.cross(n, tmp))
    v = normalize(np.cross(n, u))

    vals = np.linspace(-size / 2, size / 2, int(grid))
    pts = []
    for a in vals:
        for b in vals:
            pts.append(center + a * u + b * v)

    return np.asarray(pts, dtype=np.float64)


def save_compare_visible(shell, real_duck, table_points, path):
    # 蓝色：可见壳
    # 红色：真实 partial
    # 绿色：桌面
    p1 = np_to_pcd(shell, color=[0.1, 0.35, 1.0])
    p2 = np_to_pcd(real_duck, color=[1.0, 0.05, 0.02])
    p3 = np_to_pcd(table_points, color=[0.1, 0.9, 0.2])
    combined = p1 + p2 + p3
    ok = o3d.io.write_point_cloud(path, combined)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}")


def save_compare_full(full, real_duck, table_points, path):
    # 蓝色：完整鸭子
    # 红色：真实 partial
    # 绿色：桌面
    p1 = np_to_pcd(full, color=[0.1, 0.35, 1.0])
    p2 = np_to_pcd(real_duck, color=[1.0, 0.05, 0.02])
    p3 = np_to_pcd(table_points, color=[0.1, 0.9, 0.2])
    combined = p1 + p2 + p3
    ok = o3d.io.write_point_cloud(path, combined)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}")


# ============================================================
# 主搜索
# ============================================================

def run_visible_shell_search(src_single_full, real_duck_full, n_real, d_real, init_scale, init_R, init_t, args):
    src_search = downsample_points(src_single_full, max_points=args.search_source_points, seed=args.seed)
    real_search = downsample_points(real_duck_full, max_points=args.search_target_points, seed=args.seed + 1)

    target_diag = robust_bbox_diag(real_search)
    coverage_thresh = float(args.coverage_thresh_ratio * target_diag)

    print("\n" + "=" * 80)
    print("[Visible Shell Search]")
    print(f"  src_search points  = {len(src_search)}")
    print(f"  real_search points = {len(real_search)}")
    print(f"  init_scale         = {init_scale:.8f}")
    print(f"  coverage_thresh    = {coverage_thresh:.6f}")
    print("=" * 80)

    scale_values = np.linspace(args.scale_min, args.scale_max, int(args.scale_steps))

    yaw_values = np.arange(
        -float(args.yaw_range_deg),
        float(args.yaw_range_deg) + 1e-9,
        float(args.yaw_step_deg),
        dtype=np.float64,
    )

    best = None
    total = 0

    for scale in scale_values:
        for yaw_delta in yaw_values:
            total += 1

            R_delta = rotation_about_axis(n_real, np.deg2rad(yaw_delta))
            R = R_delta @ init_R

            t0 = make_initial_translation_for_candidate(
                src_points_single=src_search,
                real_duck=real_search,
                scale=scale,
                R=R,
                n_real=n_real,
                d_real=d_real,
                table_offset=args.table_offset,
                bottom_percentile=args.bottom_percentile,
            )

            t = refine_inplane_translation_visible(
                src_points_single=src_search,
                real_duck=real_search,
                scale=scale,
                R=R,
                t_init=t0,
                n_real=n_real,
                d_real=d_real,
                args=args,
                iters=args.translation_refine_iters,
            )

            full = apply_transform(src_search, scale, R, t)
            shell = extract_visible_shell(
                full,
                fx=args.fx,
                fy=args.fy,
                cx=args.cx,
                cy=args.cy,
                width=args.width,
                height=args.height,
                depth_margin=args.depth_margin,
                z_mode=args.camera_z_mode,
                max_points=args.visible_max_points,
                seed=args.seed + total,
            )

            sc = score_visible_shell(
                shell=shell,
                real_duck=real_search,
                full_points=full,
                n_real=n_real,
                d_real=d_real,
                coverage_thresh=coverage_thresh,
                p2v_trim=args.p2v_trim,
                v2p_trim=args.v2p_trim,
            )

            rec = {
                "scale": float(scale),
                "yaw_delta_deg": float(yaw_delta),
                "R": R,
                "t": t,
                "num_visible_shell": int(len(shell)),
                **sc,
            }

            if best is None or rec["objective"] < best["objective"]:
                best = rec
                print(
                    f"[Best coarse] "
                    f"obj={best['objective']:.6f}, "
                    f"p2v={best['rmse_p2v']:.6f}, "
                    f"v2p={best['rmse_v2p']:.6f}, "
                    f"cov={best['coverage']:.4f}, "
                    f"scale={best['scale']:.6f}, "
                    f"yaw_delta={best['yaw_delta_deg']:.2f}, "
                    f"visible={best['num_visible_shell']}, "
                    f"bottom_h={best['bottom_h']:.6f}"
                )

    print(f"[Visible Shell Search] coarse candidates = {total}")

    # Fine search
    print("\n" + "=" * 80)
    print("[Visible Shell Fine Search]")
    print("=" * 80)

    fine_best = best.copy()

    scale_center = float(best["scale"])
    yaw_center = float(best["yaw_delta_deg"])

    fine_scales = np.linspace(
        scale_center * args.fine_scale_min_mult,
        scale_center * args.fine_scale_max_mult,
        int(args.fine_scale_steps),
    )

    fine_yaws = np.arange(
        yaw_center - args.yaw_step_deg,
        yaw_center + args.yaw_step_deg + 1e-9,
        float(args.fine_yaw_step_deg),
        dtype=np.float64,
    )

    fine_total = 0

    for scale in fine_scales:
        for yaw_delta in fine_yaws:
            fine_total += 1

            R_delta = rotation_about_axis(n_real, np.deg2rad(yaw_delta))
            R = R_delta @ init_R

            t0 = make_initial_translation_for_candidate(
                src_points_single=src_search,
                real_duck=real_search,
                scale=scale,
                R=R,
                n_real=n_real,
                d_real=d_real,
                table_offset=args.table_offset,
                bottom_percentile=args.bottom_percentile,
            )

            t = refine_inplane_translation_visible(
                src_points_single=src_search,
                real_duck=real_search,
                scale=scale,
                R=R,
                t_init=t0,
                n_real=n_real,
                d_real=d_real,
                args=args,
                iters=args.translation_refine_iters + 4,
            )

            full = apply_transform(src_search, scale, R, t)
            shell = extract_visible_shell(
                full,
                fx=args.fx,
                fy=args.fy,
                cx=args.cx,
                cy=args.cy,
                width=args.width,
                height=args.height,
                depth_margin=args.depth_margin,
                z_mode=args.camera_z_mode,
                max_points=args.visible_max_points,
                seed=args.seed + 10000 + fine_total,
            )

            sc = score_visible_shell(
                shell=shell,
                real_duck=real_search,
                full_points=full,
                n_real=n_real,
                d_real=d_real,
                coverage_thresh=coverage_thresh,
                p2v_trim=args.p2v_trim,
                v2p_trim=args.v2p_trim,
            )

            rec = {
                "scale": float(scale),
                "yaw_delta_deg": float(yaw_delta),
                "R": R,
                "t": t,
                "num_visible_shell": int(len(shell)),
                **sc,
            }

            if rec["objective"] < fine_best["objective"]:
                fine_best = rec
                print(
                    f"[Best fine] "
                    f"obj={fine_best['objective']:.6f}, "
                    f"p2v={fine_best['rmse_p2v']:.6f}, "
                    f"v2p={fine_best['rmse_v2p']:.6f}, "
                    f"cov={fine_best['coverage']:.4f}, "
                    f"scale={fine_best['scale']:.6f}, "
                    f"yaw_delta={fine_best['yaw_delta_deg']:.2f}, "
                    f"visible={fine_best['num_visible_shell']}, "
                    f"bottom_h={fine_best['bottom_h']:.6f}"
                )

    print(f"[Visible Shell Fine Search] candidates = {fine_total}")

    return fine_best, coverage_thresh


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam3d-single", required=True)
    parser.add_argument("--real-partial", required=True)
    parser.add_argument("--real-table-plane", required=True)
    parser.add_argument("--init-transform", required=True)

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--out-dir", default="./output_visible_shell_align")

    parser.add_argument("--sample-points", type=int, default=180000)
    parser.add_argument("--search-source-points", type=int, default=45000)
    parser.add_argument("--search-target-points", type=int, default=6000)

    # 真实 partial 过滤
    parser.add_argument("--real-duck-above-thresh", type=float, default=0.008)
    parser.add_argument("--real-table-thresh", type=float, default=0.006)
    parser.add_argument("--no-filter-real-table", action="store_true")

    # visible shell 参数
    parser.add_argument("--depth-margin", type=float, default=0.005)
    parser.add_argument("--camera-z-mode", choices=["min", "max"], default="min")
    parser.add_argument("--visible-max-points", type=int, default=45000)

    # 小范围搜索
    parser.add_argument("--scale-min", type=float, default=-1.0)
    parser.add_argument("--scale-max", type=float, default=-1.0)
    parser.add_argument("--scale-steps", type=int, default=5)
    parser.add_argument("--yaw-range-deg", type=float, default=20.0)
    parser.add_argument("--yaw-step-deg", type=float, default=5.0)

    # fine search
    parser.add_argument("--fine-scale-min-mult", type=float, default=0.98)
    parser.add_argument("--fine-scale-max-mult", type=float, default=1.02)
    parser.add_argument("--fine-scale-steps", type=int, default=5)
    parser.add_argument("--fine-yaw-step-deg", type=float, default=1.0)

    # 桌面贴合
    parser.add_argument("--bottom-percentile", type=float, default=1.0)
    parser.add_argument("--table-offset", type=float, default=0.0)

    # 平移微调
    parser.add_argument("--translation-refine-iters", type=int, default=8)
    parser.add_argument("--translation-refine-trim", type=float, default=0.90)

    # 评分
    parser.add_argument("--p2v-trim", type=float, default=0.90)
    parser.add_argument("--v2p-trim", type=float, default=0.30)
    parser.add_argument("--coverage-thresh-ratio", type=float, default=0.04)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 桌面约束 + 可见壳 visible shell 精配准")
    print("=" * 80)

    src_single = geometry_to_points(args.sam3d_single, sample_points=args.sample_points)
    real_all = geometry_to_points(args.real_partial, sample_points=args.sample_points)

    n_real, d_real = load_table_plane(args.real_table_plane)
    n_real, d_real = orient_table_normal_to_object_side(real_all, n_real, d_real)

    init_scale, init_R, init_t, init_data = load_init_transform(args.init_transform)

    if args.scale_min <= 0 or args.scale_max <= 0:
        args.scale_min = init_scale * 0.97
        args.scale_max = init_scale * 1.03
        print(f"[Scale Range] 自动设置: {args.scale_min:.6f} ~ {args.scale_max:.6f}")

    real_duck, real_table = split_real_duck_and_table(
        real_all,
        n_real=n_real,
        d_real=d_real,
        above_thresh=args.real_duck_above_thresh,
        table_thresh=args.real_table_thresh,
        no_filter=args.no_filter_real_table,
    )

    if len(real_table) < 100:
        table_vis = make_table_plane_vis(
            n_real,
            d_real,
            center_hint=real_duck.mean(axis=0),
            size=max(0.25, robust_bbox_diag(real_duck) * 1.8),
            grid=70,
        )
    else:
        table_vis = real_table

    # 保存初始可见壳，方便检查相机方向
    init_full = apply_transform(src_single, init_scale, init_R, init_t)
    init_shell = extract_visible_shell(
        init_full,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
        depth_margin=args.depth_margin,
        z_mode=args.camera_z_mode,
        max_points=args.visible_max_points,
        seed=args.seed,
    )

    save_pcd(real_duck, os.path.join(args.out_dir, "real_duck_candidate.ply"), color=[1.0, 0.05, 0.02])
    save_pcd(table_vis, os.path.join(args.out_dir, "real_table_plane_vis.ply"), color=[0.1, 0.9, 0.2])
    save_pcd(init_shell, os.path.join(args.out_dir, "visible_shell_init.ply"), color=[0.1, 0.35, 1.0])

    save_compare_visible(
        init_shell,
        real_duck,
        table_vis,
        os.path.join(args.out_dir, "compare_visible_shell_init.ply"),
    )

    print("\n[Init Visible Shell]")
    print(f"  init shell points = {len(init_shell)}")
    print("  已保存 compare_visible_shell_init.ply")
    print("  如果这里蓝色可见壳在背面，重跑时加：--camera-z-mode max")

    best, coverage_thresh = run_visible_shell_search(
        src_single_full=src_single,
        real_duck_full=real_duck,
        n_real=n_real,
        d_real=d_real,
        init_scale=init_scale,
        init_R=init_R,
        init_t=init_t,
        args=args,
    )

    scale = float(best["scale"])
    R = np.asarray(best["R"], dtype=np.float64)
    t = np.asarray(best["t"], dtype=np.float64)

    final_full = apply_transform(src_single, scale, R, t)
    final_shell = extract_visible_shell(
        final_full,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
        depth_margin=args.depth_margin,
        z_mode=args.camera_z_mode,
        max_points=None,
        seed=args.seed + 999,
    )

    out_full = os.path.join(args.out_dir, "aligned_sam3d_duck_visible_refined.ply")
    out_shell = os.path.join(args.out_dir, "visible_shell_refined.ply")
    out_compare_shell = os.path.join(args.out_dir, "compare_visible_shell_real.ply")
    out_compare_full = os.path.join(args.out_dir, "compare_full_duck_real_table.ply")
    out_json = os.path.join(args.out_dir, "visible_shell_transform.json")

    save_pcd(final_full, out_full, color=[0.1, 0.35, 1.0])
    save_pcd(final_shell, out_shell, color=[0.1, 0.35, 1.0])

    save_compare_visible(final_shell, real_duck, table_vis, out_compare_shell)
    save_compare_full(final_full, real_duck, table_vis, out_compare_full)

    info = {
        "definition": "p_real = scale * R @ p_single + t",
        "source_sam3d_single": os.path.abspath(args.sam3d_single),
        "source_real_partial": os.path.abspath(args.real_partial),
        "real_table_plane": os.path.abspath(args.real_table_plane),
        "init_transform": os.path.abspath(args.init_transform),

        "scale": scale,
        "R_single_to_real": R.tolist(),
        "t_single_to_real": t.tolist(),

        "yaw_delta_from_init_deg": float(best["yaw_delta_deg"]),
        "objective": float(best["objective"]),
        "rmse_partial_to_visible": float(best["rmse_p2v"]),
        "rmse_visible_to_partial_trimmed": float(best["rmse_v2p"]),
        "coverage": float(best["coverage"]),
        "coverage_thresh": float(coverage_thresh),
        "bottom_h": float(best["bottom_h"]),
        "penetration": float(best["penetration"]),
        "num_visible_shell": int(len(final_shell)),

        "camera": {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
            "width": int(args.width),
            "height": int(args.height),
            "z_mode": args.camera_z_mode,
            "depth_margin": float(args.depth_margin),
        },

        "outputs": {
            "aligned_full": out_full,
            "visible_shell": out_shell,
            "compare_visible_shell_real": out_compare_shell,
            "compare_full_duck_real_table": out_compare_full,
        }
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "█" * 80)
    print("✅ 完成：桌面约束 + 可见壳精配准")
    print("重点输出：")
    print(f"  1. 完整鸭子最终结果      : {os.path.abspath(out_full)}")
    print(f"  2. 最终可见壳            : {os.path.abspath(out_shell)}")
    print(f"  3. 可见壳对比            : {os.path.abspath(out_compare_shell)}")
    print(f"  4. 完整鸭子对比          : {os.path.abspath(out_compare_full)}")
    print(f"  5. 变换 JSON             : {os.path.abspath(out_json)}")
    print("█" * 80)

    print("\n颜色说明：")
    print("  compare_visible_shell_real.ply:")
    print("    蓝色 = SAM3D 可见壳")
    print("    红色 = 真实 partial")
    print("    绿色 = 真实桌面")
    print("  compare_full_duck_real_table.ply:")
    print("    蓝色 = 完整 SAM3D 鸭子")
    print("    红色 = 真实 partial")
    print("    绿色 = 真实桌面")

    print("\n关键指标：")
    print(f"  scale                    = {scale:.6f}")
    print(f"  yaw_delta_from_init_deg  = {best['yaw_delta_deg']:.2f}")
    print(f"  rmse_partial_to_visible  = {best['rmse_p2v']:.6f}")
    print(f"  coverage                 = {best['coverage']:.4f}")
    print(f"  visible_shell_points     = {len(final_shell)}")
    print(f"  penetration              = {best['penetration']:.6f}")


if __name__ == "__main__":
    main()