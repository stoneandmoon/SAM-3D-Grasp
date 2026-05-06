#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step_table_axis_constrained_align.py

目标：
  使用“桌面竖直轴约束”把单独 SAM3D 鸭子点云配准到真实残缺鸭子点云。

输入：
  1. 单独 SAM3D 鸭子点云：
       sam3d_duck_clean.ply

  2. 真实残缺鸭子点云：
       duck_partial_real_clean_interp_clean.ply
     注意：如果这个点云里还带桌面，也可以，脚本会根据真实 table_plane.json 自动过滤桌面。

  3. 真实桌面平面：
       output_table_axis_probe_xy/table_plane.json

  4. single duck 坐标系下的桌面法向：
       output_joint_table_extract/table_normal_in_single_duck_frame.json

核心思想：
  1. 读取 single duck 里的桌面法向 n_single
  2. 读取真实桌面法向 n_real
  3. 先把 n_single 对齐到 n_real，锁定 roll / pitch
  4. 高度方向贴到真实桌面
  5. 只搜索：
       scale
       yaw around n_real
       translation on real table plane
  6. 输出最终配准后的 SAM3D 完整鸭子

输出：
  out_dir/
    aligned_sam3d_duck_to_real.ply
    real_duck_candidate.ply
    real_table_points.ply
    compare_aligned_sam3d_real.ply
    table_axis_constrained_transform.json

变换定义：
  p_real = scale * R_total @ p_single + t_total

其中：
  R_total = R_yaw_about_real_table_normal @ R_align_single_normal_to_real_normal
"""

import os
import json
import argparse
import itertools
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
        [-y, x, 0.0]
    ], dtype=np.float64)


def rotation_from_a_to_b(a, b):
    """
    返回 R，使得：
      R @ a ≈ b
    """
    a = normalize(a)
    b = normalize(b)

    c = float(np.dot(a, b))

    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-10:
        # 反向，找一个任意垂直轴旋转 180 度
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize(np.cross(a, tmp))
        K = skew(axis)
        return np.eye(3) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)
    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))
    return R


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


def downsample_points(points, max_points=50000, seed=0):
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def geometry_to_pcd(path, sample_points=160000):
    """
    兼容读取 mesh / point cloud。
    如果是 mesh，则采样成点云。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    print(f"[Load] {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"  读取为 mesh: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        return pcd

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        print(f"  读取为 point cloud: points={len(pcd.points)}")
        return pcd

    raise RuntimeError(f"无法读取有效点云/mesh: {path}")


def get_points(pcd):
    return np.asarray(pcd.points, dtype=np.float64)


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

    # Open3D 不能写 0 点 PLY。真实 partial 如果已经去掉桌面，
    # real_table_points 可能就是空的，这是正常情况。
    if len(points) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\\n")
            f.write("format ascii 1.0\\n")
            f.write("element vertex 0\\n")
            f.write("property float x\\n")
            f.write("property float y\\n")
            f.write("property float z\\n")
            f.write("end_header\\n")
        print(f"[Save empty] {path}  points=0")
        return

    pcd = np_to_pcd(points, color=color)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def apply_transform(points, scale, R, t):
    points = np.asarray(points, dtype=np.float64)
    return float(scale) * (points @ R.T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


# ============================================================
# 读取真实 table_plane.json
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
        raise RuntimeError("平面 normal 长度接近 0")

    return normal / norm, d / norm


def load_table_plane(json_path):
    """
    兼容多种 table_plane.json 格式：
      {"normal": [...], "d": ...}
      {"plane_normal": [...], "plane_d": ...}
      {"plane_model": [a,b,c,d]}
      {"plane": [a,b,c,d]}
      {"coefficients": [a,b,c,d]}
      嵌套 dict 也会递归查找。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 table_plane.json: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normal_keys = [
        "normal",
        "plane_normal",
        "normal_camera",
        "table_normal",
        "table_plane_normal",
        "n",
    ]

    d_keys = [
        "d",
        "plane_d",
        "d_camera",
        "table_d",
        "table_plane_d",
        "offset",
    ]

    # 1. 同一个 dict 里找 normal + d
    for path, value in recursive_items(data):
        if not isinstance(value, dict):
            continue

        for nk in normal_keys:
            if nk not in value:
                continue

            n_arr = as_float_list(value[nk])
            if n_arr is None or len(n_arr) != 3:
                continue

            for dk in d_keys:
                if dk in value and is_num(value[dk]):
                    normal, d = normalize_plane(n_arr, value[dk])
                    print(f"[Real Table] 使用来源: {path}: {nk} + {dk}")
                    print(f"[Real Table] normal={normal.tolist()}, d={d:.8f}")
                    return normal, d

    # 2. 找长度为 4 的 plane 参数
    candidates = []
    for path, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 4:
            lower = path.lower()
            score = 0
            if "plane" in lower:
                score += 10
            if "model" in lower:
                score += 5
            if "coeff" in lower or "coef" in lower:
                score += 5
            if "table" in lower:
                score += 5
            candidates.append((score, path, arr))

    if candidates:
        candidates = sorted(candidates, key=lambda x: -x[0])
        _, path, arr = candidates[0]
        normal, d = normalize_plane(arr[:3], arr[3])
        print(f"[Real Table] 使用来源: {path} = {arr}")
        print(f"[Real Table] normal={normal.tolist()}, d={d:.8f}")
        return normal, d

    print(json.dumps(data, indent=2, ensure_ascii=False))
    raise RuntimeError("无法识别 table_plane.json 格式")


def orient_real_table_normal_to_object_side(points, normal, d):
    """
    真实桌面法向有正负歧义。
    这里让真实鸭子主体位于 signed distance 为正的一侧。
    """
    points = np.asarray(points, dtype=np.float64)
    normal = normalize(normal)
    h = points @ normal + float(d)

    p05 = float(np.percentile(h, 5))
    p50 = float(np.percentile(h, 50))
    p95 = float(np.percentile(h, 95))

    # 鸭子应该在桌面一侧，通常 p95 的绝对值应该更大。
    # 如果负侧更大，则翻转。
    if abs(p05) > abs(p95):
        normal = -normal
        d = -d
        h = -h
        print("[Real Table] 法向已翻转，使真实鸭子主体位于正方向。")
    else:
        print("[Real Table] 法向保持不变。")

    print(
        f"[Real Table] signed height percentiles: "
        f"p05={np.percentile(h,5):.6f}, "
        f"p50={np.percentile(h,50):.6f}, "
        f"p95={np.percentile(h,95):.6f}"
    )

    return normal, float(d)


# ============================================================
# 读取 single duck 坐标系中的桌面法向
# ============================================================

def load_single_table_normal(json_path):
    """
    读取 yazishuzhizhousuanfa.py 输出的：
      table_normal_in_single_duck_frame.json

    主要读取：
      normal_table_single
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 single-table-normal json: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidate_keys = [
        "normal_table_single",
        "normal_single",
        "table_normal_single",
        "normal",
        "n_single",
    ]

    for k in candidate_keys:
        if k in data:
            arr = as_float_list(data[k])
            if arr is not None and len(arr) == 3:
                n = normalize(arr)
                print(f"[Single Table Normal] 使用 key: {k}")
                print(f"[Single Table Normal] normal={n.tolist()}")
                return n

    # 递归查找带 single / normal 的 3 维向量
    candidates = []
    for path, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 3:
            lower = path.lower()
            score = 0
            if "single" in lower:
                score += 10
            if "normal" in lower:
                score += 10
            if "table" in lower:
                score += 5
            candidates.append((score, path, arr))

    if candidates:
        candidates = sorted(candidates, key=lambda x: -x[0])
        _, path, arr = candidates[0]
        n = normalize(arr)
        print(f"[Single Table Normal] 使用来源: {path}")
        print(f"[Single Table Normal] normal={n.tolist()}")
        return n

    print(json.dumps(data, indent=2, ensure_ascii=False))
    raise RuntimeError("无法从 json 中读取 normal_table_single")


# ============================================================
# 真实 partial 中过滤桌面
# ============================================================

def split_real_duck_and_table(points, n_real, d_real, above_thresh=0.008, table_thresh=0.006, no_filter=False):
    """
    如果输入 real partial 带桌面，则根据真实桌面平面过滤出鸭子点。
    如果输入本来就是鸭子 partial，也基本会保留大部分桌面上方点。

    h = n_real dot x + d_real
      table: |h| < table_thresh
      duck : h > above_thresh
    """
    points = np.asarray(points, dtype=np.float64)
    h = points @ n_real + float(d_real)

    table_mask = np.abs(h) < float(table_thresh)
    table_pts = points[table_mask]

    if no_filter:
        duck_pts = points
        print("[Real Split] --no-filter-real-table 已开启，不过滤真实点云中的桌面。")
    else:
        duck_mask = h > float(above_thresh)
        duck_pts = points[duck_mask]

        if len(duck_pts) < 200:
            print("[WARN] 根据桌面过滤后鸭子点太少，自动退回使用原始 real partial。")
            duck_pts = points

    print("\n[Real Split]")
    print(f"  all real points = {len(points)}")
    print(f"  table points    = {len(table_pts)}")
    print(f"  duck candidate  = {len(duck_pts)}")
    print(f"  above_thresh    = {above_thresh}")
    print(f"  table_thresh    = {table_thresh}")

    return duck_pts, table_pts


# ============================================================
# 配准评分
# ============================================================

def rmse_trimmed_from_distances(dists, trim_ratio):
    dists = np.asarray(dists, dtype=np.float64)
    dists = dists[np.isfinite(dists)]
    if len(dists) == 0:
        return 1e9

    keep_n = max(10, int(float(trim_ratio) * len(dists)))
    keep_n = min(keep_n, len(dists))
    keep = np.partition(dists, keep_n - 1)[:keep_n]
    return float(np.sqrt(np.mean(keep ** 2)))


def score_alignment(src_tf, tgt, n_real, d_real, p2s_trim=0.85, s2p_trim=0.25, coverage_thresh=0.015):
    """
    src_tf:
      变换后的完整 SAM3D 鸭子

    tgt:
      真实残缺鸭子

    使用两个方向：
      partial -> full 作为主项
      full -> partial 只取很小 trim 作为辅助项

    因为真实 partial 是完整鸭子的子集，不能用完整 source 全量惩罚。
    """
    src_tf = np.asarray(src_tf, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)

    tree_src = cKDTree(src_tf)
    d_p2s, _ = tree_src.query(tgt, k=1, workers=-1)

    tree_tgt = cKDTree(tgt)
    d_s2p, _ = tree_tgt.query(src_tf, k=1, workers=-1)

    rmse_p2s = rmse_trimmed_from_distances(d_p2s, p2s_trim)
    rmse_s2p = rmse_trimmed_from_distances(d_s2p, s2p_trim)

    coverage = float(np.mean(d_p2s < coverage_thresh))

    # 桌面约束检查：底部应该接近桌面，且不要严重穿桌
    h = src_tf @ n_real + float(d_real)
    bottom_h = float(np.percentile(h, 1.0))
    median_h = float(np.percentile(h, 50.0))

    penetration = max(0.0, -bottom_h)

    # 主目标：partial 到 full 贴合
    # 辅助目标：少量 full 到 partial 贴合
    # 桌面穿模惩罚
    objective = rmse_p2s + 0.25 * rmse_s2p + 2.0 * penetration

    return {
        "objective": float(objective),
        "rmse_p2s": float(rmse_p2s),
        "rmse_s2p": float(rmse_s2p),
        "coverage": float(coverage),
        "bottom_h": float(bottom_h),
        "median_h": float(median_h),
        "penetration": float(penetration),
    }


def refine_inplane_translation(src_base, tgt, n_real, init_t, iters=8, trim_ratio=0.80):
    """
    固定旋转和尺度，只优化桌面平面内的平移 tx/ty。

    src_base:
      已经 scale + rotation + normal-height 处理后的点。
      还没有加平面内平移。

    init_t:
      初始平移，通常是平面内 centroid 对齐量。
    """
    t = np.asarray(init_t, dtype=np.float64).reshape(3).copy()
    n_real = normalize(n_real)

    for _ in range(int(iters)):
        src_tf = src_base + t.reshape(1, 3)
        tree_src = cKDTree(src_tf)
        dists, idx = tree_src.query(tgt, k=1, workers=-1)

        keep_n = max(50, int(trim_ratio * len(tgt)))
        keep_n = min(keep_n, len(tgt))
        order = np.argsort(dists)
        keep = order[:keep_n]

        matched_src = src_tf[idx[keep]]
        matched_tgt = tgt[keep]

        delta = matched_tgt - matched_src
        step = np.median(delta, axis=0)
        step = project_to_plane_vec(step, n_real)

        if np.linalg.norm(step) < 1e-7:
            break

        t = t + step

    return t


# ============================================================
# 主搜索
# ============================================================

def generate_scales(base_scale, scale_min_mult, scale_max_mult, steps):
    if steps <= 1:
        return np.array([base_scale], dtype=np.float64)
    return base_scale * np.linspace(float(scale_min_mult), float(scale_max_mult), int(steps))


def run_search(
    single_points,
    target_points,
    n_single,
    n_real,
    d_real,
    args,
):
    """
    搜索：
      sign of n_single
      scale
      yaw around n_real
      in-plane translation

    返回最优变换：
      p_real = scale * R_total @ p_single + t_total
    """
    src_full = np.asarray(single_points, dtype=np.float64)
    tgt_full = np.asarray(target_points, dtype=np.float64)

    src = downsample_points(src_full, max_points=args.search_source_points, seed=args.seed)
    tgt = downsample_points(tgt_full, max_points=args.search_target_points, seed=args.seed + 1)

    src_diag = robust_bbox_diag(src)
    tgt_diag = robust_bbox_diag(tgt)

    base_scale = tgt_diag / max(src_diag, 1e-12)
    if args.fixed_scale > 0:
        base_scale = float(args.fixed_scale)

    print("\n" + "=" * 80)
    print("[Search] 桌面竖直轴约束配准")
    print(f"  source points = {len(src)}")
    print(f"  target points = {len(tgt)}")
    print(f"  source diag   = {src_diag:.6f}")
    print(f"  target diag   = {tgt_diag:.6f}")
    print(f"  base scale    = {base_scale:.6f}")
    print("=" * 80)

    # coverage 阈值随真实物体尺寸自适应
    coverage_thresh = float(args.coverage_thresh_ratio * tgt_diag)
    print(f"[Search] coverage_thresh = {coverage_thresh:.6f}")

    scales = generate_scales(
        base_scale=base_scale,
        scale_min_mult=args.scale_min_mult,
        scale_max_mult=args.scale_max_mult,
        steps=args.scale_steps,
    )

    yaw_angles = np.arange(0.0, 360.0, float(args.yaw_step_deg), dtype=np.float64)

    target_centroid = tgt.mean(axis=0)

    best = None
    total = 0

    for sign in [1.0, -1.0]:
        n_single_use = sign * normalize(n_single)
        R_align = rotation_from_a_to_b(n_single_use, n_real)

        for scale in scales:
            for yaw_deg in yaw_angles:
                total += 1

                R_yaw = rotation_about_axis(n_real, np.deg2rad(yaw_deg))
                R_total = R_yaw @ R_align

                src_rs = float(scale) * (src @ R_total.T)

                # 高度贴桌面：
                # table plane: n_real dot x + d_real = 0
                # 让 source 的底部 1% 分位贴近桌面
                bottom_dot = float(np.percentile(src_rs @ n_real, args.bottom_percentile))
                t_normal = (-float(d_real) + float(args.table_offset) - bottom_dot) * n_real

                src_h = src_rs + t_normal.reshape(1, 3)

                # 平面内 centroid 初始化
                src_centroid = src_h.mean(axis=0)
                diff = target_centroid - src_centroid
                t_plane0 = project_to_plane_vec(diff, n_real)

                # 固定姿态和尺度，只优化平面内平移
                t_plane = refine_inplane_translation(
                    src_base=src_h,
                    tgt=tgt,
                    n_real=n_real,
                    init_t=t_plane0,
                    iters=args.translation_refine_iters,
                    trim_ratio=args.translation_refine_trim,
                )

                t_total = t_normal + t_plane
                src_tf = src_rs + t_total.reshape(1, 3)

                sc = score_alignment(
                    src_tf=src_tf,
                    tgt=tgt,
                    n_real=n_real,
                    d_real=d_real,
                    p2s_trim=args.p2s_trim,
                    s2p_trim=args.s2p_trim,
                    coverage_thresh=coverage_thresh,
                )

                rec = {
                    "normal_sign": sign,
                    "scale": float(scale),
                    "yaw_deg": float(yaw_deg),
                    "R_align": R_align,
                    "R_yaw": R_yaw,
                    "R_total": R_total,
                    "t_total": t_total,
                    **sc,
                }

                if best is None or rec["objective"] < best["objective"]:
                    best = rec
                    print(
                        f"[Best coarse] "
                        f"obj={best['objective']:.6f}, "
                        f"p2s={best['rmse_p2s']:.6f}, "
                        f"s2p={best['rmse_s2p']:.6f}, "
                        f"cov={best['coverage']:.4f}, "
                        f"scale={best['scale']:.6f}, "
                        f"yaw={best['yaw_deg']:.2f}, "
                        f"sign={best['normal_sign']:+.0f}, "
                        f"bottom_h={best['bottom_h']:.6f}"
                    )

    print(f"[Search] coarse candidates = {total}")

    # ------------------------------------------------------------
    # Fine search around best yaw / scale
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Fine Search] 在 coarse 最优附近细化 yaw / scale")
    print("=" * 80)

    fine_best = best.copy()

    best_yaw = float(best["yaw_deg"])
    best_scale = float(best["scale"])
    best_sign = float(best["normal_sign"])

    if args.fixed_scale > 0:
        fine_scales = np.array([best_scale], dtype=np.float64)
    else:
        fine_scales = best_scale * np.linspace(args.fine_scale_min_mult, args.fine_scale_max_mult, args.fine_scale_steps)

    fine_yaws = np.arange(
        best_yaw - args.yaw_step_deg,
        best_yaw + args.yaw_step_deg + 1e-9,
        args.fine_yaw_step_deg,
        dtype=np.float64,
    )

    R_align = rotation_from_a_to_b(best_sign * normalize(n_single), n_real)

    fine_total = 0

    for scale in fine_scales:
        for yaw_deg in fine_yaws:
            fine_total += 1
            yaw_wrapped = yaw_deg % 360.0

            R_yaw = rotation_about_axis(n_real, np.deg2rad(yaw_wrapped))
            R_total = R_yaw @ R_align

            src_rs = float(scale) * (src @ R_total.T)

            bottom_dot = float(np.percentile(src_rs @ n_real, args.bottom_percentile))
            t_normal = (-float(d_real) + float(args.table_offset) - bottom_dot) * n_real

            src_h = src_rs + t_normal.reshape(1, 3)

            src_centroid = src_h.mean(axis=0)
            diff = target_centroid - src_centroid
            t_plane0 = project_to_plane_vec(diff, n_real)

            t_plane = refine_inplane_translation(
                src_base=src_h,
                tgt=tgt,
                n_real=n_real,
                init_t=t_plane0,
                iters=args.translation_refine_iters + 4,
                trim_ratio=args.translation_refine_trim,
            )

            t_total = t_normal + t_plane
            src_tf = src_rs + t_total.reshape(1, 3)

            sc = score_alignment(
                src_tf=src_tf,
                tgt=tgt,
                n_real=n_real,
                d_real=d_real,
                p2s_trim=args.p2s_trim,
                s2p_trim=args.s2p_trim,
                coverage_thresh=coverage_thresh,
            )

            rec = {
                "normal_sign": best_sign,
                "scale": float(scale),
                "yaw_deg": float(yaw_wrapped),
                "R_align": R_align,
                "R_yaw": R_yaw,
                "R_total": R_total,
                "t_total": t_total,
                **sc,
            }

            if rec["objective"] < fine_best["objective"]:
                fine_best = rec
                print(
                    f"[Best fine] "
                    f"obj={fine_best['objective']:.6f}, "
                    f"p2s={fine_best['rmse_p2s']:.6f}, "
                    f"s2p={fine_best['rmse_s2p']:.6f}, "
                    f"cov={fine_best['coverage']:.4f}, "
                    f"scale={fine_best['scale']:.6f}, "
                    f"yaw={fine_best['yaw_deg']:.2f}, "
                    f"bottom_h={fine_best['bottom_h']:.6f}"
                )

    print(f"[Fine Search] candidates = {fine_total}")

    return fine_best


# ============================================================
# 可视化输出
# ============================================================

def save_compare(aligned_sam, real_duck, real_table, path):
    """
    蓝色：aligned SAM3D 完整鸭子
    红色：真实残缺鸭子
    绿色：真实桌面点
    """
    p1 = np_to_pcd(aligned_sam, color=[0.1, 0.35, 1.0])
    p2 = np_to_pcd(real_duck, color=[1.0, 0.1, 0.05])
    p3 = np_to_pcd(real_table, color=[0.1, 0.9, 0.2])

    combined = p1 + p2 + p3
    ok = o3d.io.write_point_cloud(path, combined)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}")


def make_table_plane_vis(n, d, center_hint, size=0.35, grid=60):
    """
    生成一个真实桌面平面可视化点云。
    """
    n = normalize(n)
    center_hint = np.asarray(center_hint, dtype=np.float64)

    # 把 center_hint 投影到平面
    center = center_hint - (np.dot(n, center_hint) + float(d)) * n

    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = normalize(np.cross(n, tmp))
    v = normalize(np.cross(n, u))

    vals = np.linspace(-size / 2, size / 2, int(grid))
    pts = []
    for a in vals:
        for b in vals:
            pts.append(center + a * u + b * v)

    return np.asarray(pts, dtype=np.float64)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sam3d-single", required=True, help="单独 SAM3D 鸭子点云/mesh，例如 ./sam3d_duck_clean.ply")
    parser.add_argument("--real-partial", required=True, help="真实残缺鸭子点云；带桌面也可以")
    parser.add_argument("--real-table-plane", required=True, help="真实 table_plane.json")
    parser.add_argument("--single-table-normal", required=True, help="table_normal_in_single_duck_frame.json")
    parser.add_argument("--out-dir", default="./output_table_axis_constrained_align")

    parser.add_argument("--sample-points", type=int, default=180000)

    # 真实点云中过滤桌面
    parser.add_argument("--real-duck-above-thresh", type=float, default=0.008, help="真实点云中，高于桌面多少米认为是鸭子")
    parser.add_argument("--real-table-thresh", type=float, default=0.006, help="真实点云中，离桌面多少米以内认为是桌面")
    parser.add_argument("--no-filter-real-table", action="store_true", help="不从真实 partial 中过滤桌面")

    # 搜索点数
    parser.add_argument("--search-source-points", type=int, default=35000)
    parser.add_argument("--search-target-points", type=int, default=25000)

    # scale 搜索
    parser.add_argument("--fixed-scale", type=float, default=-1.0, help="指定固定 scale；默认自动按 bbox 估计并搜索")
    parser.add_argument("--scale-min-mult", type=float, default=0.70)
    parser.add_argument("--scale-max-mult", type=float, default=1.30)
    parser.add_argument("--scale-steps", type=int, default=11)

    # fine scale
    parser.add_argument("--fine-scale-min-mult", type=float, default=0.92)
    parser.add_argument("--fine-scale-max-mult", type=float, default=1.08)
    parser.add_argument("--fine-scale-steps", type=int, default=9)

    # yaw 搜索
    parser.add_argument("--yaw-step-deg", type=float, default=10.0)
    parser.add_argument("--fine-yaw-step-deg", type=float, default=2.0)

    # 高度贴桌面
    parser.add_argument("--bottom-percentile", type=float, default=1.0)
    parser.add_argument("--table-offset", type=float, default=0.0, help="让 SAM3D 底部高于桌面的偏移，单位同真实点云，一般 0 或 0.002")

    # 平面内平移优化
    parser.add_argument("--translation-refine-iters", type=int, default=8)
    parser.add_argument("--translation-refine-trim", type=float, default=0.80)

    # scoring
    parser.add_argument("--p2s-trim", type=float, default=0.85, help="partial->source 的 trimmed ratio")
    parser.add_argument("--s2p-trim", type=float, default=0.25, help="source->partial 的 trimmed ratio")
    parser.add_argument("--coverage-thresh-ratio", type=float, default=0.04)

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 桌面竖直轴约束配准：single SAM3D duck -> real partial duck")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. 读取数据
    # ------------------------------------------------------------
    sam_pcd = geometry_to_pcd(args.sam3d_single, sample_points=args.sample_points)
    real_pcd = geometry_to_pcd(args.real_partial, sample_points=args.sample_points)

    sam_points = get_points(sam_pcd)
    real_points = get_points(real_pcd)

    print(f"[SAM3D single] points={len(sam_points)}")
    print(f"[Real partial]  points={len(real_points)}")

    # ------------------------------------------------------------
    # 2. 读取真实桌面法向和平面
    # ------------------------------------------------------------
    n_real, d_real = load_table_plane(args.real_table_plane)
    n_real, d_real = orient_real_table_normal_to_object_side(real_points, n_real, d_real)

    # ------------------------------------------------------------
    # 3. 读取 single duck 坐标系下桌面法向
    # ------------------------------------------------------------
    n_single = load_single_table_normal(args.single_table_normal)

    # ------------------------------------------------------------
    # 4. 从真实 partial 中分离鸭子候选和桌面候选
    # ------------------------------------------------------------
    real_duck_points, real_table_points = split_real_duck_and_table(
        real_points,
        n_real=n_real,
        d_real=d_real,
        above_thresh=args.real_duck_above_thresh,
        table_thresh=args.real_table_thresh,
        no_filter=args.no_filter_real_table,
    )

    real_duck_path = os.path.join(args.out_dir, "real_duck_candidate.ply")
    real_table_path = os.path.join(args.out_dir, "real_table_points.ply")

    save_pcd(real_duck_points, real_duck_path, color=[1.0, 0.1, 0.05])
    save_pcd(real_table_points, real_table_path, color=[0.1, 0.9, 0.2])

    # ------------------------------------------------------------
    # 5. 搜索 constrained transform
    # ------------------------------------------------------------
    best = run_search(
        single_points=sam_points,
        target_points=real_duck_points,
        n_single=n_single,
        n_real=n_real,
        d_real=d_real,
        args=args,
    )

    scale = float(best["scale"])
    R_total = np.asarray(best["R_total"], dtype=np.float64)
    R_align = np.asarray(best["R_align"], dtype=np.float64)
    R_yaw = np.asarray(best["R_yaw"], dtype=np.float64)
    t_total = np.asarray(best["t_total"], dtype=np.float64)

    aligned = apply_transform(sam_points, scale, R_total, t_total)

    # ------------------------------------------------------------
    # 6. 保存结果
    # ------------------------------------------------------------
    aligned_path = os.path.join(args.out_dir, "aligned_sam3d_duck_to_real.ply")
    compare_path = os.path.join(args.out_dir, "compare_aligned_sam3d_real.ply")
    plane_vis_path = os.path.join(args.out_dir, "real_table_plane_vis.ply")
    transform_json = os.path.join(args.out_dir, "table_axis_constrained_transform.json")

    save_pcd(aligned, aligned_path, color=[0.1, 0.35, 1.0])

    # 如果真实 table 点太少，生成一个平面可视化
    if len(real_table_points) < 100:
        table_vis = make_table_plane_vis(
            n_real,
            d_real,
            center_hint=real_duck_points.mean(axis=0),
            size=max(0.25, robust_bbox_diag(real_duck_points) * 1.5),
            grid=70,
        )
        real_table_for_compare = table_vis
    else:
        real_table_for_compare = real_table_points

    save_pcd(real_table_for_compare, plane_vis_path, color=[0.1, 0.9, 0.2])
    save_compare(aligned, real_duck_points, real_table_for_compare, compare_path)

    # 检查对齐后 n_single 是否对上 n_real
    mapped_normal = normalize(R_total @ (best["normal_sign"] * normalize(n_single)))
    normal_angle = float(np.degrees(np.arccos(np.clip(np.dot(mapped_normal, n_real), -1.0, 1.0))))

    info = {
        "definition": "p_real = scale * R_total @ p_single + t_total",
        "source_sam3d_single": os.path.abspath(args.sam3d_single),
        "source_real_partial": os.path.abspath(args.real_partial),
        "real_table_plane_json": os.path.abspath(args.real_table_plane),
        "single_table_normal_json": os.path.abspath(args.single_table_normal),

        "scale": scale,
        "R_total_single_to_real": R_total.tolist(),
        "R_align_single_table_normal_to_real_table_normal": R_align.tolist(),
        "R_yaw_about_real_table_normal": R_yaw.tolist(),
        "t_total_single_to_real": t_total.tolist(),

        "normal_table_single_original": normalize(n_single).tolist(),
        "normal_sign_used": float(best["normal_sign"]),
        "normal_table_real": normalize(n_real).tolist(),
        "d_table_real": float(d_real),
        "mapped_single_normal_in_real": mapped_normal.tolist(),
        "normal_alignment_error_deg": normal_angle,

        "yaw_deg": float(best["yaw_deg"]),
        "objective": float(best["objective"]),
        "rmse_partial_to_sam3d": float(best["rmse_p2s"]),
        "rmse_sam3d_to_partial_trimmed": float(best["rmse_s2p"]),
        "coverage": float(best["coverage"]),
        "bottom_h_after_align": float(best["bottom_h"]),
        "median_h_after_align": float(best["median_h"]),
        "penetration": float(best["penetration"]),

        "outputs": {
            "aligned_sam3d_duck_to_real": aligned_path,
            "real_duck_candidate": real_duck_path,
            "real_table_points": real_table_path,
            "real_table_plane_vis": plane_vis_path,
            "compare": compare_path,
        },

        "notes": [
            "Blue in compare file = aligned single SAM3D duck.",
            "Red in compare file = real partial duck candidate.",
            "Green in compare file = real table points or table plane visualization.",
            "This registration locks roll/pitch using table axis, then searches yaw, in-plane translation, and scale.",
        ],
    }

    with open(transform_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "█" * 80)
    print("✅ 完成：桌面竖直轴约束配准")
    print("重点输出：")
    print(f"  1. 配准后的完整 SAM3D 鸭子 : {os.path.abspath(aligned_path)}")
    print(f"  2. 真实鸭子候选点云         : {os.path.abspath(real_duck_path)}")
    print(f"  3. 真实桌面点 / 平面        : {os.path.abspath(plane_vis_path)}")
    print(f"  4. 对比可视化               : {os.path.abspath(compare_path)}")
    print(f"  5. 变换参数 JSON            : {os.path.abspath(transform_json)}")
    print("█" * 80)

    print("\n颜色说明 compare_aligned_sam3d_real.ply：")
    print("  蓝色 = 配准后的单独 SAM3D 完整鸭子")
    print("  红色 = 真实残缺鸭子")
    print("  绿色 = 真实桌面点 / 桌面平面")

    print("\n关键指标：")
    print(f"  normal_alignment_error_deg = {normal_angle:.4f} deg")
    print(f"  yaw_deg                    = {best['yaw_deg']:.2f}")
    print(f"  scale                      = {scale:.6f}")
    print(f"  rmse_partial_to_sam3d       = {best['rmse_p2s']:.6f}")
    print(f"  coverage                   = {best['coverage']:.4f}")
    print(f"  bottom_h_after_align        = {best['bottom_h']:.6f}")
    print(f"  penetration                = {best['penetration']:.6f}")


if __name__ == "__main__":
    main()