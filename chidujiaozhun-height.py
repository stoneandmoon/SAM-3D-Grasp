#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def skew(v):
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ], dtype=np.float64)


def rotation_from_a_to_b(a, b):
    a = normalize(a)
    b = normalize(b)
    c = float(np.dot(a, b))

    if c > 1.0 - 1e-10:
        return np.eye(3)

    if c < -1.0 + 1e-10:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, tmp))
        K = skew(axis)
        return np.eye(3) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = skew(v)
    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2))
    return R


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
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 优先找 plane_model = [a,b,c,d]
    for key in ["plane_model", "plane", "coefficients"]:
        if key in data:
            arr = as_float_list(data[key])
            if arr is not None and len(arr) == 4:
                n = normalize(arr[:3])
                d = float(arr[3]) / max(np.linalg.norm(arr[:3]), 1e-12)
                return n, d

    # 找 normal + d
    for path2, value in recursive_items(data):
        if not isinstance(value, dict):
            continue
        normal = None
        d = None
        for nk in ["normal", "plane_normal", "normal_camera", "table_normal"]:
            if nk in value:
                arr = as_float_list(value[nk])
                if arr is not None and len(arr) == 3:
                    normal = arr
                    break
        for dk in ["d", "plane_d", "d_camera", "table_d", "offset"]:
            if dk in value and isinstance(value[dk], (int, float)):
                d = float(value[dk])
                break
        if normal is not None and d is not None:
            n = normalize(normal)
            d = d / max(np.linalg.norm(normal), 1e-12)
            return n, d

    # 递归找长度为 4 的数组
    candidates = []
    for p, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 4:
            score = 0
            lower = p.lower()
            if "plane" in lower:
                score += 10
            if "table" in lower:
                score += 5
            candidates.append((score, p, arr))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        arr = candidates[0][2]
        n = normalize(arr[:3])
        d = float(arr[3]) / max(np.linalg.norm(arr[:3]), 1e-12)
        return n, d

    raise RuntimeError(f"无法读取真实桌面平面: {path}")


def load_single_table_normal(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ["normal_table_single", "normal_single", "table_normal_single", "normal"]:
        if key in data:
            arr = as_float_list(data[key])
            if arr is not None and len(arr) == 3:
                return normalize(arr)

    for p, value in recursive_items(data):
        arr = as_float_list(value)
        if arr is not None and len(arr) == 3:
            lower = p.lower()
            if "normal" in lower and ("single" in lower or "table" in lower):
                return normalize(arr)

    raise RuntimeError(f"无法读取 normal_table_single: {path}")


def geometry_to_points(path, sample_points=180000):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        return np.asarray(pcd.points, dtype=np.float64)

    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"无法读取点云/mesh: {path}")
    return np.asarray(pcd.points, dtype=np.float64)


def orient_real_normal_to_object_side(real_points, n, d):
    h = real_points @ n + d
    p05 = np.percentile(h, 5)
    p95 = np.percentile(h, 95)
    if abs(p05) > abs(p95):
        n = -n
        d = -d
        h = -h
    return n, d, h


def robust_extent(vals, low=1.0, high=99.0):
    return float(np.percentile(vals, high) - np.percentile(vals, low))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d-single", required=True)
    parser.add_argument("--real-partial", required=True)
    parser.add_argument("--real-table-plane", required=True)
    parser.add_argument("--single-table-normal", required=True)
    parser.add_argument("--sample-points", type=int, default=180000)

    parser.add_argument("--real-height-percentile", type=float, default=98.0)
    parser.add_argument("--sam-bottom-percentile", type=float, default=1.0)
    parser.add_argument("--sam-top-percentile", type=float, default=99.0)

    args = parser.parse_args()

    sam = geometry_to_points(args.sam3d_single, sample_points=args.sample_points)
    real = geometry_to_points(args.real_partial, sample_points=args.sample_points)

    n_real, d_real = load_table_plane(args.real_table_plane)
    n_real, d_real, h_real_all = orient_real_normal_to_object_side(real, n_real, d_real)

    n_single = load_single_table_normal(args.single_table_normal)

    print("=" * 80)
    print("[Scale Estimate] 基于桌面高度估计 SAM3D -> real 的 scale")
    print("=" * 80)

    print(f"[Real table normal] {n_real.tolist()}, d={d_real:.8f}")
    print(f"[Single table normal] {n_single.tolist()}")

    # 真实高度：真实 partial 到桌面的高度
    h_real = real @ n_real + d_real
    h_real_pos = h_real[h_real > 0]

    if len(h_real_pos) < 50:
        raise RuntimeError("真实点云在桌面正侧的点太少，检查 table normal 是否反了。")

    real_height = float(np.percentile(h_real_pos, args.real_height_percentile))

    print("\n[Real partial height above table]")
    print(f"  p50 = {np.percentile(h_real_pos, 50):.6f}")
    print(f"  p90 = {np.percentile(h_real_pos, 90):.6f}")
    print(f"  p95 = {np.percentile(h_real_pos, 95):.6f}")
    print(f"  p98 = {np.percentile(h_real_pos, 98):.6f}")
    print(f"  use p{args.real_height_percentile:.1f} = {real_height:.6f}")

    results = []

    for sign in [1.0, -1.0]:
        R_align = rotation_from_a_to_b(sign * n_single, n_real)
        sam_rot = sam @ R_align.T
        h_sam = sam_rot @ n_real

        sam_bottom = float(np.percentile(h_sam, args.sam_bottom_percentile))
        sam_top = float(np.percentile(h_sam, args.sam_top_percentile))
        sam_height = sam_top - sam_bottom

        scale = real_height / max(sam_height, 1e-12)

        results.append((sign, scale, sam_height, sam_bottom, sam_top))

    print("\n[SAM3D height and estimated scale]")
    for sign, scale, sam_height, sam_bottom, sam_top in results:
        print(f"  normal_sign={sign:+.0f}")
        print(f"    sam_bottom_p{args.sam_bottom_percentile:.1f} = {sam_bottom:.6f}")
        print(f"    sam_top_p{args.sam_top_percentile:.1f}       = {sam_top:.6f}")
        print(f"    sam_height                 = {sam_height:.6f}")
        print(f"    estimated_scale             = {scale:.6f}")

    best = results[0]
    print("\n默认建议先试 normal_sign=+1 对应的 scale。")
    print("如果后续配准日志里 normal_sign_used 是 -1，或者可视化翻转，再试 -1 那个 scale。")

    print("\n推荐运行方式示例：")
    print(f"  --fixed-scale {best[1]:.6f}")

    print("=" * 80)


if __name__ == "__main__":
    main()