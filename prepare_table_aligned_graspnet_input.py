#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_table_aligned_graspnet_input.py

作用：
  把 pipeline4 输出的 RGB-D 坐标系完整物体点云，
  转到 table-aligned frame：

    - table plane -> z = 0
    - table normal -> +Z
    - 物体看起来平放在桌面上

输出：
  1. sam3d_pure_table_aligned.ply
  2. contact_graspnet_input_table_aligned.npy
  3. contact_graspnet_input_table_aligned.npz
  4. rgbd_to_table_aligned.json

注意：
  为避免 Open3D 写 PLY 时 segfault，本脚本只用 Open3D 读取点云；
  写 PLY 使用纯 Python ASCII PLY。
"""

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


def load_points(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"[Load] {path}")

    # 先按 point cloud 读
    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(f"[Load pcd] points={len(pts)}")
        return pts

    # 再按 mesh 读
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"[Load mesh] vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=180000)
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(f"[Sample mesh] points={len(pts)}")
        return pts

    raise RuntimeError(f"无法读取点云/mesh: {path}")


def write_ascii_ply(points, path, color=None):
    """
    纯 Python 写 ASCII PLY，避免 Open3D write_point_cloud segfault。
    """
    points = np.asarray(points, dtype=np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if color is None:
        has_color = False
    else:
        has_color = True
        color = np.asarray(color, dtype=np.float64).reshape(3)
        color = np.clip(color, 0, 1)
        rgb = (color * 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if has_color:
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            for p in points:
                f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {r} {g} {b}\n")
        else:
            for p in points:
                f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

    print(f"[Save PLY ASCII] {path}  points={len(points)}")


def load_plane(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arr = None

    if "plane_model" in data:
        arr = data["plane_model"]
    elif "plane" in data:
        arr = data["plane"]
    elif "coefficients" in data:
        arr = data["coefficients"]

    if arr is not None and len(arr) == 4:
        a, b, c, d = arr
        n = np.asarray([a, b, c], dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise RuntimeError("平面法向长度为 0")
        return n / norm, float(d) / norm, data

    if "normal" in data and "d" in data:
        n = np.asarray(data["normal"], dtype=np.float64)
        d = float(data["d"])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise RuntimeError("平面法向长度为 0")
        return n / norm, d / norm, data

    raise RuntimeError("table_plane.json 里找不到 plane_model / plane / coefficients / normal+d")


def rotation_align_vec_to_vec(a, b):
    """
    返回 R，使 R @ a = b
    """
    a = normalize(a)
    b = normalize(b)

    c = float(np.dot(a, b))

    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-10:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, aux)) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = normalize(np.cross(a, aux))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3) + 2.0 * (K @ K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2 + 1e-12))

    # 正交化一下
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    return R


def downsample(points, n, seed=0):
    points = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]

    if n <= 0 or len(points) <= n:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=int(n), replace=False)
    return points[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--table-plane-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--npy-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pts_rgbd = load_points(args.input_ply)
    n_rgbd, d_rgbd, plane_raw = load_plane(args.table_plane_json)

    # 让物体位于桌面法向正侧
    h = pts_rgbd @ n_rgbd + d_rgbd
    if np.median(h) < 0:
        n_rgbd = -n_rgbd
        d_rgbd = -d_rgbd
        h = -h
        print("[Plane] flip normal so object is above table")

    print(f"[Plane] normal_rgbd = {n_rgbd.tolist()}")
    print(f"[Plane] d_rgbd      = {d_rgbd:.8f}")
    print(
        f"[Height before align] "
        f"min={np.min(h):.6f}, "
        f"p1={np.percentile(h, 1):.6f}, "
        f"p50={np.percentile(h, 50):.6f}, "
        f"p99={np.percentile(h, 99):.6f}"
    )

    # 让桌面法向对齐 +Z
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R = rotation_align_vec_to_vec(n_rgbd, z_axis)

    # 桌面上一点：n^T x + d = 0
    plane_origin_rgbd = -d_rgbd * n_rgbd

    # p_table = R @ (p_rgbd - plane_origin)
    pts_table = (R @ (pts_rgbd - plane_origin_rgbd.reshape(1, 3)).T).T

    # 数值上让物体底部 p1 对齐 z=0
    z_floor = float(np.percentile(pts_table[:, 2], 1.0))
    pts_table[:, 2] -= z_floor

    h2 = pts_table[:, 2]
    print(
        f"[Height after align] "
        f"min={np.min(h2):.6f}, "
        f"p1={np.percentile(h2, 1):.6f}, "
        f"p50={np.percentile(h2, 50):.6f}, "
        f"p99={np.percentile(h2, 99):.6f}"
    )

    # 保存 table-aligned PLY
    ply_out = os.path.join(args.out_dir, "sam3d_pure_table_aligned.ply")
    write_ascii_ply(pts_table, ply_out, color=[0.2, 0.5, 1.0])

    # 保存 CGN 输入
    pts_cgn = downsample(pts_table.astype(np.float32), args.npy_points, seed=args.seed)

    npy_out = os.path.join(args.out_dir, "contact_graspnet_input_table_aligned.npy")
    npz_out = os.path.join(args.out_dir, "contact_graspnet_input_table_aligned.npz")

    np.save(npy_out, pts_cgn.astype(np.float32))
    np.savez(npz_out, pc=pts_cgn.astype(np.float32))

    print(f"[Save NPY] {npy_out} points={len(pts_cgn)}")
    print(f"[Save NPZ] {npz_out}")

    # 等价变换：
    # pts_table = R @ pts_rgbd + t
    #
    # 原先：
    # pts_table = R @ (pts_rgbd - plane_origin) - [0,0,z_floor]
    #           = R @ pts_rgbd + (-R @ plane_origin + [0,0,-z_floor])
    t = -R @ plane_origin_rgbd + np.array([0.0, 0.0, -z_floor], dtype=np.float64)

    T_rgbd_to_table = np.eye(4, dtype=np.float64)
    T_rgbd_to_table[:3, :3] = R
    T_rgbd_to_table[:3, 3] = t

    T_table_to_rgbd = np.eye(4, dtype=np.float64)
    T_table_to_rgbd[:3, :3] = R.T
    T_table_to_rgbd[:3, 3] = -R.T @ t

    meta = {
        "definition": "p_table = R_rgbd_to_table @ p_rgbd + t_rgbd_to_table",
        "input_ply": os.path.abspath(args.input_ply),
        "table_plane_json": os.path.abspath(args.table_plane_json),

        "normal_rgbd": n_rgbd.tolist(),
        "d_rgbd": float(d_rgbd),
        "plane_origin_rgbd": plane_origin_rgbd.tolist(),
        "z_floor_shift": float(z_floor),

        "R_rgbd_to_table_aligned": R.tolist(),
        "t_rgbd_to_table_aligned": t.tolist(),
        "T_rgbd_to_table_aligned": T_rgbd_to_table.tolist(),
        "T_table_aligned_to_rgbd": T_table_to_rgbd.tolist(),

        "outputs": {
            "ply_table_aligned": os.path.abspath(ply_out),
            "npy_contact_graspnet": os.path.abspath(npy_out),
            "npz_contact_graspnet": os.path.abspath(npz_out),
        }
    }

    json_out = os.path.join(args.out_dir, "rgbd_to_table_aligned.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Save JSON] {json_out}")

    print("\n✅ 完成：桌面对齐点云和 Contact-GraspNet 输入已生成。")
    print(f"  table-aligned ply : {ply_out}")
    print(f"  Contact-GraspNet npy: {npy_out}")
    print(f"  transform json: {json_out}")


if __name__ == "__main__":
    main()