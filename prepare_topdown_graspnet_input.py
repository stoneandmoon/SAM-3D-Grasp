#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_topdown_graspnet_input.py

把 pipeline4 输出的 RGB-D 坐标系完整物体点云，变换成 Contact-GraspNet 更容易处理的
“虚拟俯视相机坐标系”。

坐标定义：
  +Z_topdown：从物体上方向桌面方向，也就是“从上往下看”
  X/Y：桌面平面内
  桌面：z ≈ camera_height
  物体：z < camera_height，但仍为正

输出：
  topdown_graspnet_input.ply
  topdown_graspnet_input.npy
  topdown_graspnet_input.npz
  rgbd_to_topdown_graspnet.json

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

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(f"[Load pcd] points={len(pts)}")
        return pts

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"[Load mesh] vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=180000)
        pts = np.asarray(pcd.points, dtype=np.float64)
        print(f"[Sample mesh] points={len(pts)}")
        return pts

    raise RuntimeError(f"无法读取点云/mesh: {path}")


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
            raise RuntimeError("table plane normal length is zero")
        return n / norm, float(d) / norm, data

    if "normal" in data and "d" in data:
        n = np.asarray(data["normal"], dtype=np.float64)
        d = float(data["d"])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise RuntimeError("table plane normal length is zero")
        return n / norm, d / norm, data

    raise RuntimeError("table_plane.json 中找不到 plane_model / plane / coefficients / normal+d")


def write_ascii_ply(points, path, color=(0.2, 0.5, 1.0)):
    points = np.asarray(points, dtype=np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    rgb = np.clip(np.asarray(color, dtype=np.float64) * 255, 0, 255).astype(np.uint8)
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {r} {g} {b}\n")

    print(f"[Save PLY ASCII] {path} points={len(points)}")


def downsample(points, n, seed=0):
    points = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]

    if n <= 0 or len(points) <= n:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=int(n), replace=False)
    return points[idx]


def build_topdown_frame(n_up_rgbd, ref_axis=np.array([1.0, 0.0, 0.0])):
    """
    构造虚拟俯视相机坐标系。

    z_topdown = -n_up_rgbd
      表示从物体上方朝桌面方向看。

    x_topdown 取 RGB-D 的 ref_axis 在桌面平面上的投影。
    y_topdown = z_topdown × x_topdown，保证右手系。
    """
    n_up = normalize(n_up_rgbd)
    z_axis = normalize(-n_up)

    ref = np.asarray(ref_axis, dtype=np.float64)
    x_axis = ref - np.dot(ref, z_axis) * z_axis

    if np.linalg.norm(x_axis) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = ref - np.dot(ref, z_axis) * z_axis

    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    # 行向量形式：p_top = R @ p_rgbd + t
    R = np.stack([x_axis, y_axis, z_axis], axis=0)

    return R, x_axis, y_axis, z_axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--table-plane-json", required=True)
    parser.add_argument("--out-dir", required=True)

    # 虚拟相机到桌面的距离。桌面约在 z=camera_height。
    parser.add_argument("--camera-height", type=float, default=0.50)
    parser.add_argument("--npy-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pts_rgbd = load_points(args.input_ply)
    n, d, raw_plane = load_plane(args.table_plane_json)

    # 保证 n 是桌面向上法向：物体在 n 的正侧
    h = pts_rgbd @ n + d
    if np.median(h) < 0:
        n = -n
        d = -d
        h = -h
        print("[Plane] flip normal so object is above table")

    print(f"[Plane] n_up_rgbd = {n.tolist()}")
    print(f"[Plane] d_rgbd    = {d:.8f}")
    print(
        f"[Height above table] "
        f"min={np.min(h):.6f}, "
        f"p1={np.percentile(h, 1):.6f}, "
        f"p50={np.percentile(h, 50):.6f}, "
        f"p99={np.percentile(h, 99):.6f}"
    )

    R, x_axis, y_axis, z_axis = build_topdown_frame(n)

    # 桌面上一点：n^T x + d = 0
    plane_origin_rgbd = -d * n

    # p_top = R @ (p_rgbd - plane_origin) + [0,0,camera_height]
    pts_top = (R @ (pts_rgbd - plane_origin_rgbd.reshape(1, 3)).T).T
    pts_top[:, 2] += float(args.camera_height)

    z = pts_top[:, 2]
    print(
        f"[Topdown z] "
        f"min={z.min():.6f}, "
        f"p1={np.percentile(z, 1):.6f}, "
        f"p50={np.percentile(z, 50):.6f}, "
        f"p99={np.percentile(z, 99):.6f}, "
        f"max={z.max():.6f}"
    )

    out_ply = os.path.join(args.out_dir, "topdown_graspnet_input.ply")
    write_ascii_ply(pts_top, out_ply, color=(0.2, 0.5, 1.0))

    pts_cgn = downsample(pts_top.astype(np.float32), args.npy_points, seed=args.seed)

    out_npy = os.path.join(args.out_dir, "topdown_graspnet_input.npy")
    out_npz = os.path.join(args.out_dir, "topdown_graspnet_input.npz")
    np.save(out_npy, pts_cgn.astype(np.float32))
    np.savez(out_npz, pc=pts_cgn.astype(np.float32))

    print(f"[Save NPY] {out_npy} points={len(pts_cgn)}")
    print(f"[Save NPZ] {out_npz}")

    # 等价变换：
    # p_top = R @ p_rgbd + t
    t = -R @ plane_origin_rgbd + np.array([0.0, 0.0, float(args.camera_height)], dtype=np.float64)

    T_rgbd_to_top = np.eye(4, dtype=np.float64)
    T_rgbd_to_top[:3, :3] = R
    T_rgbd_to_top[:3, 3] = t

    T_top_to_rgbd = np.eye(4, dtype=np.float64)
    T_top_to_rgbd[:3, :3] = R.T
    T_top_to_rgbd[:3, 3] = -R.T @ t

    meta = {
        "definition": "Top-down virtual camera frame: +Z points downward from above the table.",
        "input_ply": os.path.abspath(args.input_ply),
        "table_plane_json": os.path.abspath(args.table_plane_json),

        "camera_height": float(args.camera_height),

        "n_up_rgbd": n.tolist(),
        "d_rgbd": float(d),
        "plane_origin_rgbd": plane_origin_rgbd.tolist(),

        "x_axis_rgbd": x_axis.tolist(),
        "y_axis_rgbd": y_axis.tolist(),
        "z_axis_rgbd_downward": z_axis.tolist(),

        "R_rgbd_to_topdown": R.tolist(),
        "t_rgbd_to_topdown": t.tolist(),
        "T_rgbd_to_topdown": T_rgbd_to_top.tolist(),
        "T_topdown_to_rgbd": T_top_to_rgbd.tolist(),

        "outputs": {
            "ply": os.path.abspath(out_ply),
            "npy": os.path.abspath(out_npy),
            "npz": os.path.abspath(out_npz),
        }
    }

    out_json = os.path.join(args.out_dir, "rgbd_to_topdown_graspnet.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Save JSON] {out_json}")

    print("\n✅ 完成：已生成从上往下看的 Contact-GraspNet 输入。")
    print(f"PLY : {out_ply}")
    print(f"NPY : {out_npy}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
