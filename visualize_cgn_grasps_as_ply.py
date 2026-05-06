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


def load_pcd_points(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise RuntimeError(f"无法读取点云: {path}")

    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts


def downsample_points(points, max_points=120000, seed=0):
    points = np.asarray(points, dtype=np.float64)
    if max_points <= 0 or len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def add_points(all_pts, all_cols, pts, color):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return

    color = np.asarray(color, dtype=np.uint8).reshape(1, 3)
    cols = np.repeat(color, len(pts), axis=0)

    all_pts.append(pts)
    all_cols.append(cols)


def sample_line(a, b, n=80):
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    t = np.linspace(0.0, 1.0, int(n)).reshape(-1, 1)
    return a.reshape(1, 3) * (1.0 - t) + b.reshape(1, 3) * t


def sample_sphere(center, radius=0.004, n=120):
    center = np.asarray(center, dtype=np.float64).reshape(3)
    pts = []
    golden = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        pts.append(center + radius * np.array([x, y, z]))

    return np.asarray(pts, dtype=np.float64)


def load_table_plane(path):
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
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
        return n / norm, float(d) / norm

    if "normal" in data and "d" in data:
        n = np.asarray(data["normal"], dtype=np.float64)
        d = float(data["d"])
        norm = np.linalg.norm(n)
        return n / norm, d / norm

    return None


def sample_plane_grid(n, d, cloud_pts, grid_n=70, margin=0.05):
    n = normalize(n)

    # 找平面上的一个点
    origin = -d * n

    # 构造平面内两个正交方向
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, n)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    u = normalize(ref - np.dot(ref, n) * n)
    v = normalize(np.cross(n, u))

    rel = cloud_pts - origin.reshape(1, 3)
    cu = rel @ u
    cv = rel @ v

    u_min, u_max = cu.min() - margin, cu.max() + margin
    v_min, v_max = cv.min() - margin, cv.max() + margin

    us = np.linspace(u_min, u_max, grid_n)
    vs = np.linspace(v_min, v_max, grid_n)

    pts = []
    for a in us:
        for b in vs:
            pts.append(origin + a * u + b * v)

    return np.asarray(pts, dtype=np.float64)


def write_ascii_ply(points, colors, path):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

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

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )

    print(f"[Save] {path}")
    print(f"points = {len(points)}")


def add_grasp_visual(all_pts, all_cols, grasp, rank, args):
    T = np.asarray(grasp["T_grasp_rgbd"], dtype=np.float64)
    score = float(grasp.get("score", 0.0))

    center = T[:3, 3]
    R = T[:3, :3]

    x_axis = normalize(R[:, 0])
    y_axis = normalize(R[:, 1])
    z_axis = normalize(R[:, 2])

    axis_len = args.axis_len
    width = args.gripper_width
    finger_len = args.finger_len

    # rank 0 用亮紫色中心，其余用橙色
    if rank == 0:
        center_color = [255, 0, 255]
        grip_color = [255, 120, 0]
    else:
        center_color = [255, 180, 0]
        grip_color = [255, 160, 0]

    # 抓取中心
    add_points(all_pts, all_cols, sample_sphere(center, radius=args.center_radius, n=160), center_color)

    # contact point
    if "contact_point_rgbd" in grasp:
        cp = np.asarray(grasp["contact_point_rgbd"], dtype=np.float64)
        add_points(all_pts, all_cols, sample_sphere(cp, radius=args.contact_radius, n=120), [255, 255, 0])
        add_points(all_pts, all_cols, sample_line(center, cp, n=60), [255, 255, 0])

    # 只给 top1 画完整三轴，避免太乱；也可以 --draw-all-axes
    if rank == 0 or args.draw_all_axes:
        add_points(all_pts, all_cols, sample_line(center, center + x_axis * axis_len, n=100), [255, 0, 0])
        add_points(all_pts, all_cols, sample_line(center, center + y_axis * axis_len, n=100), [0, 255, 0])
        add_points(all_pts, all_cols, sample_line(center, center + z_axis * axis_len, n=100), [0, 80, 255])

    # 简化夹爪：开口方向用 x_axis，两根手指沿 -z_axis 伸出
    left = center - x_axis * (width / 2.0)
    right = center + x_axis * (width / 2.0)

    finger_dir = -z_axis * float(args.finger_sign)

    left_tip = left + finger_dir * finger_len
    right_tip = right + finger_dir * finger_len

    add_points(all_pts, all_cols, sample_line(left, right, n=100), grip_color)
    add_points(all_pts, all_cols, sample_line(left, left_tip, n=100), grip_color)
    add_points(all_pts, all_cols, sample_line(right, right_tip, n=100), grip_color)

    # 分数越高，额外画一个稍大的中心球
    if score > args.score_big_sphere_thresh:
        add_points(all_pts, all_cols, sample_sphere(center, radius=args.center_radius * 1.8, n=200), [255, 0, 180])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud-ply", required=True)
    parser.add_argument("--grasp-json", required=True)
    parser.add_argument("--out-ply", required=True)
    parser.add_argument("--table-plane-json", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cloud-max-points", type=int, default=120000)

    parser.add_argument("--gripper-width", type=float, default=0.065)
    parser.add_argument("--finger-len", type=float, default=0.045)
    parser.add_argument("--finger-sign", type=float, default=1.0)
    parser.add_argument("--axis-len", type=float, default=0.045)

    parser.add_argument("--center-radius", type=float, default=0.004)
    parser.add_argument("--contact-radius", type=float, default=0.003)

    parser.add_argument("--draw-all-axes", action="store_true")
    parser.add_argument("--score-big-sphere-thresh", type=float, default=0.68)

    args = parser.parse_args()

    cloud = load_pcd_points(args.cloud_ply)
    cloud_vis = downsample_points(cloud, args.cloud_max_points, seed=0)

    all_pts = []
    all_cols = []

    # 物体点云：浅蓝
    add_points(all_pts, all_cols, cloud_vis, [80, 150, 255])

    # 桌面：绿色点阵
    plane = load_table_plane(args.table_plane_json)
    if plane is not None:
        n, d = plane

        h = cloud @ n + d
        if np.median(h) < 0:
            n = -n
            d = -d

        plane_pts = sample_plane_grid(n, d, cloud, grid_n=70, margin=0.06)
        add_points(all_pts, all_cols, plane_pts, [40, 220, 80])

    with open(args.grasp_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    grasps = data["grasps"][:args.top_k]

    print(f"[Info] loaded grasps: {len(data['grasps'])}")
    print(f"[Info] visualize top-k: {len(grasps)}")

    for rank, g in enumerate(grasps):
        add_grasp_visual(all_pts, all_cols, g, rank, args)

    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)

    write_ascii_ply(pts, cols, args.out_ply)

    print("\n颜色说明：")
    print("  浅蓝 = SAM3D 水壶点云")
    print("  绿色 = 桌面平面点阵")
    print("  橙色 = Contact-GraspNet 夹爪")
    print("  黄色 = contact point")
    print("  紫色 = top1 grasp center")
    print("  红/绿/蓝 = top1 抓取坐标轴 x/y/z")
    print("\n如果夹爪方向看起来反了，重新运行时加：--finger-sign -1")


if __name__ == "__main__":
    main()
