#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRot


# =========================================================
# 0. 基础工具
# =========================================================
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def angle_between_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    c = np.clip(float(np.dot(a, b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def make_pcd(points, color=None):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None and len(points) > 0:
        c = np.asarray(color, dtype=np.float64).reshape(1, 3)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(points), 1)))

    return pcd


def save_pcd(path, points, color=None):
    pcd = make_pcd(points, color=color)
    o3d.io.write_point_cloud(str(path), pcd)


def sample_points(points, n, seed=0):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    if len(points) <= n:
        return points.copy()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def uvz_to_xyz(u, v, z_m, intr):
    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    x = (u - cx) * z_m / fx
    y = (v - cy) * z_m / fy

    pts = np.stack([x, y, z_m], axis=1)
    return pts


def load_rgb_depth_mask(rgb_path, depth_path, mask_path):
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if rgb is None:
        raise FileNotFoundError(f"RGB 加载失败: {rgb_path}")
    if depth is None:
        raise FileNotFoundError(f"Depth 加载失败: {depth_path}")
    if mask is None:
        raise FileNotFoundError(f"Mask 加载失败: {mask_path}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)

    if mask.shape[:2] != depth.shape[:2]:
        print("[Warn] mask 尺寸和 depth 不一致，自动 resize mask")
        mask = cv2.resize(
            mask,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_bool = mask > 127

    return rgb, depth, mask_bool


def backproject_pixels_to_points(depth_mm, mask_bool, intr, depth_scale=0.001):
    valid = mask_bool & (depth_mm > 0)
    v, u = np.where(valid)

    if len(u) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    z = depth_mm[v, u].astype(np.float64) * float(depth_scale)

    pts = uvz_to_xyz(
        u.astype(np.float64),
        v.astype(np.float64),
        z,
        intr,
    )

    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


# =========================================================
# 1. 桌面 plane 提取
# =========================================================
def get_bbox_from_mask(mask_bool):
    ys, xs = np.where(mask_bool)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return x1, y1, x2, y2


def expand_bbox(bbox, w, h, scale=2.5, min_pad=40):
    x1, y1, x2, y2 = bbox

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    nw = max(bw * scale, bw + 2 * min_pad)
    nh = max(bh * scale, bh + 2 * min_pad)

    nx1 = int(max(0, cx - nw / 2))
    nx2 = int(min(w - 1, cx + nw / 2))
    ny1 = int(max(0, cy - nh / 2))
    ny2 = int(min(h - 1, cy + nh / 2))

    return nx1, ny1, nx2, ny2


def build_table_candidate_mask(depth_mm, obj_mask, expand_scale=2.5):
    h, w = depth_mm.shape[:2]

    bbox = get_bbox_from_mask(obj_mask)

    if bbox is None:
        print("[Warn] 目标 mask 为空，使用整张图做平面候选")
        roi_mask = np.ones_like(obj_mask, dtype=bool)
    else:
        x1, y1, x2, y2 = expand_bbox(
            bbox,
            w=w,
            h=h,
            scale=expand_scale,
            min_pad=40,
        )

        roi_mask = np.zeros_like(obj_mask, dtype=bool)
        roi_mask[y1:y2 + 1, x1:x2 + 1] = True

    # 排除目标物体，只用周围背景点拟合桌面
    candidate = roi_mask & (~obj_mask) & (depth_mm > 0)

    if int(candidate.sum()) < 800:
        print("[Warn] 局部 table 候选点太少，退化到整图非目标区域")
        candidate = (~obj_mask) & (depth_mm > 0)

    return candidate


def fit_table_plane_from_depth(
    depth_mm,
    obj_mask,
    intr,
    depth_scale=0.001,
    expand_scale=2.5,
    voxel_size=0.006,
    ransac_dist=0.008,
    ransac_iter=1500,
    min_inliers=400,
):
    cand_mask = build_table_candidate_mask(
        depth_mm=depth_mm,
        obj_mask=obj_mask,
        expand_scale=expand_scale,
    )

    cand_pts = backproject_pixels_to_points(
        depth_mm=depth_mm,
        mask_bool=cand_mask,
        intr=intr,
        depth_scale=depth_scale,
    )

    print(f"[Table] candidate points before downsample = {len(cand_pts)}")

    if len(cand_pts) < min_inliers:
        raise RuntimeError(f"table candidate 点太少: {len(cand_pts)}")

    pcd = make_pcd(cand_pts, color=[0.6, 0.6, 0.6])

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    print(f"[Table] candidate points after voxel = {len(pcd.points)}")

    if len(pcd.points) < min_inliers:
        raise RuntimeError(f"voxel 后 table candidate 点太少: {len(pcd.points)}")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=ransac_dist,
        ransac_n=3,
        num_iterations=ransac_iter,
    )

    a, b, c, d_raw = [float(x) for x in plane_model]

    normal_raw = np.array([a, b, c], dtype=np.float64)
    norm_raw = np.linalg.norm(normal_raw) + 1e-12

    normal = normal_raw / norm_raw
    d = float(d_raw) / norm_raw

    inlier_pcd = pcd.select_by_index(inliers)
    inlier_pts = np.asarray(inlier_pcd.points).astype(np.float64)

    print(
        f"[Table] plane raw: normal={normal.tolist()}, "
        f"d={d:.6f}, inliers={len(inlier_pts)}"
    )

    if len(inlier_pts) < min_inliers:
        print(f"[Warn] table inliers 偏少: {len(inlier_pts)}")

    return normal, d, inlier_pts, cand_pts


def orient_table_normal_to_object_side(normal, d, object_pts):
    """
    保证目标物体位于桌面的正侧：
        normal · x + d > 0
    """
    object_pts = np.asarray(object_pts, dtype=np.float64).reshape(-1, 3)

    if len(object_pts) == 0:
        print("[Warn] object points 为空，无法根据物体侧修正 table normal")
        return normal, d

    dist = object_pts @ normal + d
    med = float(np.median(dist))

    if med < 0:
        normal = -normal
        d = -d
        print("[Table] normal 已翻转，使目标物体位于桌面正侧")

    dist2 = object_pts @ normal + d

    print(
        f"[Table] object signed distance after orient: "
        f"median={np.median(dist2):.6f}m, "
        f"p1={np.percentile(dist2, 1):.6f}m, "
        f"p99={np.percentile(dist2, 99):.6f}m"
    )

    return normal, d


# =========================================================
# 2. Table frame：把桌面变成 XY 平面
# =========================================================
def rotation_from_a_to_b(a, b):
    """
    返回 R，使得 R @ a ≈ b
    """
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float64)

        # 180 度反向，找任意垂直轴旋转
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, tmp)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        axis = normalize(np.cross(a, tmp))
        return SciRot.from_rotvec(np.pi * axis).as_matrix()

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))
    return R.astype(np.float64)


def build_table_frame_transform(normal, d):
    """
    构造 table frame：

    输入相机坐标 plane:
        normal · x + d = 0

    输出：
        R_table: camera/world -> table frame 的旋转
        table_z: 旋转后桌面所在 z，高度平移时减掉它

    变换方式：
        pts_rot = pts @ R_table.T
        pts_table = pts_rot
        pts_table[:, 2] -= table_z

    变换后：
        桌面 z = 0
        桌面法向 = +Z
        物体应该在 z > 0
    """
    normal = normalize(normal)

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    R_table = rotation_from_a_to_b(normal, z_axis)

    # 桌面上一点：p0 = -d * normal
    p0 = -float(d) * normal

    p0_rot = p0 @ R_table.T
    table_z = float(p0_rot[2])

    return R_table, table_z


def transform_points_to_table_frame(points, R_table, table_z):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    pts = points @ R_table.T
    pts[:, 2] -= float(table_z)

    return pts


def transform_vectors_to_table_frame(vectors, R_table):
    vectors = np.asarray(vectors, dtype=np.float64)

    if vectors.ndim == 1:
        return R_table @ vectors

    return vectors @ R_table.T


def verify_table_frame(table_inliers_table, object_pts_table):
    if len(table_inliers_table) > 0:
        z = table_inliers_table[:, 2]
        print(
            f"[TableFrame] table z stats: "
            f"min={z.min():.6f}, "
            f"p50={np.percentile(z, 50):.6f}, "
            f"max={z.max():.6f}"
        )

    if len(object_pts_table) > 0:
        z = object_pts_table[:, 2]
        print(
            f"[TableFrame] object z stats: "
            f"min={z.min():.6f}, "
            f"p1={np.percentile(z, 1):.6f}, "
            f"p50={np.percentile(z, 50):.6f}, "
            f"p99={np.percentile(z, 99):.6f}"
        )


# =========================================================
# 3. SAM-3D rotation candidates + quaternion
# =========================================================
def get_24_rotations():
    rotations = []

    for p in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=np.float64)
            R[0, p[0]] = signs[0]
            R[1, p[1]] = signs[1]
            R[2, p[2]] = signs[2]

            if np.linalg.det(R) > 0:
                rotations.append(R)

    return rotations


def decode_quaternion_candidates(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64).reshape(-1)

    if len(raw_quat) != 4:
        raise ValueError(f"rotation_quat 长度不是 4: {raw_quat}")

    cands = []

    # 假设 raw = [w, x, y, z]
    w, x, y, z = raw_quat
    R1 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("wxyz_R", R1))
    cands.append(("wxyz_RT", R1.T))

    # 假设 raw = [x, y, z, w]
    x, y, z, w = raw_quat
    R2 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("xyzw_R", R2))
    cands.append(("xyzw_RT", R2.T))

    return cands


def compose_candidate_rotation(base_R, adapter_R, compose_mode):
    if compose_mode == "left":
        return adapter_R @ base_R

    if compose_mode == "right":
        return base_R @ adapter_R

    raise ValueError(compose_mode)


def load_pose_json_rotation_candidates(pose_json):
    if pose_json is None or not os.path.exists(pose_json):
        print("[Pose] 未提供 pose-json，只使用 identity")
        return [("identity", np.eye(3))]

    with open(pose_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "rotation_quat" not in data:
        print("[Pose] pose-json 中没有 rotation_quat，只使用 identity")
        return [("identity", np.eye(3))]

    raw_quat = data["rotation_quat"]
    print(f"[Pose] raw rotation_quat = {raw_quat}")

    return decode_quaternion_candidates(raw_quat)


def load_sam3d_points(path, sample_n=50000):
    if path is None or not os.path.exists(path):
        print("[SAM3D] 未提供 sam3d 点云，跳过物体点云统计")
        return None

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float64)

    if len(pts) == 0:
        mesh = o3d.io.read_triangle_mesh(path)

        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError(f"无法读取 SAM-3D 点云/mesh: {path}")

        if len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=sample_n)
            pts = np.asarray(pcd.points).astype(np.float64)
        else:
            pts = np.asarray(mesh.vertices).astype(np.float64)

    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = sample_points(pts, sample_n, seed=123)

    print(f"[SAM3D] loaded points = {len(pts)}")
    return pts


# =========================================================
# 4. 在 table frame 中探测 SAM-3D up axis
# =========================================================
def find_best_up_axis_in_table_frame(R_candidate, R_table):
    """
    SAM canonical axis -> camera/world:
        world_up = R_candidate @ axis

    camera/world -> table frame:
        table_up = R_table @ world_up

    由于桌面已被变换成 XY 面，真实 up = +Z。
    """
    axes = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }

    table_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    best = None

    for name, axis in axes.items():
        world_up = R_candidate @ axis
        up_in_table = R_table @ world_up
        up_in_table = normalize(up_in_table)

        angle = angle_between_deg(up_in_table, table_z)

        if best is None or angle < best["angle_deg"]:
            best = {
                "axis_name": name,
                "axis": axis.tolist(),
                "world_up": normalize(world_up).tolist(),
                "up_in_table": up_in_table.tolist(),
                "angle_deg": float(angle),
            }

    return best


def probe_up_axis_candidates_table_frame(
    R_table,
    pose_json,
    sam3d_points=None,
    topk=40,
):
    base_rots = load_pose_json_rotation_candidates(pose_json)
    adapters = get_24_rotations()

    rows = []

    for base_name, base_R in base_rots:
        for ai, A in enumerate(adapters):
            for compose_mode in ["left", "right"]:
                R_candidate = compose_candidate_rotation(
                    base_R,
                    A,
                    compose_mode,
                )

                best_up = find_best_up_axis_in_table_frame(
                    R_candidate=R_candidate,
                    R_table=R_table,
                )

                row = {
                    "base_name": base_name,
                    "adapter_idx": int(ai),
                    "compose_mode": compose_mode,
                    "best_up_axis": best_up["axis_name"],
                    "up_angle_deg": float(best_up["angle_deg"]),
                    "world_up": best_up["world_up"],
                    "up_in_table": best_up["up_in_table"],
                }

                if sam3d_points is not None and len(sam3d_points) > 0:
                    center = np.mean(sam3d_points, axis=0)
                    local = sam3d_points - center

                    # canonical -> camera/world
                    pts_world = local @ R_candidate.T

                    # camera/world -> table frame
                    pts_table = pts_world @ R_table.T

                    h = pts_table[:, 2]

                    row["height_p1_table_z"] = float(np.percentile(h, 1))
                    row["height_p50_table_z"] = float(np.percentile(h, 50))
                    row["height_p99_table_z"] = float(np.percentile(h, 99))
                    row["height_span_1_99_table_z"] = float(
                        np.percentile(h, 99) - np.percentile(h, 1)
                    )

                rows.append(row)

    rows = sorted(rows, key=lambda r: r["up_angle_deg"])

    print("\n[UpAxis/TableFrame] Top candidates by angle to +Z:")
    for i, r in enumerate(rows[:topk], 1):
        msg = (
            f"#{i:02d} "
            f"{r['base_name']} + A[{r['adapter_idx']}] + {r['compose_mode']} | "
            f"best_up={r['best_up_axis']} | "
            f"angle_to_+Z={r['up_angle_deg']:.2f} deg"
        )

        if "height_span_1_99_table_z" in r:
            msg += f" | z_span={r['height_span_1_99_table_z']:.4f}m"

        print(msg)

    axis_count = {}
    for r in rows[:topk]:
        axis_count[r["best_up_axis"]] = axis_count.get(r["best_up_axis"], 0) + 1

    print("\n[UpAxis/TableFrame] Top-k axis frequency:")
    for k, v in sorted(axis_count.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}/{topk}")

    return rows


# =========================================================
# 5. 输出结果
# =========================================================
def save_results(
    out_dir,
    normal,
    d,
    R_table,
    table_z,
    table_inliers,
    table_candidates,
    object_pts,
    table_inliers_table,
    table_candidates_table,
    object_pts_table,
    up_rows,
):
    ensure_dir(out_dir)

    out_dir = Path(out_dir)

    table_json = out_dir / "table_plane.json"
    table_transform_json = out_dir / "table_frame_transform.json"

    table_ply = out_dir / "table_plane_inliers_camera.ply"
    table_candidates_ply = out_dir / "table_candidates_camera.ply"
    object_ply = out_dir / "object_points_from_mask_camera.ply"
    scene_ply = out_dir / "table_object_debug_scene_camera.ply"

    table_xy_ply = out_dir / "table_plane_inliers_table_xy.ply"
    table_candidates_xy_ply = out_dir / "table_candidates_table_xy.ply"
    object_xy_ply = out_dir / "object_points_from_mask_table_xy.ply"
    scene_xy_ply = out_dir / "table_object_debug_scene_table_xy.ply"

    up_json = out_dir / "up_axis_candidates_table_frame.json"
    up_csv = out_dir / "up_axis_candidates_table_frame.csv"

    info = {
        "normal_camera": normal.tolist(),
        "d_camera": float(d),
        "plane_equation_camera": {
            "type": "n_dot_x_plus_d_equals_0",
            "normal": normal.tolist(),
            "d": float(d),
        },
        "meaning": "points with normal dot x + d > 0 are on object side after orientation",
        "num_table_inliers": int(len(table_inliers)),
        "num_table_candidates": int(len(table_candidates)),
        "num_object_points": int(len(object_pts)),
    }

    transform_info = {
        "R_table_camera_to_table": R_table.tolist(),
        "table_z_after_rotation": float(table_z),
        "transform_formula": {
            "rotation": "pts_rot = pts_camera @ R_table.T",
            "translation": "pts_table[:, 2] = pts_rot[:, 2] - table_z_after_rotation",
        },
        "table_frame_definition": {
            "table_plane": "z = 0",
            "table_normal": "+Z",
            "object_side": "z > 0",
        },
    }

    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    with open(table_transform_json, "w", encoding="utf-8") as f:
        json.dump(transform_info, f, indent=2, ensure_ascii=False)

    with open(up_json, "w", encoding="utf-8") as f:
        json.dump(up_rows, f, indent=2, ensure_ascii=False)

    if len(up_rows) > 0:
        fieldnames = list(up_rows[0].keys())
        with open(up_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(up_rows)

    # camera frame 输出
    save_pcd(table_ply, table_inliers, color=[0.0, 0.8, 0.0])
    save_pcd(table_candidates_ply, table_candidates, color=[0.5, 0.5, 0.5])
    save_pcd(object_ply, object_pts, color=[1.0, 0.0, 0.0])

    pcd_table = make_pcd(table_inliers, color=[0.0, 0.8, 0.0])
    pcd_obj = make_pcd(object_pts, color=[1.0, 0.0, 0.0])
    merged = pcd_table + pcd_obj
    o3d.io.write_point_cloud(str(scene_ply), merged)

    # table frame 输出：桌面应该变成 XY 面，z≈0
    save_pcd(table_xy_ply, table_inliers_table, color=[0.0, 0.8, 0.0])
    save_pcd(table_candidates_xy_ply, table_candidates_table, color=[0.5, 0.5, 0.5])
    save_pcd(object_xy_ply, object_pts_table, color=[1.0, 0.0, 0.0])

    pcd_table_xy = make_pcd(table_inliers_table, color=[0.0, 0.8, 0.0])
    pcd_obj_xy = make_pcd(object_pts_table, color=[1.0, 0.0, 0.0])
    merged_xy = pcd_table_xy + pcd_obj_xy
    o3d.io.write_point_cloud(str(scene_xy_ply), merged_xy)

    print("\n[Save]")
    print(f"  table json camera       : {table_json}")
    print(f"  table transform json    : {table_transform_json}")
    print(f"  camera debug scene      : {scene_ply}")
    print(f"  table-XY debug scene    : {scene_xy_ply}")
    print(f"  table-XY plane inliers  : {table_xy_ply}")
    print(f"  table-XY object points  : {object_xy_ply}")
    print(f"  up candidates json      : {up_json}")
    print(f"  up candidates csv       : {up_csv}")


# =========================================================
# 6. CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="测试：从真实 depth 提取桌面 plane，把桌面变成 XY 面，并用 SAM-3D 四元数探测 canonical up 轴"
    )

    parser.add_argument(
        "--rgb",
        default="/root/SAM-3D-Grasp/data/test/000002/rgb/000000.png",
    )
    parser.add_argument(
        "--depth",
        default="/root/SAM-3D-Grasp/data/test/000002/depth/000000.png",
    )
    parser.add_argument(
        "--mask",
        default="/root/SAM-3D-Grasp/output_pipeline_real_depth/target_mask.png",
    )

    parser.add_argument(
        "--sam3d",
        default="/root/SAM-3D-Grasp/output_3d/reconstructed_mesh.ply",
    )
    parser.add_argument(
        "--pose-json",
        default="/root/SAM-3D-Grasp/output_3d/sam3d_pose.json",
    )

    parser.add_argument(
        "--out-dir",
        default="/root/SAM-3D-Grasp/output_table_axis_probe",
    )

    parser.add_argument("--fx", type=float, default=572.4114)
    parser.add_argument("--fy", type=float, default=573.2265)
    parser.add_argument("--cx", type=float, default=325.2611)
    parser.add_argument("--cy", type=float, default=242.0489)
    parser.add_argument("--depth-scale", type=float, default=0.001)

    parser.add_argument("--expand-scale", type=float, default=2.5)
    parser.add_argument("--voxel-size", type=float, default=0.006)
    parser.add_argument("--ransac-dist", type=float, default=0.008)
    parser.add_argument("--ransac-iter", type=int, default=1500)
    parser.add_argument("--topk", type=int, default=40)

    args = parser.parse_args()

    intr = {
        "fx": float(args.fx),
        "fy": float(args.fy),
        "cx": float(args.cx),
        "cy": float(args.cy),
    }

    print("\n" + "=" * 80)
    print("🚀 Table plane -> XY frame + SAM-3D up-axis probe")
    print("=" * 80)
    print(f"[Input] rgb       : {args.rgb}")
    print(f"[Input] depth     : {args.depth}")
    print(f"[Input] mask      : {args.mask}")
    print(f"[Input] sam3d     : {args.sam3d}")
    print(f"[Input] pose-json : {args.pose_json}")
    print(f"[Output] out-dir  : {args.out_dir}")
    print("=" * 80)

    _, depth, obj_mask = load_rgb_depth_mask(
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_path=args.mask,
    )

    object_pts = backproject_pixels_to_points(
        depth_mm=depth,
        mask_bool=obj_mask,
        intr=intr,
        depth_scale=args.depth_scale,
    )

    print(f"[Object] points from mask = {len(object_pts)}")

    normal, d, table_inliers, table_candidates = fit_table_plane_from_depth(
        depth_mm=depth,
        obj_mask=obj_mask,
        intr=intr,
        depth_scale=args.depth_scale,
        expand_scale=args.expand_scale,
        voxel_size=args.voxel_size,
        ransac_dist=args.ransac_dist,
        ransac_iter=args.ransac_iter,
    )

    normal, d = orient_table_normal_to_object_side(
        normal=normal,
        d=d,
        object_pts=object_pts,
    )

    print("\n[Table Plane Final - Camera Frame]")
    print(f"  normal = {normal.tolist()}")
    print(f"  d      = {d:.8f}")
    print("  plane  = normal · x + d = 0")
    print("  object side: normal · x + d > 0")

    R_table, table_z = build_table_frame_transform(
        normal=normal,
        d=d,
    )

    print("\n[Table Frame Transform]")
    print("  R_table camera->table:")
    print(R_table)
    print(f"  table_z_after_rotation = {table_z:.8f}")
    print("  table frame: table plane = z=0, table up = +Z")

    table_inliers_table = transform_points_to_table_frame(
        table_inliers,
        R_table=R_table,
        table_z=table_z,
    )
    table_candidates_table = transform_points_to_table_frame(
        table_candidates,
        R_table=R_table,
        table_z=table_z,
    )
    object_pts_table = transform_points_to_table_frame(
        object_pts,
        R_table=R_table,
        table_z=table_z,
    )

    verify_table_frame(
        table_inliers_table=table_inliers_table,
        object_pts_table=object_pts_table,
    )

    sam3d_points = None

    if args.sam3d and os.path.exists(args.sam3d):
        sam3d_points = load_sam3d_points(
            args.sam3d,
            sample_n=50000,
        )

    up_rows = probe_up_axis_candidates_table_frame(
        R_table=R_table,
        pose_json=args.pose_json,
        sam3d_points=sam3d_points,
        topk=args.topk,
    )

    save_results(
        out_dir=args.out_dir,
        normal=normal,
        d=d,
        R_table=R_table,
        table_z=table_z,
        table_inliers=table_inliers,
        table_candidates=table_candidates,
        object_pts=object_pts,
        table_inliers_table=table_inliers_table,
        table_candidates_table=table_candidates_table,
        object_pts_table=object_pts_table,
        up_rows=up_rows,
    )

    print("\n✅ 完成。重点打开这个看桌面是否已经变成 XY 面：")
    print(f"  {Path(args.out_dir) / 'table_object_debug_scene_table_xy.ply'}")
    print("绿色是桌面 plane inliers，应该接近 z=0 的 XY 平面；红色是目标 mask 回投影点，应在 z>0。")
    print("=" * 80)


if __name__ == "__main__":
    main()