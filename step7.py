#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


# =========================================================
# 0. 基础工具
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def safe_mesh_vertices_triangles(mesh: o3d.geometry.TriangleMesh):
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("mesh.vertices 形状异常")
    if tris.ndim != 2 or tris.shape[1] != 3:
        raise ValueError("mesh.triangles 形状异常")

    vmask = np.isfinite(verts).all(axis=1)
    if not np.all(vmask):
        old_to_new = -np.ones(len(verts), dtype=np.int64)
        keep_idx = np.where(vmask)[0]
        old_to_new[keep_idx] = np.arange(len(keep_idx))
        tris_mask = np.all(vmask[tris], axis=1)
        tris = old_to_new[tris[tris_mask]]
        verts = verts[keep_idx]

    valid_tri = (
        (tris[:, 0] >= 0) & (tris[:, 0] < len(verts)) &
        (tris[:, 1] >= 0) & (tris[:, 1] < len(verts)) &
        (tris[:, 2] >= 0) & (tris[:, 2] < len(verts))
    )
    tris = tris[valid_tri]

    return verts, tris


def load_source_geometry(path: str, sample_points=25000):
    # 先尝试读 mesh
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        try:
            verts, tris = safe_mesh_vertices_triangles(mesh)
            mesh_clean = o3d.geometry.TriangleMesh()
            mesh_clean.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh_clean.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
            if len(mesh_clean.vertices) > 0 and len(mesh_clean.triangles) > 0:
                pts = clean_points(np.asarray(mesh_clean.sample_points_uniformly(sample_points).points))
                return "mesh", mesh_clean, pts
        except Exception as e:
            print(f"[警告] Mesh 读取成功但清洗失败，回退为点云模式: {e}")

    # 再读点云
    pcd = o3d.io.read_point_cloud(path)
    pts = clean_points(np.asarray(pcd.points))
    if len(pts) == 0:
        raise RuntimeError(f"无法从 {path} 读取有效 mesh 或 point cloud")
    return "pcd", pcd, pts


def sample_points(pts: np.ndarray, n: int, seed=42) -> np.ndarray:
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def save_point_cloud(path: str, pts: np.ndarray, color=None):
    pts = clean_points(pts)
    if len(pts) == 0:
        print(f"[跳过] 空点云未保存: {path}")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    if color is not None:
        color = np.asarray(color, dtype=np.float64).reshape(1, 3)
        colors = np.repeat(color, len(pts), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(path, pcd)
    print(f"[保存点云] {path} -> {'成功' if ok else '失败'}")


def transform_points(pts: np.ndarray, R_mat: np.ndarray, S: float, T: np.ndarray):
    pts = clean_points(pts)
    return (pts @ R_mat.T) * float(S) + T.reshape(1, 3)


def rebuild_and_save_mesh(path: str, mesh: o3d.geometry.TriangleMesh,
                          R_mat: np.ndarray, S: float, T: np.ndarray):
    """
    不在原始 mesh C++ 对象上原地 rotate/scale/translate，
    而是取 numpy 顶点自己算，再重建一个全新的 mesh。
    """
    verts, tris = safe_mesh_vertices_triangles(mesh)
    verts_new = transform_points(verts, R_mat, S, T)

    mesh_new = o3d.geometry.TriangleMesh()
    mesh_new.vertices = o3d.utility.Vector3dVector(verts_new.astype(np.float64))
    mesh_new.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))

    # 清理退化三角形/重复顶点，尽量避免写文件时 C++ 崩
    try:
        mesh_new.remove_duplicated_vertices()
        mesh_new.remove_degenerate_triangles()
        mesh_new.remove_duplicated_triangles()
        mesh_new.remove_non_manifold_edges()
        mesh_new.compute_vertex_normals()
    except Exception as e:
        print(f"[警告] mesh 清理阶段报错，但继续尝试写出: {e}")

    ok = o3d.io.write_triangle_mesh(
        path,
        mesh_new,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=False,
        write_vertex_colors=False,
        write_triangle_uvs=False,
        print_progress=False
    )
    print(f"[保存网格] {path} -> {'成功' if ok else '失败'}")
    return ok


# =========================================================
# 1. 相机视角 Z-Buffer 剥壳
# =========================================================
def extract_visible_shell_z_axis(pts: np.ndarray, pixel_size=0.005, shell_thickness=0.015):
    pts = clean_points(pts)
    if pts.shape[0] < 50:
        return pts

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    u_min, v_min = np.min(x), np.min(y)
    ui = np.floor((x - u_min) / pixel_size).astype(np.int32)
    vi = np.floor((y - v_min) / pixel_size).astype(np.int32)

    w = int(np.max(ui)) + 1
    h = int(np.max(vi)) + 1

    # 防止极端情况下网格过大
    if w <= 0 or h <= 0 or w * h > 20_000_000:
        return pts

    zbuf = np.full((w, h), np.inf, dtype=np.float64)
    np.minimum.at(zbuf, (ui, vi), z)

    mask = z <= (zbuf[ui, vi] + shell_thickness)
    shell_pts = pts[mask]
    return shell_pts if shell_pts.shape[0] >= 50 else pts


# =========================================================
# 2. 稳健 ICP 与打分
# =========================================================
def icp_shell_to_scene(shell_pts: np.ndarray, scene_pts: np.ndarray, max_iter=30, max_dist=0.05):
    shell_pts = clean_points(shell_pts)
    scene_pts = clean_points(scene_pts)

    curr_pts = shell_pts.copy()
    t_total = np.zeros(3, dtype=np.float64)
    tree_scene = cKDTree(scene_pts)

    for _ in range(max_iter):
        if len(curr_pts) == 0:
            break

        dist, idx = tree_scene.query(curr_pts, distance_upper_bound=max_dist)
        valid = np.isfinite(dist) & (dist < max_dist)
        if not np.any(valid):
            break

        diff = scene_pts[idx[valid]] - curr_pts[valid]
        norms = np.linalg.norm(diff, axis=1)

        if len(norms) > 10:
            thresh = np.percentile(norms, 60)
            keep = norms <= thresh
            if not np.any(keep):
                break
            t_step = np.mean(diff[keep], axis=0)
        else:
            t_step = np.mean(diff, axis=0)

        if not np.isfinite(t_step).all():
            break

        curr_pts += t_step.reshape(1, 3)
        t_total += t_step

        if np.linalg.norm(t_step) < 1e-4:
            break

    return t_total


def score_shell_to_scene(shell_pts: np.ndarray, scene_pts: np.ndarray, dist_thresh=0.015):
    shell_pts = clean_points(shell_pts)
    scene_pts = clean_points(scene_pts)

    if len(shell_pts) == 0 or len(scene_pts) == 0:
        return 0.0

    tree_scene = cKDTree(scene_pts)
    dist_t2s, _ = tree_scene.query(shell_pts)
    prec = float(np.mean(dist_t2s < dist_thresh))

    tree_shell = cKDTree(shell_pts)
    dist_s2t, _ = tree_shell.query(scene_pts)
    rec = float(np.mean(dist_s2t < dist_thresh))

    return (2.0 * prec * rec) / (prec + rec + 1e-9)


# =========================================================
# 3. 统一视角语义配准
# =========================================================
def unified_perspective_registration(blue_pts: np.ndarray, red_pts: np.ndarray,
                                     pose_json: str, dist_thresh=0.015):
    with open(pose_json, "r") as f:
        pose_data = json.load(f)

    q = pose_data["rotation_quat"]
    print(f"[解析] 成功读取原生相机四元数: {q}")

    r1 = R.from_quat([q[1], q[2], q[3], q[0]])
    r2 = R.from_quat([q[0], q[1], q[2], q[3]])
    rotations_pool = [
        r1.inv().as_matrix(),
        r1.as_matrix(),
        r2.inv().as_matrix(),
        r2.as_matrix(),
    ]

    blue_eval = sample_points(blue_pts, 8000, seed=1)
    red_eval = sample_points(red_pts, 8000, seed=2)

    blue_center = np.mean(blue_eval, axis=0)
    red_center = np.mean(red_eval, axis=0)

    blue_centered = blue_eval - blue_center
    red_centered = red_eval - red_center

    scales_pool = np.logspace(np.log10(0.05), np.log10(3.0), num=40)
    best = {"score": -1.0, "rot_mat": None, "scale": 1.0, "t_icp": np.zeros(3)}

    print("[扫描] 统一视角语义：旋转蓝鸭子 -> 剥离蓝正脸 -> 贴合红残壳...")

    for rot_mat in rotations_pool:
        blue_rot = blue_centered @ rot_mat.T

        for scale in scales_pool:
            blue_scaled = blue_rot * scale
            x_span = np.max(blue_scaled[:, 0]) - np.min(blue_scaled[:, 0])
            pixel_size = max(0.002, x_span / 100.0)

            blue_shell = extract_visible_shell_z_axis(
                blue_scaled, pixel_size, shell_thickness=dist_thresh
            )
            if blue_shell.shape[0] < 50:
                continue

            t_icp = icp_shell_to_scene(
                blue_shell, red_centered, max_iter=15, max_dist=dist_thresh * 3
            )
            blue_shell_aligned = blue_shell + t_icp.reshape(1, 3)
            score = score_shell_to_scene(blue_shell_aligned, red_centered, dist_thresh)

            if score > best["score"]:
                best = {
                    "score": score,
                    "rot_mat": rot_mat.copy(),
                    "scale": float(scale),
                    "t_icp": t_icp.copy(),
                }

    if best["score"] < 0 or best["rot_mat"] is None:
        raise RuntimeError("配准失败，未找到重合点。")

    best_idx = None
    for i, rr in enumerate(rotations_pool):
        if np.allclose(rr, best["rot_mat"], atol=1e-8):
            best_idx = i
            break

    print(
        f"🎯 语义对齐成功！格式 #{best_idx}, "
        f"Scale: {best['scale']:.4f}, 综合F得分: {best['score'] * 100:.2f}%"
    )

    R_mat = best["rot_mat"]
    S = best["scale"]

    blue_centered_full = blue_pts - blue_center
    blue_rot_scaled_full = (blue_centered_full @ R_mat.T) * S

    pixel_size_full = max(
        0.002,
        (np.max(blue_rot_scaled_full[:, 0]) - np.min(blue_rot_scaled_full[:, 0])) / 150.0
    )
    blue_shell_full = extract_visible_shell_z_axis(
        blue_rot_scaled_full, pixel_size_full, shell_thickness=dist_thresh
    )
    blue_shell_shifted = blue_shell_full + best["t_icp"].reshape(1, 3) + red_center.reshape(1, 3)

    t_icp_final = icp_shell_to_scene(
        blue_shell_shifted, red_pts, max_iter=30, max_dist=dist_thresh
    )

    T_total_icp = best["t_icp"] + t_icp_final
    blue_final_full = blue_rot_scaled_full + T_total_icp.reshape(1, 3) + red_center.reshape(1, 3)

    return blue_final_full, R_mat, S, blue_center, red_center, T_total_icp


# =========================================================
# 4. 主程序
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dist-thresh", type=float, default=0.015)
    parser.add_argument("--no-mesh", action="store_true",
                        help="只保存点云与json，不写mesh，用于排查 segfault")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("\n" + "=" * 78)
    print("[🚀] 启动：基于统一视角语义的 2.5D 极速对齐 (稳定写出版)")
    print("=" * 78)

    geom_type, geom_raw, blue_pts = load_source_geometry(args.sam3d)
    red_pts = clean_points(np.asarray(o3d.io.read_point_cloud(args.partial).points))

    if len(blue_pts) == 0 or len(red_pts) == 0:
        raise RuntimeError("输入点云为空，无法配准")

    blue_real_world, R_mat, S, blue_c, red_c, T_icp = unified_perspective_registration(
        blue_pts, red_pts, args.pose, args.dist_thresh
    )

    T_total = T_icp + red_c - S * (blue_c @ R_mat.T)

    print("\n" + "-" * 72)
    print("[最终抓取基准变换矩阵 (Canonical Mesh -> Real Camera Pose)]")
    print("Mesh_Camera = Scale * (Mesh_Canonical @ Rotation.T) + Translation")
    print(f"Rotation Mat =\n{np.round(R_mat, 4)}")
    print(f"Scale        = {S:.4f}")
    print(f"Translation  = {np.round(T_total, 4)}")
    print("-" * 72)

    # 1) 保存点云
    save_point_cloud(os.path.join(args.out_dir, "red_camera_partial.ply"),
                     red_pts, color=[0.85, 0.10, 0.10])
    save_point_cloud(os.path.join(args.out_dir, "blue_aligned_full.ply"),
                     blue_real_world, color=[0.10, 0.45, 0.85])

    pixel_size = max(
        0.002,
        (np.max(blue_real_world[:, 0]) - np.min(blue_real_world[:, 0])) / 150.0
    )
    blue_face_only = extract_visible_shell_z_axis(
        blue_real_world, pixel_size, shell_thickness=args.dist_thresh
    )
    save_point_cloud(os.path.join(args.out_dir, "blue_aligned_FACE_ONLY.ply"),
                     blue_face_only, color=[0.10, 0.85, 0.45])

    # 2) 合并点云
    merged_pts = clean_points(np.vstack([blue_real_world, red_pts]).astype(np.float64))
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(merged_pts)

    blue_n = len(clean_points(blue_real_world))
    red_n = len(clean_points(red_pts))
    merged_colors = np.vstack([
        np.tile(np.array([[0.10, 0.45, 0.85]], dtype=np.float64), (blue_n, 1)),
        np.tile(np.array([[0.85, 0.10, 0.10]], dtype=np.float64), (red_n, 1)),
    ])
    merged.colors = o3d.utility.Vector3dVector(merged_colors.astype(np.float64))

    ok_merge = o3d.io.write_point_cloud(
        os.path.join(args.out_dir, "vis_merged_real_world.ply"), merged
    )
    print(f"[保存合并点云] {'成功' if ok_merge else '失败'}")

    # 3) 保存 mesh：不再原地改 geom_raw
    if (not args.no_mesh) and geom_type == "mesh":
        try:
            ok_mesh = rebuild_and_save_mesh(
                os.path.join(args.out_dir, "final_grasp_ready_mesh.obj"),
                geom_raw,
                R_mat,
                S,
                T_total,
            )
            if not ok_mesh:
                print("[警告] mesh 写出失败，但主流程已完成，点云结果可用。")
        except Exception as e:
            print(f"[警告] mesh 写出阶段失败，但已避免段错误: {e}")

    # 4) JSON
    with open(os.path.join(args.out_dir, "final_transform_to_camera.json"), "w") as f:
        json.dump({
            "R_canonical_to_camera": R_mat.tolist(),
            "Scale_canonical_to_camera": float(S),
            "T_canonical_to_camera": T_total.tolist(),
        }, f, indent=2)

    print("[保存完成] 点云/JSON 已完成；若 mesh 正常则也已写出。")


if __name__ == "__main__":
    main()