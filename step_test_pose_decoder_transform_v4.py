#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot


def ensure_clean_point_cloud(pcd):
    tmp = pcd.remove_non_finite_points()
    if isinstance(tmp, tuple):
        return tmp[0]
    return tmp


def load_points_any(path: str, mesh_samples: int = 50000) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=mesh_samples)
        pts = np.asarray(pcd.points).astype(np.float64)
        if len(pts) > 0:
            return pts

    pcd = o3d.io.read_point_cloud(path)
    pcd = ensure_clean_point_cloud(pcd)
    pts = np.asarray(pcd.points).astype(np.float64)
    if len(pts) == 0:
        raise RuntimeError(f"无法从 {path} 读取有效点云/网格。")
    return pts


def save_point_cloud(path, pts, color=None):
    pts = np.asarray(pts, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None:
        c = np.asarray(color, dtype=np.float64).reshape(1, 3)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(pts), 1)))
    o3d.io.write_point_cloud(path, pcd)


def save_merged_point_cloud(path, pts_a, pts_b,
                            color_a=(0.10, 0.45, 0.85),
                            color_b=(0.85, 0.10, 0.10)):
    pts_a = np.asarray(pts_a, dtype=np.float64)
    pts_b = np.asarray(pts_b, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([pts_a, pts_b]))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([
        np.tile(np.asarray(color_a, dtype=np.float64), (len(pts_a), 1)),
        np.tile(np.asarray(color_b, dtype=np.float64), (len(pts_b), 1)),
    ]))
    o3d.io.write_point_cloud(path, pcd)


def decode_candidates(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64).reshape(-1)
    if len(raw_quat) != 4:
        raise RuntimeError(f"rotation 长度不是 4: {raw_quat}")

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


def evaluate_alignment(src_pts: np.ndarray, tgt_pts: np.ndarray, thresh=0.015):
    tree_tgt = cKDTree(tgt_pts)
    d_st, _ = tree_tgt.query(src_pts, workers=-1)

    tree_src = cKDTree(src_pts)
    d_ts, _ = tree_src.query(tgt_pts, workers=-1)

    fit15_src_to_tgt = float(np.mean(d_st < thresh))
    fit15_tgt_to_src = float(np.mean(d_ts < thresh))
    rmse_src_to_tgt = float(np.sqrt(np.mean(d_st ** 2)))
    rmse_tgt_to_src = float(np.sqrt(np.mean(d_ts ** 2)))

    score = (
        0.65 * fit15_tgt_to_src +
        0.35 * fit15_src_to_tgt
        - 0.35 * rmse_tgt_to_src
        - 0.10 * rmse_src_to_tgt
    )

    return {
        "fit15_full_to_partial": fit15_src_to_tgt,
        "fit15_partial_to_full": fit15_tgt_to_src,
        "rmse_full_to_partial": rmse_src_to_tgt,
        "rmse_partial_to_full": rmse_tgt_to_src,
        "score": float(score),
    }


def normalize_mesh_verts_np(verts: np.ndarray):
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmax + vmin) / 2.0
    extent = vmax - vmin
    max_extent = np.max(extent)
    if max_extent == 0:
        vertices = verts - center
        scale = 1.0
    else:
        scale = 1.0 / max_extent
        vertices = (verts - center) * scale
    return vertices, scale, center


def apply_rot(points: np.ndarray, R: np.ndarray):
    return points @ R.T


# get_mesh 里的固定轴旋转
A_GET_MESH = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0]
], dtype=np.float64)

# convert_to_halo 里的两个固定旋转
A_HALO_MESH = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0]
], dtype=np.float64)

A_HALO_PM = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


def apply_pose(points: np.ndarray, scale_xyz: np.ndarray, R_pose: np.ndarray, t_xyz: np.ndarray):
    pts = points * scale_xyz.reshape(1, 3)
    pts = pts @ R_pose.T
    pts = pts + t_xyz.reshape(1, 3)
    return pts


def run_chain_variant(full_pts, normalize_first, pose_name, R_pose, scale_xyz, t_xyz, halo_mode):
    """
    normalize_first:
        False -> 直接用输入 full
        True  -> 先按 normalize_mesh_verts 做规范化

    halo_mode:
        "none"
        "left"         : A_HALO_MESH @ pose
        "right"        : pose @ A_HALO_PM
        "both_lr"      : A_HALO_MESH @ pose @ A_HALO_PM
        "both_rl"      : A_HALO_PM @ pose @ A_HALO_MESH   (额外试一个反顺序)
    """

    pts = full_pts.copy()

    # 1) 可选归一化
    if normalize_first:
        pts, norm_scale, norm_center = normalize_mesh_verts_np(pts)
    else:
        norm_scale, norm_center = None, None

    # 2) get_mesh 固定轴旋转
    pts = apply_rot(pts, A_GET_MESH)

    # 3) pose decoder 输出的 scale + rotation + translation
    pts = apply_pose(pts, scale_xyz, R_pose, t_xyz)

    # 4) 可选 halo 坐标修正
    if halo_mode == "none":
        pass
    elif halo_mode == "left":
        pts = apply_rot(pts, A_HALO_MESH)
    elif halo_mode == "right":
        pts = apply_rot(pts, A_HALO_PM)
    elif halo_mode == "both_lr":
        pts = apply_rot(pts, A_HALO_PM)
        pts = apply_rot(pts, A_HALO_MESH)
    elif halo_mode == "both_rl":
        pts = apply_rot(pts, A_HALO_MESH)
        pts = apply_rot(pts, A_HALO_PM)
    else:
        raise ValueError(f"unknown halo_mode: {halo_mode}")

    tag = f"norm{int(normalize_first)}_{pose_name}_{halo_mode}"
    aux = {
        "tag": tag,
        "norm_scale": norm_scale,
        "norm_center": None if norm_center is None else norm_center.tolist(),
    }
    return pts, aux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--sam-partial", required=True)
    parser.add_argument("--pose-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--thresh", type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("[🚀] 启动：按 normalize + get_mesh + halo 全链测试 pose decoder")
    print("=" * 60)

    full_pts = load_points_any(args.full, mesh_samples=args.samples)
    sam_partial_pts = load_points_any(args.sam_partial, mesh_samples=args.samples)

    with open(args.pose_json, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    raw_rot = pose_data.get("rotation_quat", pose_data.get("rotation", None))
    if raw_rot is None:
        raise RuntimeError("pose json 里没有 rotation / rotation_quat")

    t = np.asarray(pose_data["translation"], dtype=np.float64).reshape(-1)
    s = np.asarray(pose_data["scale"], dtype=np.float64).reshape(-1)

    print(f"[1] rotation = {raw_rot}")
    print(f"[2] translation = {t}")
    print(f"[3] scale = {s}")

    candidates = decode_candidates(raw_rot)
    halo_modes = ["none", "left", "right", "both_lr", "both_rl"]
    norm_flags = [False, True]

    all_results = []
    best = None
    best_score = -1e18

    print("[4] 开始穷举链条...\n")

    for normalize_first in norm_flags:
        for pose_name, R_pose in candidates:
            for halo_mode in halo_modes:
                full_try, aux = run_chain_variant(
                    full_pts=full_pts,
                    normalize_first=normalize_first,
                    pose_name=pose_name,
                    R_pose=R_pose,
                    scale_xyz=s,
                    t_xyz=t,
                    halo_mode=halo_mode,
                )
                metrics = evaluate_alignment(full_try, sam_partial_pts, thresh=args.thresh)

                rec = {
                    "tag": aux["tag"],
                    "normalize_first": normalize_first,
                    "pose_name": pose_name,
                    "halo_mode": halo_mode,
                    "norm_scale": aux["norm_scale"],
                    "norm_center": aux["norm_center"],
                    **metrics,
                }
                all_results.append(rec)

                print(f"[{rec['tag']}]")
                print(f"   fit15(full -> sam_partial) = {metrics['fit15_full_to_partial']*100:.2f}%")
                print(f"   fit15(sam_partial -> full) = {metrics['fit15_partial_to_full']*100:.2f}%")
                print(f"   rmse(full -> sam_partial)  = {metrics['rmse_full_to_partial']:.4f} m")
                print(f"   rmse(sam_partial -> full)  = {metrics['rmse_partial_to_full']:.4f} m")
                print(f"   score                      = {metrics['score']:.6f}")

                out_sub = os.path.join(args.out_dir, rec["tag"])
                os.makedirs(out_sub, exist_ok=True)
                save_point_cloud(os.path.join(out_sub, "full_transformed.ply"), full_try, color=[0.10, 0.45, 0.85])
                save_merged_point_cloud(os.path.join(out_sub, "merged_full_vs_sam_partial.ply"), full_try, sam_partial_pts)

                if metrics["score"] > best_score:
                    best_score = metrics["score"]
                    best = {
                        "rec": rec,
                        "full_transformed": full_try,
                    }

    result_json = os.path.join(args.out_dir, "pose_decoder_test_results_v4.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    best_full_path = os.path.join(args.out_dir, "best_full_transformed_v4.ply")
    best_merge_path = os.path.join(args.out_dir, "best_merged_full_vs_sam_partial_v4.ply")
    save_point_cloud(best_full_path, best["full_transformed"], color=[0.10, 0.45, 0.85])
    save_merged_point_cloud(best_merge_path, best["full_transformed"], sam_partial_pts)

    best_json = os.path.join(args.out_dir, "best_pose_decoder_candidate_v4.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best["rec"], f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ 测试完成")
    print(f"   最优解释: {best['rec']['tag']}")
    print(f"   fit15(full -> sam_partial): {best['rec']['fit15_full_to_partial']*100:.2f}%")
    print(f"   fit15(sam_partial -> full): {best['rec']['fit15_partial_to_full']*100:.2f}%")
    print(f"   rmse(full -> sam_partial):  {best['rec']['rmse_full_to_partial']:.4f} m")
    print(f"   rmse(sam_partial -> full):  {best['rec']['rmse_partial_to_full']:.4f} m")
    print(f"   best_full:   {best_full_path}")
    print(f"   best_merged: {best_merge_path}")
    print(f"   all_results: {result_json}")
    print(f"   best_json:   {best_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()