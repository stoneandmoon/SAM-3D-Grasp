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

    w, x, y, z = raw_quat
    R1 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("wxyz_R", R1))
    cands.append(("wxyz_RT", R1.T))

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
        0.60 * fit15_tgt_to_src +
        0.40 * fit15_src_to_tgt
        - 0.30 * rmse_tgt_to_src
        - 0.10 * rmse_src_to_tgt
    )

    return {
        "fit15_full_to_partial": fit15_src_to_tgt,
        "fit15_partial_to_full": fit15_tgt_to_src,
        "rmse_full_to_partial": rmse_src_to_tgt,
        "rmse_partial_to_full": rmse_tgt_to_src,
        "score": float(score),
    }


def apply_fixed_axis_rotation(pts: np.ndarray):
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ], dtype=np.float64)
    return pts @ A.T


def transform_mode_get_mesh_style(full_pts, scale, R, t):
    pts = apply_fixed_axis_rotation(full_pts)
    pts = pts * scale.reshape(1, 3)
    pts = pts @ R.T
    pts = pts + t.reshape(1, 3)
    return pts


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
    print("[🚀] 启动：按 get_mesh 源码链测试 pose decoder")
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
    all_results = []
    best = None
    best_score = -1e18

    print("[4] 正在测试 get_mesh 风格的 4 种解释...\n")

    for cand_name, R in candidates:
        full_try = transform_mode_get_mesh_style(full_pts, s, R, t)
        metrics = evaluate_alignment(full_try, sam_partial_pts, thresh=args.thresh)

        print(f"  [{cand_name}]")
        print(f"     fit15(full -> sam_partial) = {metrics['fit15_full_to_partial']*100:.2f}%")
        print(f"     fit15(sam_partial -> full) = {metrics['fit15_partial_to_full']*100:.2f}%")
        print(f"     rmse(full -> sam_partial)  = {metrics['rmse_full_to_partial']:.4f} m")
        print(f"     rmse(sam_partial -> full)  = {metrics['rmse_partial_to_full']:.4f} m")
        print(f"     score                      = {metrics['score']:.6f}")

        out_sub = os.path.join(args.out_dir, cand_name)
        os.makedirs(out_sub, exist_ok=True)
        save_point_cloud(os.path.join(out_sub, "full_transformed.ply"), full_try, color=[0.10, 0.45, 0.85])
        save_point_cloud(os.path.join(out_sub, "sam_partial_rgb.ply"), sam_partial_pts, color=[0.85, 0.10, 0.10])
        save_merged_point_cloud(os.path.join(out_sub, "merged_full_vs_sam_partial.ply"), full_try, sam_partial_pts)

        rec = {"tag": cand_name, **metrics}
        all_results.append(rec)

        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best = {
                "tag": cand_name,
                "full_transformed": full_try,
                "metrics": metrics,
            }

    result_json = os.path.join(args.out_dir, "pose_decoder_test_results_v3.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    best_full_path = os.path.join(args.out_dir, "best_full_transformed_v3.ply")
    best_merge_path = os.path.join(args.out_dir, "best_merged_full_vs_sam_partial_v3.ply")
    save_point_cloud(best_full_path, best["full_transformed"], color=[0.10, 0.45, 0.85])
    save_merged_point_cloud(best_merge_path, best["full_transformed"], sam_partial_pts)

    print("\n" + "=" * 50)
    print("✅ 测试完成")
    print(f"   最优解释: {best['tag']}")
    print(f"   fit15(full -> sam_partial): {best['metrics']['fit15_full_to_partial']*100:.2f}%")
    print(f"   fit15(sam_partial -> full): {best['metrics']['fit15_partial_to_full']*100:.2f}%")
    print(f"   rmse(full -> sam_partial):  {best['metrics']['rmse_full_to_partial']:.4f} m")
    print(f"   rmse(sam_partial -> full):  {best['metrics']['rmse_partial_to_full']:.4f} m")
    print(f"   best_full:   {best_full_path}")
    print(f"   best_merged: {best_merge_path}")
    print(f"   all_results: {result_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()