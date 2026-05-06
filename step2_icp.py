#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import json
import copy
import glob
import numpy as np
import open3d as o3d

def _maybe_fix_ho3d_paths(p: str) -> str:
    if os.path.exists(p): return p
    p2 = p.replace("HO3D_v3models", "HO3D_v3/models").replace("HO3D_v3train", "HO3D_v3/train")
    return p2 if os.path.exists(p2) else p

def robust_bbox_diag_from_points(pts: np.ndarray, q=2.0) -> float:
    if pts.shape[0] < 10: return 0.0
    lo = np.percentile(pts, q, axis=0)
    hi = np.percentile(pts, 100.0 - q, axis=0)
    return float(np.linalg.norm(hi - lo))

def get_pcd_scale_diag(pcd: o3d.geometry.PointCloud, robust=True) -> float:
    pts = np.asarray(pcd.points)
    if pts.size == 0: return 0.0
    if robust:
        d = robust_bbox_diag_from_points(pts, q=2.0)
        if d > 0: return d
    minb = pcd.get_min_bound()
    maxb = pcd.get_max_bound()
    return float(np.linalg.norm(maxb - minb))

def find_pointcloud_file(path_or_dir: str) -> str:
    path_or_dir = _maybe_fix_ho3d_paths(path_or_dir)
    if os.path.isfile(path_or_dir): return path_or_dir
    cands = [os.path.join(path_or_dir, fn) for fn in os.listdir(path_or_dir) if fn.lower().endswith((".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"))]
    key_priority = ["visible_obj", "visible_object", "visible_points", "obj", "object", "strict", "merged", "final"]
    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        for i, k in enumerate(key_priority):
            if k in name: s += (len(key_priority) - i) * 10
        if name.endswith(".ply"): s += 3
        return s
    cands.sort(key=score, reverse=True)
    return cands[0]

def find_mesh_in_model_dir(model_dir: str) -> str:
    model_dir = _maybe_fix_ho3d_paths(model_dir)
    if os.path.isfile(model_dir) and model_dir.lower().endswith((".ply", ".obj")): return model_dir
    prefer = ["textured_simple.ply", "textured_simple.obj", "textured.ply", "textured.obj", "model.ply", "model.obj"]
    lower_map = {fn.lower(): fn for fn in os.listdir(model_dir)}
    for pf in prefer:
        if pf in lower_map: return os.path.join(model_dir, lower_map[pf])
    return [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.lower().endswith((".ply", ".obj"))][0]

def load_meta(meta_path: str) -> dict:
    meta_path = _maybe_fix_ho3d_paths(meta_path)
    with open(meta_path, "rb") as f: meta = pickle.load(f)
    if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict): meta = meta[0]
    return meta

def mesh_true_scale_diag(mesh_path: str) -> float:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    return float(np.linalg.norm(extent))

# ==========================================
# 🛡️ 核心修复：防弹级点云缩放与清洗
# ==========================================
def clean_pcd(pcd: o3d.geometry.PointCloud):
    """清除 NaN, Inf 和重复点，防止 Open3D 崩溃"""
    pcd.remove_non_finite_points()
    pcd.remove_duplicated_points()
    return pcd

def scale_pcd_inplace(pcd: o3d.geometry.PointCloud, s: float):
    if s <= 0 or not np.isfinite(s):
        raise ValueError(f"非法缩放因子: {s}")
    # 彻底弃用 pcd.scale，使用 Numpy 绝对安全的矩阵运算
    pts = np.asarray(pcd.points)
    center = np.mean(pts, axis=0)
    pts = (pts - center) * s + center
    pcd.points = o3d.utility.Vector3dVector(pts)

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T

def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def centroid_init(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud):
    cs = np.asarray(source.get_center())
    ct = np.asarray(target.get_center())
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (ct - cs)
    return T

def estimate_normals_safe(pcd: o3d.geometry.PointCloud, radius: float, max_nn: int):
    if len(pcd.points) == 0: return
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.normalize_normals()

def coarse_score_icp(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, T_init: np.ndarray, max_dist: float):
    s = copy.deepcopy(source)
    s.transform(T_init)
    res = o3d.pipelines.registration.registration_icp(
        s, target, max_dist, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    score = float(res.fitness) - float(res.inlier_rmse) / max(max_dist, 1e-9)
    return score, res.fitness, res.inlier_rmse, res.transformation

def icp_refine_multistage(source, target, init_T, base_dist):
    s = copy.deepcopy(source)
    s.transform(init_T)
    estimate_normals_safe(s, radius=base_dist * 3.0, max_nn=50)
    estimate_normals_safe(target, radius=base_dist * 3.0, max_nn=50)
    stages = [
        ("ICP-coarse", base_dist * 2.0, 60),
        ("ICP-mid",    base_dist * 1.0, 60),
        ("ICP-fine",   base_dist * 0.5, 80),
    ]
    T = np.eye(4)
    last = None
    for name, dist, iters in stages:
        print(f"   [{name}] max_corr_dist={dist:.6f}, iters={iters}")
        last = o3d.pipelines.registration.registration_icp(
            s, target, dist, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=iters)
        )
        T = last.transformation
        print(f"      fitness={last.fitness:.6f}, rmse={last.inlier_rmse:.6f}")
    T_total = T @ init_T
    return last, T_total

def objrot_to_R(objRot):
    r = np.array(objRot, dtype=np.float64).reshape(-1)
    if r.size == 9: return r.reshape(3, 3)
    if r.size == 3:
        import cv2
        R, _ = cv2.Rodrigues(r.reshape(3, 1))
        return R
    raise ValueError(f"不支持的 objRot")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--voxel", type=float, default=-1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.model_dir = _maybe_fix_ho3d_paths(args.model_dir)
    args.meta = _maybe_fix_ho3d_paths(args.meta)
    args.sam3d = _maybe_fix_ho3d_paths(args.sam3d)
    args.partial = _maybe_fix_ho3d_paths(args.partial)

    print("[0] 读取 meta...")
    meta = load_meta(args.meta)
    
    print("[1] 读取 HO3D GT 模型并计算真实尺度...")
    mesh_path = find_mesh_in_model_dir(args.model_dir)
    true_diag = mesh_true_scale_diag(mesh_path)
    
    print("[2] 读取点云并清理内存脏数据...")
    sam_pcd = clean_pcd(o3d.io.read_point_cloud(args.sam3d))
    partial_path = find_pointcloud_file(args.partial)
    part_pcd = clean_pcd(o3d.io.read_point_cloud(partial_path))

    sam_diag = get_pcd_scale_diag(sam_pcd, robust=True)
    part_diag = get_pcd_scale_diag(part_pcd, robust=True)

    print("[3] 用真实尺度缩放 SAM3D...")
    print(f"    SAM3D diag(before):   {sam_diag:.8f}")
    print(f"    Partial diag(before): {part_diag:.8f}")

    sam_scale = true_diag / max(sam_diag, 1e-12)
    scale_pcd_inplace(sam_pcd, sam_scale)

    sam_diag2 = get_pcd_scale_diag(sam_pcd, robust=True)
    print(f"    SAM3D diag(after):    {sam_diag2:.8f}")
    print(f"    scale SAM3D by:       {sam_scale:.10f}")

    part_ratio = true_diag / max(part_diag, 1e-12)
    if part_ratio < 0.05 or part_ratio > 20.0:
        print(f"    ⚠️ Partial unit mismatch suspected, scale partial by {part_ratio:.10f}")
        scale_pcd_inplace(part_pcd, part_ratio)

    base = true_diag / 20.0 if args.voxel <= 0 else args.voxel
    base = max(base, 1e-4)

    print("[5] 构造初始化并自动选择最优初值...")
    inits = [("centroid", centroid_init(sam_pcd, part_pcd))]
    if "objRot" in meta and "objTrans" in meta:
        R = objrot_to_R(meta["objRot"])
        t = np.array(meta["objTrans"], dtype=np.float64).reshape(3)
        inits.append(("meta_obj_to_cam", make_T(R, t)))
        inits.append(("meta_cam_to_obj", inv_T(make_T(R, t))))

    best, best_detail = None, None
    for name, T0 in inits:
        score, fit, rmse, dT = coarse_score_icp(sam_pcd, part_pcd, T0, max_dist=true_diag * 2.0)
        print(f"    init={name:14s} score={score:+.6f} fitness={fit:.6f} rmse={rmse:.6f}")
        if best is None or score > best:
            best = score
            best_detail = (name, dT @ T0, fit, rmse)

    init_name, T_init_best, fit0, rmse0 = best_detail
    print(f"[5.5] best init = {init_name} (fitness={fit0:.6f}, rmse={rmse0:.6f})")

    print("[6] 多阶段ICP精配准...")
    icp_res, T_final = icp_refine_multistage(sam_pcd, part_pcd, T_init_best, base_dist=base)

    print("[7] 保存结果...")
    sam_aligned = copy.deepcopy(sam_pcd)
    sam_aligned.transform(T_final)
    
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "aligned_sam3d_to_partial.ply"), sam_aligned)
    sam_aligned.paint_uniform_color([0.2, 0.6, 1.0])
    part_pcd.paint_uniform_color([1.0, 0.2, 0.2])
    vis_path = os.path.join(args.out_dir, "vis_merged.ply")
    o3d.io.write_point_cloud(vis_path, sam_aligned + part_pcd)
    
    with open(os.path.join(args.out_dir, "scale_and_stats.json"), "w") as f:
        json.dump({"init_coarse_fitness": float(fit0), "icp_fitness": float(icp_res.fitness), "icp_rmse": float(icp_res.inlier_rmse)}, f, indent=2)

    print(f"✅ 配准大成功！")
    print(f"  最终 Fitness (重合度): {icp_res.fitness:.6f} (越近1越好)")
    print(f"  最终 RMSE (误差):      {icp_res.inlier_rmse:.6f} 米 (越小越好)")

if __name__ == "__main__":
    main()
