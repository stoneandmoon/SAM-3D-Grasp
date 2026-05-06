#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pickle
import json
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# ==========================================
# 🛡️ 纯 Numpy 的数学引擎
# ==========================================
def best_fit_transform(A, B):
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def numpy_score(src_pts, dst_pts, T_init, max_dist=0.03):
    """严苛打分器：只看 3 厘米以内的绝对重合点"""
    src_homo = np.hstack([src_pts, np.ones((src_pts.shape[0], 1))])
    src_transformed = (src_homo @ T_init.T)[:, :3]
    
    tree = KDTree(dst_pts)
    distances, _ = tree.query(src_transformed)
    
    inliers = distances < max_dist
    fitness = np.sum(inliers) / len(distances)
    inlier_rmse = np.sqrt(np.mean(distances[inliers]**2)) if fitness > 0 else max_dist
        
    score = fitness - (inlier_rmse / max_dist)
    return score, fitness, inlier_rmse

def numpy_icp(source_pts, target_pts, init_T, max_iterations=50, tolerance=1e-6):
    src = np.copy(source_pts)
    dst = np.copy(target_pts)
    
    src_homo = np.hstack([src, np.ones((src.shape[0], 1))])
    src = (src_homo @ init_T.T)[:, :3]
    
    tree = KDTree(dst)
    prev_error = float('inf')
    T_accumulated = np.copy(init_T)

    for _ in range(max_iterations):
        distances, indices = tree.query(src)
        matched_dst = dst[indices]
        T_step = best_fit_transform(src, matched_dst)
        
        src_homo = np.hstack([src, np.ones((src.shape[0], 1))])
        src = (src_homo @ T_step.T)[:, :3]
        T_accumulated = T_step @ T_accumulated
        
        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
    final_distances, _ = tree.query(src)
    fitness = np.sum(final_distances < 0.02) / len(final_distances)
    rmse = np.sqrt(np.mean(final_distances**2))
    return T_accumulated, fitness, rmse

# ==========================================
# 👁️ 暴力翻转初始阵列生成器
# ==========================================
def generate_flip_inits(sam_pts, part_pts):
    """生成 4 种翻转姿态，防止陷入上下颠倒的局部最优"""
    sam_c = np.mean(sam_pts, axis=0)
    part_c = np.mean(part_pts, axis=0)
    
    inits = []
    # 0. 原始朝向
    inits.append(("Normal", np.eye(3)))
    # 1. 绕 X 轴翻转 (上下颠倒)
    inits.append(("Flip_X_180", np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])))
    # 2. 绕 Y 轴翻转
    inits.append(("Flip_Y_180", np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])))
    # 3. 绕 Z 轴翻转
    inits.append(("Flip_Z_180", np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])))

    candidate_Ts = []
    for name, R in inits:
        T = np.eye(4)
        T[:3, :3] = R
        # 核心：先绕自己质心旋转，再平移到目标质心
        T[:3, 3] = part_c - R @ sam_c
        candidate_Ts.append((name, T))
    return candidate_Ts

# ==========================================
# 🗂️ 辅助函数
# ==========================================
def get_pcd_scale_diag(pts: np.ndarray, q=2.0) -> float:
    if pts.shape[0] < 10: return 0.0
    lo = np.percentile(pts, q, axis=0)
    hi = np.percentile(pts, 100.0 - q, axis=0)
    return float(np.linalg.norm(hi - lo))

def _maybe_fix_ho3d_paths(p: str) -> str:
    if os.path.exists(p): return p
    p2 = p.replace("HO3D_v3models", "HO3D_v3/models").replace("HO3D_v3train", "HO3D_v3/train")
    return p2 if os.path.exists(p2) else p

def find_mesh_in_model_dir(model_dir: str) -> str:
    model_dir = _maybe_fix_ho3d_paths(model_dir)
    prefer = ["textured_simple.ply", "textured_simple.obj", "textured.ply", "textured.obj"]
    lower_map = {fn.lower(): fn for fn in os.listdir(model_dir)}
    for pf in prefer:
        if pf in lower_map: return os.path.join(model_dir, lower_map[pf])
    return [os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.lower().endswith((".ply", ".obj"))][0]

# ==========================================
# 🚀 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    args.model_dir = _maybe_fix_ho3d_paths(args.model_dir)
    args.sam3d = _maybe_fix_ho3d_paths(args.sam3d)
    args.partial = _maybe_fix_ho3d_paths(args.partial)

    print("[1] 计算 GT 物理尺度...")
    mesh = o3d.io.read_triangle_mesh(find_mesh_in_model_dir(args.model_dir))
    true_diag = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))

    print("[2] 读取点云并清理...")
    sam_pcd = o3d.io.read_point_cloud(args.sam3d).remove_non_finite_points()
    part_pcd = o3d.io.read_point_cloud(args.partial).remove_non_finite_points()
    
    sam_pts = np.asarray(sam_pcd.points).astype(np.float64)
    part_pts = np.asarray(part_pcd.points).astype(np.float64)

    print("[3] 物理真实尺度缩放...")
    sam_scale = true_diag / max(get_pcd_scale_diag(sam_pts), 1e-12)
    sam_center = np.mean(sam_pts, axis=0)
    sam_pts = (sam_pts - sam_center) * sam_scale + sam_center
    
    part_ratio = true_diag / max(get_pcd_scale_diag(part_pts), 1e-12)
    if part_ratio < 0.05 or part_ratio > 20.0:
        part_center = np.mean(part_pts, axis=0)
        part_pts = (part_pts - part_center) * part_ratio + part_center

    print("[4] 执行 Numpy 下采样提取骨架...")
    def quick_downsample(pts, target=2000):
        if len(pts) <= target: return pts
        return pts[np.random.choice(len(pts), target, replace=False)]
        
    sam_down = quick_downsample(sam_pts, 1500)
    part_down = quick_downsample(part_pts, 1500)

    print("[5] 暴力翻转搜索最优朝向 (3cm 严苛打分)...")
    inits = generate_flip_inits(sam_down, part_down)
    
    best, best_T = None, None
    for name, T0 in inits:
        # 关键修正：容差缩小到 3 厘米！绝不放过任何穿模！
        score, fit, rmse = numpy_score(sam_down, part_down, T0, max_dist=0.03)
        print(f"    方向=[{name:10s}] 得分={score:+.4f} (Fitness={fit:.4f})")
        if best is None or score > best:
            best, best_T, best_name = score, T0, name
    print(f"  👉 锁定最优朝向: {best_name}")

    print("[6] 启动 Numpy SVD ICP 精配准...")
    T_final, final_fit, final_rmse = numpy_icp(sam_down, part_down, init_T=best_T)

    print("[7] 生成高清结果并安全保存...")
    homo_pts = np.hstack([sam_pts, np.ones((len(sam_pts), 1))])
    sam_pts_final = (homo_pts @ T_final.T)[:, :3]
    
    merged_pts = np.vstack((sam_pts_final, part_pts))
    merged_colors = np.vstack((
        np.tile([0.2, 0.6, 1.0], (len(sam_pts_final), 1)), 
        np.tile([1.0, 0.2, 0.2], (len(part_pts), 1))      
    ))
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(merged_pts))
    merged_pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(merged_colors))
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "vis_ultimate.ply"), merged_pcd)
    
    with open(os.path.join(args.out_dir, "ultimate_stats.json"), "w") as f:
        json.dump({"best_init": best_name, "icp_fitness": float(final_fit), "icp_rmse": float(final_rmse)}, f, indent=2)

    print("\n✅ 具身智能配准终局之战胜利！")
    print(f"   最终 Fitness (2cm容差): {final_fit:.4f}")
    print(f"   最终 RMSE 物理误差:     {final_rmse:.6f} 米\n")

if __name__ == "__main__":
    main()