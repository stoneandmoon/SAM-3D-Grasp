import os
import argparse
import pickle
import json
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# ---------------------------------------------------------
# 1. 绝对准确的 SVD 配准数学函数
# ---------------------------------------------------------
def svd_align(A, B):
    """计算将 A 变换到 B 的最优 R, t"""
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = (A - centroid_A).T @ (B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B - R @ centroid_A
    return R, t

def numpy_icp(source_pts, target_pts, T_init, max_iter=50):
    """基于 Numpy 的稳健 ICP"""
    src = np.copy(source_pts)
    tree = KDTree(target_pts)
    
    # 应用初始变换
    R_total = T_init[:3, :3]
    t_total = T_init[:3, 3]
    curr_pts = (src @ R_total.T) + t_total
    
    prev_rmse = float('inf')
    for _ in range(max_iter):
        dist, idx = tree.query(curr_pts)
        # 只取最近的 80% 匹配点，过滤噪点
        mask = dist < np.percentile(dist, 80)
        if np.sum(mask) < 10: break
        
        R_step, t_step = svd_align(curr_pts[mask], target_pts[idx[mask]])
        
        curr_pts = (curr_pts @ R_step.T) + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step
        
        rmse = np.sqrt(np.mean(dist[mask]**2))
        if abs(prev_rmse - rmse) < 1e-6: break
        prev_rmse = rmse
        
    # 计算 Fitness (5cm内点率)
    final_dist, _ = tree.query(curr_pts)
    fitness = np.sum(final_dist < 0.05) / len(final_dist)
    return R_total, t_total, fitness, np.mean(final_dist)

# ---------------------------------------------------------
# 2. 辅助工具
# ---------------------------------------------------------
def objrot_to_R(objRot):
    r = np.array(objRot).reshape(-1)
    if r.size == 9: return r.reshape(3, 3)
    import cv2
    R, _ = cv2.Rodrigues(r.reshape(3, 1))
    return R

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--sam3d", required=True)
    parser.add_argument("--partial", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 加载数据
    print("[1] 正在加载点云与 Meta...")
    sam_pcd = o3d.io.read_point_cloud(args.sam3d)
    part_pcd = o3d.io.read_point_cloud(args.partial)
    sam_pts = np.asarray(sam_pcd.points).astype(np.float64)
    part_pts = np.asarray(part_pcd.points).astype(np.float64)
    
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
        if isinstance(meta, list): meta = meta[0]

    # 2. 尺度恢复 (从 HO3D Mesh 获取真实尺度)
    mesh = o3d.io.read_triangle_mesh(os.path.join(args.model_dir, "textured_simple.obj"))
    gt_diag = np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent())
    sam_diag = np.linalg.norm(np.max(sam_pts, 0) - np.min(sam_pts, 0))
    scale = gt_diag / max(sam_diag, 1e-9)
    sam_pts = (sam_pts - np.mean(sam_pts, axis=0)) * scale
    print(f"    ✓ 尺度对齐完成: Scale={scale:.4f}")

    # 3. 构造初始化方案 (借鉴 step2 的精髓)
    print("[2] 正在测试最优初始方向...")
    candidates = []
    
    # 方案 A: 质心直接对齐
    T_centroid = np.eye(4)
    T_centroid[:3, 3] = np.mean(part_pts, axis=0)
    candidates.append(("Centroid", T_centroid))
    
    # 方案 B: Meta 位姿 (Obj to Cam)
    if "objRot" in meta:
        R_m = objrot_to_R(meta["objRot"])
        t_m = np.array(meta["objTrans"]).reshape(3)
        T_meta = np.eye(4)
        T_meta[:3, :3], T_meta[:3, 3] = R_m, t_m
        candidates.append(("Meta_GT", T_meta))
        
        # 方案 C: Meta 翻转 (解决 Y-up/Z-up 冲突)
        R_flip = np.array([[1,0,0],[0,-1,0],[0,0,-1]]) @ R_m
        T_flip = np.eye(4)
        T_flip[:3, :3], T_flip[:3, 3] = R_flip, t_m
        candidates.append(("Meta_Flip", T_flip))

    # 4. 跑分筛选
    best_res = None
    best_fit = -1
    
    sam_sub = sam_pts[np.random.choice(len(sam_pts), 1000)]
    part_sub = part_pts[np.random.choice(len(part_pts), 2000)]
    
    for name, T_init in candidates:
        R, t, fit, err = numpy_icp(sam_sub, part_sub, T_init, max_iter=20)
        print(f"    • 方案 [{name:12s}] | Fitness: {fit:.4f} | RMSE: {err:.4f}m")
        if fit > best_fit:
            best_fit = fit
            best_res = (R, t, name)

    final_R, final_t, win_name = best_res
    print(f"[3] 最终胜出方案: {win_name} (Fitness: {best_fit:.4f})")

    # 5. 生成结果
    sam_aligned = (sam_pts @ final_R.T) + final_t
    
    merged_pcd = o3d.geometry.PointCloud()
    merged_points = np.vstack([sam_aligned, part_pts])
    merged_colors = np.vstack([
        np.tile([0.1, 0.6, 1.0], (len(sam_aligned), 1)), 
        np.tile([1.0, 0.2, 0.2], (len(part_pts), 1))
    ])
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)
    
    out_path = os.path.join(args.out_dir, "vis_fusion.ply")
    o3d.io.write_point_cloud(out_path, merged_pcd)
    
    # 6. 最终评估 (严苛模式)
    tree = KDTree(part_pts)
    dists, _ = tree.query(sam_aligned)
    print("\n" + "="*30)
    print(f"✅ 配准大功告成!")
    print(f"平均距离: {np.mean(dists):.4f}m")
    print(f"中位数距离: {np.median(dists):.4f}m")
    print(f"输出路径: {out_path}")
    print("="*30)

if __name__ == "__main__":
    main()