#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import copy
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# =========================================================
# 0. 基础工具
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_points(pts: np.ndarray, dedup_decimals=6) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts
    _, unique_idx = np.unique(
        np.round(pts, decimals=dedup_decimals),
        axis=0,
        return_index=True
    )
    return pts[np.sort(unique_idx)]

def apply_affine(pts: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts @ A.T + t.reshape(1, 3)

def load_source_geometry(path: str, sample_points=40000):
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        pts = clean_points(np.asarray(pcd.points))
        return "mesh", mesh, pts

    pcd = o3d.io.read_point_cloud(path)
    pts = clean_points(np.asarray(pcd.points))
    if pts.shape[0] == 0:
        raise ValueError(f"无法读取有效 source: {path}")
    return "pcd", pcd, pts

def load_point_cloud_points(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = clean_points(np.asarray(pcd.points))
    if pts.shape[0] == 0:
        raise ValueError(f"无法读取有效点云: {path}")
    return pts

def save_point_cloud(path: str, pts: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    if color is not None:
        colors = np.repeat(np.asarray(color, dtype=np.float64).reshape(1, 3), len(pts), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)

def sample_points(pts: np.ndarray, n: int, seed=42) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]

def normalize(v: np.ndarray, eps=1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

# =========================================================
# 1. 稳健 ICP 与视角投影核心
# =========================================================
def translation_icp_template_to_scene(template_pts: np.ndarray, scene_pts: np.ndarray, max_iter=30, tol=1e-5, max_dist=0.03):
    scene_current = scene_pts.copy()
    t_total = np.zeros(3, dtype=np.float64)

    for _ in range(max_iter):
        tree = cKDTree(scene_current)
        dist, idx = tree.query(template_pts, distance_upper_bound=max_dist)
        valid = dist < max_dist
        if not np.any(valid):
            break

        matched_template = template_pts[valid]
        matched_scene = scene_current[idx[valid]]
        
        diff = matched_template - matched_scene
        norms = np.linalg.norm(diff, axis=1)
        if len(norms) > 10:
            thresh = np.percentile(norms, 80)
            good = norms <= thresh
            t_step = np.mean(diff[good], axis=0)
        else:
            t_step = np.mean(diff, axis=0)
            
        scene_current += t_step.reshape(1, 3)
        t_total += t_step

        if np.linalg.norm(t_step) < tol:
            break

    return t_total

def build_view_basis(view_dir: np.ndarray):
    n = normalize(view_dir)
    if abs(n[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = normalize(np.cross(n, ref))
    v = normalize(np.cross(n, u))
    return u, v, n

def project_points_by_view(pts: np.ndarray, view_dir: np.ndarray):
    u, v, n = build_view_basis(view_dir)
    return pts @ u, pts @ v, pts @ n

def extract_visible_shell_by_view(pts: np.ndarray, view_dir: np.ndarray, front_mode="min"):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0: return pts

    pu, pv, pd = project_points_by_view(pts, view_dir)
    proj = np.stack([pu, pv], axis=1)
    proj_diag = float(np.linalg.norm(np.percentile(proj, 98.0, axis=0) - np.percentile(proj, 2.0, axis=0)))
    depth_span = float(np.percentile(pd, 98.0) - np.percentile(pd, 2.0))

    pixel_size = max(proj_diag / 140.0, 8e-4)
    shell_thickness = max(depth_span * 0.03, pixel_size * 1.5, 0.002)

    u_min, v_min = float(np.min(pu)) - 2 * pixel_size, float(np.min(pv)) - 2 * pixel_size
    ui = np.floor((pu - u_min) / pixel_size).astype(np.int32)
    vi = np.floor((pv - v_min) / pixel_size).astype(np.int32)

    w, h = int(np.max(ui)) + 1, int(np.max(vi)) + 1

    if front_mode == "min":
        zbuf = np.full((w, h), np.inf, dtype=np.float64)
        np.minimum.at(zbuf, (ui, vi), pd)
        mask = pd <= (zbuf[ui, vi] + shell_thickness)
    else:
        zbuf = np.full((w, h), -np.inf, dtype=np.float64)
        np.maximum.at(zbuf, (ui, vi), pd)
        mask = pd >= (zbuf[ui, vi] - shell_thickness)

    shell = pts[mask]
    return shell if shell.shape[0] >= 50 else pts

# =========================================================
# 2. 改进的 F-Beta 3D 评分系统 (防崩塌核心)
# =========================================================
def score_template_to_scene_kd(template_pts: np.ndarray, scene_pts: np.ndarray, dist_thresh=0.015):
    """
    通过 F-Beta 分数平衡 Precision 和 Recall，彻底防止尺度崩塌（Template 变极小或极大）。
    """
    if len(template_pts) == 0 or len(scene_pts) == 0:
        return -1e18, {}

    # Precision: Template 在 Scene 中找到归宿的比例
    tree_scene = cKDTree(scene_pts)
    dist_t2s, _ = tree_scene.query(template_pts, workers=-1)
    precision15 = float(np.mean(dist_t2s < dist_thresh))
    precision10 = float(np.mean(dist_t2s < 0.010))
    mean_dist_t2s = float(np.mean(dist_t2s))

    # Recall: Scene 被 Template 覆盖的比例
    # 由于 Scene 包含巨大背景墙，Recall 天然有上限（比如最高 30%）。
    tree_template = cKDTree(template_pts)
    dist_s2t, _ = tree_template.query(scene_pts, workers=-1)
    recall15 = float(np.mean(dist_s2t < dist_thresh))

    # F-Beta 融合 (Beta = 0.5, 强调 Precision，但绝不允许 Recall 趋近于 0)
    # 这一步是解决尺度严重缩水的致命武器
    f_score = (1.25 * precision15 * recall15) / (0.25 * precision15 + recall15 + 1e-9)

    score = (
        5.0 * f_score +
        1.0 * precision10 -
        0.5 * (mean_dist_t2s / max(dist_thresh, 1e-9))
    )

    return score, {
        "fit15": precision15,
        "fit10": precision10,
        "recall15": recall15,
        "f_score": f_score,
        "mean_dist": mean_dist_t2s
    }

# =========================================================
# 3. 3D 多尺度精修扫描 (暴力破解包围盒畸变)
# =========================================================
def refine_scale_and_translation_with_kd(sam_centered: np.ndarray, part_pts: np.ndarray, view_dir: np.ndarray,
                                         front_mode: str, base_scale: float, dist_thresh=0.015):
    """
    在最佳视角的 3D 阶段，展开多尺度 + 局部网格暴力搜索，强制免疫背景带来的包围盒畸变。
    """
    part_eval = sample_points(part_pts, 8000, seed=12)
    scene_center = np.mean(part_eval, axis=0)

    best = {
        "score": -1e18, "scale": base_scale, "t_vec": np.zeros(3), "metrics": None, "shell_aligned": None
    }

    # 涵盖从 0.2x 到 5.0x 的庞大扫描阵列。
    # 无论初始尺度因为背景膨胀了多少倍，总有一个乘数能卡中鸭子的真实大小。
    multipliers = [0.2, 0.35, 0.5, 0.75, 1.0, 1.33, 2.0, 3.0, 5.0]

    for mult in multipliers:
        test_scale = base_scale * mult
        sam_scaled = sam_centered * test_scale
        
        shell = extract_visible_shell_by_view(
            sam_scaled, view_dir=view_dir, front_mode=front_mode
        )
        if shell.shape[0] < 50:
            continue
            
        shell_eval = sample_points(shell, 6000, seed=11)
        shell_center = np.mean(shell_eval, axis=0)
        
        # 初始平移：基于质心粗略对齐
        t_base = scene_center - shell_center
        
        # 为了防止质心对齐直接撞进背景墙的死胡同，展开 3x3x3 空间网格突围
        local_search_radius = max(0.05 * test_scale, 0.015) 
        d_list = [-local_search_radius, 0.0, local_search_radius]
        
        for dx in d_list:
            for dy in d_list:
                for dz in d_list:
                    t_try = t_base + np.array([dx, dy, dz], dtype=np.float64)
                    shell_try = shell_eval + t_try.reshape(1, 3)
                    
                    # 稳健吸附
                    t_icp = translation_icp_template_to_scene(
                        template_pts=shell_try, scene_pts=part_eval, 
                        max_iter=15, max_dist=dist_thresh * 3.0
                    )
                    
                    shell_aligned = shell_try + t_icp.reshape(1, 3)
                    total_t = t_try + t_icp
                    
                    score, metrics = score_template_to_scene_kd(shell_aligned, part_eval, dist_thresh)
                    
                    if score > best["score"]:
                        best = {
                            "score": float(score),
                            "scale": float(test_scale),
                            "t_vec": total_t,
                            "metrics": metrics,
                            "shell_aligned": shell_aligned
                        }

    # 针对扫出的最佳尺度，做最后一次全量精度抛光
    if best["score"] != -1e18:
        sam_scaled = sam_centered * best["scale"]
        shell = extract_visible_shell_by_view(sam_scaled, view_dir=view_dir, front_mode=front_mode)
        shell_try = shell + best["t_vec"].reshape(1, 3)
        t_icp_final = translation_icp_template_to_scene(shell_try, part_pts, max_iter=30, max_dist=dist_thresh*2)
        
        best["t_vec"] += t_icp_final
        shell_final = shell_try + t_icp_final.reshape(1, 3)
        best["score"], best["metrics"] = score_template_to_scene_kd(shell_final, part_pts, dist_thresh)
        best["shell_aligned"] = shell_final

    return best

# =========================================================
# 4. 全局流程调度
# =========================================================
def fibonacci_sphere(samples=64):
    pts = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(samples):
        y = 0.0 if samples == 1 else 1 - (i / float(samples - 1)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        pts.append([np.cos(theta) * radius, y, np.sin(theta) * radius])
    return np.asarray(pts, dtype=np.float64)

def robust_projected_diag_by_view(pts: np.ndarray, view_dir: np.ndarray) -> float:
    pu, pv, _ = project_points_by_view(pts, view_dir)
    proj = np.stack([pu, pv], axis=1)
    if proj.shape[0] < 10: return 0.0
    lo = np.percentile(proj, 5.0, axis=0)
    hi = np.percentile(proj, 95.0, axis=0)
    return float(np.linalg.norm(hi - lo))

def auto_register_view_locked(sam_pts_raw: np.ndarray, part_pts: np.ndarray,
                              sphere_samples=64, dist_thresh=0.015):
    sam_center = np.mean(sam_pts_raw, axis=0)
    sam_centered = sam_pts_raw - sam_center
    part_centered = part_pts - np.mean(part_pts, axis=0)

    directions = fibonacci_sphere(samples=sphere_samples)
    print(f"[1] 开始球面搜索真实视角方向，共 {len(directions)} 个候选 ...")

    # 步骤 A：粗略视角评估
    coarse_candidates = []
    for view_dir in directions:
        sam_diag = robust_projected_diag_by_view(sam_centered, view_dir)
        part_diag = robust_projected_diag_by_view(part_centered, view_dir)
        if sam_diag < 1e-8 or part_diag < 1e-8: continue
        
        base_scale = part_diag / sam_diag
        sam_scaled = sam_centered * base_scale
        
        # 仅用非常简化的逻辑进行粗略打分，快速筛掉错误视角
        for front_mode in ["min", "max"]:
            shell = extract_visible_shell_by_view(sam_scaled, view_dir, front_mode)
            if shell.shape[0] < 50: continue
            
            # 中心对齐 + 极简打分，不卡死在精确度上
            t_base = np.mean(part_pts, axis=0) - np.mean(shell, axis=0)
            shell_try = shell + t_base.reshape(1, 3)
            
            # 这里快速算一个 Precision，用于挑出前 6 个即可
            tree_scene = cKDTree(sample_points(part_pts, 2000))
            dist_t2s, _ = tree_scene.query(sample_points(shell_try, 2000))
            coarse_score = float(np.mean(dist_t2s < (dist_thresh * 3.0)))

            coarse_candidates.append({
                "view_dir": view_dir.copy(),
                "front_mode": front_mode,
                "base_scale": float(base_scale),
                "coarse_score": coarse_score
            })

    if not coarse_candidates:
        raise RuntimeError("没有找到有效的视角候选。")

    # 步骤 B：筛选 Top-6 视角，进入深度 3D 尺度网格突围
    coarse_candidates = sorted(coarse_candidates, key=lambda x: x["coarse_score"], reverse=True)[:6]
    print(f"[2] 进入 3D 尺度与平移暴力扫描，保留 top-{len(coarse_candidates)} 个视角候选 ...")

    best = None
    for rank, cand in enumerate(coarse_candidates, 1):
        refined = refine_scale_and_translation_with_kd(
            sam_centered=sam_centered, part_pts=part_pts, view_dir=cand["view_dir"],
            front_mode=cand["front_mode"], base_scale=cand["base_scale"], dist_thresh=dist_thresh
        )

        metrics = refined["metrics"]
        msg = (
            f"    cand#{rank}: front={cand['front_mode']}, "
            f"Scale Sweep=[{refined['scale']:.4f}], "
            f"F-Beta={metrics['f_score']:.4f}, "
            f"fit15={metrics['fit15']*100:.2f}%, mean={metrics['mean_dist']*1000:.2f} mm"
        )
        print(msg)

        if best is None or refined["score"] > best["score"]:
            best = {
                "view_dir": cand["view_dir"], "front_mode": cand["front_mode"],
                "scale": refined["scale"], "t_vec": refined["t_vec"],
                "metrics": metrics, "score": refined["score"], 
                "sam_center": sam_center, "final_shell_scaled": refined["shell_aligned"]
            }

    if best is None:
        raise RuntimeError("3D 尺度精筛阶段失败。")

    return best

# =========================================================
# 5. 主程序与正则输出
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Canonical Space Alignment: 加入 3D 多尺度扫描防崩塌系统"
    )
    parser.add_argument("--sam3d", required=True, help="SAM3D 完整点云或 mesh")
    parser.add_argument("--partial", required=True, help="真实残缺点云")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--sam-sample", type=int, default=40000, help="采样点数")
    parser.add_argument("--sphere-samples", type=int, default=64, help="球面视角候选数")
    parser.add_argument("--dist-thresh", type=float, default=0.015, help="3D 评分距离阈值")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    print("\n" + "=" * 78)
    print("[🚀] 启动：Canonical Space Alignment (多尺度网格扫描对抗版)")
    print("=" * 78)

    geom_type, geom_raw, sam_pts_raw = load_source_geometry(args.sam3d, sample_points=args.sam_sample)
    part_pts = load_point_cloud_points(args.partial)

    best = auto_register_view_locked(
        sam_pts_raw=sam_pts_raw, part_pts=part_pts,
        sphere_samples=args.sphere_samples, dist_thresh=args.dist_thresh
    )

    scale = float(best["scale"])
    t_vec = np.asarray(best["t_vec"], dtype=np.float64)
    sam_center = np.asarray(best["sam_center"], dtype=np.float64)

    # 核心对齐：蓝色模型完全静止，红色相机数据逆变换至正则坐标系
    final_blue = sam_pts_raw 
    part_canonical = (part_pts - t_vec) / scale + sam_center
    shell_blue = (best["final_shell_scaled"] - t_vec) / scale + sam_center
    
    # 最后在正则空间执行一次去噪吸附
    t_icp = translation_icp_template_to_scene(
        template_pts=shell_blue, 
        scene_pts=part_canonical, 
        max_dist=(args.dist_thresh * 2) / scale
    )
    part_final = part_canonical + t_icp.reshape(1, 3)

    inv_scale = 1.0 / scale
    t_canonical = sam_center + t_icp - (t_vec / scale)
    _, final_metrics = score_template_to_scene_kd(shell_blue, part_final, dist_thresh=args.dist_thresh/scale)

    print("\n" + "-" * 72)
    print("[最终对抗结果]")
    print(f"view_dir     = {[round(v, 6) for v in best['view_dir'].tolist()]}")
    print(f"inv_scale    = {inv_scale:.8f} (Camera to Canonical)")
    print(f"F-Beta Score = {final_metrics['f_score']:.4f}")
    print(f"fit@1.5cm    = {final_metrics['fit15'] * 100:.2f}%")
    print(f"fit@1.0cm    = {final_metrics['fit10'] * 100:.2f}%")
    print("-" * 72)

    save_point_cloud(os.path.join(args.out_dir, "canonical_source.ply"), final_blue, color=[0.10, 0.45, 0.85])
    save_point_cloud(os.path.join(args.out_dir, "aligned_partial.ply"), part_final, color=[0.85, 0.10, 0.10])
    save_point_cloud(os.path.join(args.out_dir, "visible_shell_canonical.ply"), shell_blue, color=[0.10, 0.85, 0.30])

    merged_pts = np.vstack([final_blue, part_final])
    merged_colors = np.vstack([
        np.tile([0.10, 0.45, 0.85], (len(final_blue), 1)),
        np.tile([0.85, 0.10, 0.10], (len(part_final), 1)),
    ])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(merged_pts)
    merged.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "vis_merged.ply"), merged)

    if geom_type == "mesh":
        o3d.io.write_triangle_mesh(os.path.join(args.out_dir, "canonical_source_mesh.obj"), geom_raw)

    with open(os.path.join(args.out_dir, "final_transform.json"), "w", encoding="utf-8") as f:
        json.dump({
            "note": "Anti-Noise Scale Sweep + F-Beta ICP",
            "view_dir": best["view_dir"].tolist(),
            "front_mode": best["front_mode"],
            "scale_camera_to_canonical": inv_scale,
            "translation_camera_to_canonical": t_canonical.tolist(),
            "metrics": final_metrics,
        }, f, indent=2)

    print("[保存完成]\n🎯 全流程结束。")

if __name__ == "__main__":
    main()