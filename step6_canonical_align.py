#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import copy
import pickle
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# =========================================================
# 0. 通用工具
# =========================================================
def _maybe_fix_ho3d_paths(p: str) -> str:
    if not p:
        return p
    if os.path.exists(p):
        return p
    p2 = p.replace("HO3D_v3models", "HO3D_v3/models").replace("HO3D_v3train", "HO3D_v3/train")
    return p2 if os.path.exists(p2) else p


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_mean(x: np.ndarray, axis=0):
    if x.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.mean(x, axis=axis)


def apply_affine(pts: np.ndarray, A: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    行向量表示:
        x' = x @ A.T + t
    A 可以是旋转矩阵，也可以是带统一缩放的 3x3 矩阵。
    """
    return pts @ A.T + t.reshape(1, 3)


def apply_rt(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return apply_affine(pts, R, t)


def compose_rt(R_new: np.ndarray, t_new: np.ndarray,
               R_old: np.ndarray, t_old: np.ndarray):
    """
    行向量表示下，先做 old，再做 new:
        x1 = x @ R_old.T + t_old
        x2 = x1 @ R_new.T + t_new
    => x2 = x @ (R_new @ R_old).T + (t_old @ R_new.T + t_new)
    """
    R = R_new @ R_old
    t = t_old @ R_new.T + t_new
    return R, t


def rt_about_center_to_rt(R: np.ndarray, t: np.ndarray, center: np.ndarray):
    """
    将绕 center 的刚体变换:
        x' = (x - c) @ R.T + c + t
    化成标准 RT:
        x' = x @ R.T + t_equiv
    """
    t_equiv = center - center @ R.T + t
    return R, t_equiv


def euler_to_R(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ], dtype=np.float64)
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float64)
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ], dtype=np.float64)
    return Rz @ Ry @ Rx


def robust_bbox_diag_from_points(pts: np.ndarray, q=2.0) -> float:
    if pts.shape[0] < 10:
        return 0.0
    lo = np.percentile(pts, q, axis=0)
    hi = np.percentile(pts, 100.0 - q, axis=0)
    return float(np.linalg.norm(hi - lo))


def get_scale_diag(pts: np.ndarray, robust=True) -> float:
    if pts.size == 0:
        return 0.0
    if robust:
        d = robust_bbox_diag_from_points(pts, q=2.0)
        if d > 0:
            return d
    return float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0)))


def clean_points(pts: np.ndarray, dedup_decimals=6) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, 3).astype(np.float64)

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts

    rounded = np.round(pts, decimals=dedup_decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return pts[unique_idx]


def objrot_to_R(objRot):
    r = np.asarray(objRot, dtype=np.float64).reshape(-1)
    if r.size == 9:
        return r.reshape(3, 3)
    if r.size == 3:
        try:
            import cv2
            R, _ = cv2.Rodrigues(r.reshape(3, 1))
            return R
        except Exception as e:
            raise ValueError(f"objRot 是 Rodrigues 向量，但缺少 cv2: {e}")
    raise ValueError(f"不支持的 objRot 形状: {r.shape}")


def load_meta(meta_path: str):
    if not meta_path:
        return {}
    meta_path = _maybe_fix_ho3d_paths(meta_path)
    if not os.path.exists(meta_path):
        print(f"[WARN] meta 文件不存在，跳过: {meta_path}")
        return {}

    try:
        if meta_path.lower().endswith(".json"):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
            meta = meta[0]
        if not isinstance(meta, dict):
            return {}
        return meta
    except Exception as e:
        print(f"[WARN] 读取 meta 失败，跳过: {e}")
        return {}


def find_mesh_in_model_dir(model_dir: str) -> str:
    model_dir = _maybe_fix_ho3d_paths(model_dir)
    if os.path.isfile(model_dir) and model_dir.lower().endswith((".ply", ".obj")):
        return model_dir

    prefer = [
        "textured_simple.obj",
        "textured_simple.ply",
        "textured.obj",
        "textured.ply",
        "model.obj",
        "model.ply",
    ]
    files = os.listdir(model_dir)
    lower_map = {fn.lower(): fn for fn in files}
    for pf in prefer:
        if pf.lower() in lower_map:
            return os.path.join(model_dir, lower_map[pf.lower()])

    cands = [os.path.join(model_dir, fn) for fn in files if fn.lower().endswith((".obj", ".ply"))]
    if not cands:
        raise FileNotFoundError(f"在 model_dir 中找不到 mesh: {model_dir}")
    return cands[0]


def load_source_geometry(path: str, sample_points=40000, jitter_std=0.0):
    """
    同时支持 mesh / pcd。
    返回:
        geom_type: "mesh" or "pcd"
        raw_geom:  原始几何对象 (mesh / pointcloud)
        raw_pts:   原始点云坐标 (N,3)
    """
    path = _maybe_fix_ho3d_paths(path)

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        pts = np.asarray(pcd.points).astype(np.float64)
        pts = clean_points(pts)
        if jitter_std > 0:
            pts = pts + np.random.normal(0.0, jitter_std, pts.shape)
        return "mesh", mesh, pts

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float64)
    pts = clean_points(pts)
    if pts.shape[0] == 0:
        raise ValueError(f"既不是有效 mesh，也不是有效点云: {path}")
    if jitter_std > 0:
        pts = pts + np.random.normal(0.0, jitter_std, pts.shape)
    return "pcd", pcd, pts


def load_point_cloud_points(path: str) -> np.ndarray:
    path = _maybe_fix_ho3d_paths(path)
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float64)
    pts = clean_points(pts)
    if pts.shape[0] == 0:
        raise ValueError(f"无法读取有效点云: {path}")
    return pts


# =========================================================
# 1. 粗配准：全局旋转搜索 + meta 先验候选
# =========================================================
def score_alignment(src_aligned: np.ndarray, tgt_tree: cKDTree, dist_thresh: float):
    dist, _ = tgt_tree.query(src_aligned, workers=-1)
    fit = float(np.mean(dist < dist_thresh))
    rmse = float(np.sqrt(np.mean(np.minimum(dist, dist_thresh) ** 2)))
    score = fit - 0.5 * (rmse / max(dist_thresh, 1e-9))
    return score, fit, rmse


def build_rotation_candidates_from_meta(meta: dict):
    cands = [("identity", np.eye(3, dtype=np.float64))]
    if "objRot" in meta:
        try:
            Rm = objrot_to_R(meta["objRot"])
            cands.append(("meta_R", Rm))
            cands.append(("meta_R_inv", Rm.T))
        except Exception as e:
            print(f"[WARN] meta.objRot 解析失败，跳过: {e}")
    return cands


def fast_global_search(src_pts, tgt_pts, meta=None,
                       sample_n=600,
                       angle_step=22.5,
                       dist_thresh=0.025,
                       local_refine_trials=256,
                       rng_seed=42):
    print("[1] 阶段一：全局旋转盲搜 + meta 候选初始化...")

    rng = np.random.default_rng(rng_seed)
    src_n = min(sample_n, len(src_pts))
    tgt_n = min(sample_n, len(tgt_pts))
    src_sub = src_pts[rng.choice(len(src_pts), src_n, replace=False)]
    tgt_sub = tgt_pts[rng.choice(len(tgt_pts), tgt_n, replace=False)]

    tgt_center = safe_mean(tgt_sub, axis=0)
    tgt_tree = cKDTree(tgt_sub)

    best_score = -1e18
    best_fit = -1.0
    best_rmse = 1e18
    best_R = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)
    best_name = "identity"

    def evaluate_rotation(R_mat, name):
        nonlocal best_score, best_fit, best_rmse, best_R, best_t, best_name
        test_src = apply_rt(src_sub, R_mat, np.zeros(3))
        t_vec = tgt_center - safe_mean(test_src, axis=0)
        aligned = test_src + t_vec
        score, fit, rmse = score_alignment(aligned, tgt_tree, dist_thresh)
        if score > best_score:
            best_score = score
            best_fit = fit
            best_rmse = rmse
            best_R = R_mat
            best_t = t_vec
            best_name = name

    # 先评估 meta 候选
    for name, R0 in build_rotation_candidates_from_meta(meta or {}):
        evaluate_rotation(R0, f"candidate:{name}")

    # 再做欧拉角全局网格搜索
    angles = np.arange(0.0, 360.0, angle_step, dtype=np.float64)
    for rx in angles:
        for ry in angles:
            for rz in angles:
                R_mat = euler_to_R(rx, ry, rz)
                evaluate_rotation(R_mat, f"grid({rx:.1f},{ry:.1f},{rz:.1f})")

    # 对最优旋转做局部小扰动精炼
    delta = angle_step * 0.5
    for _ in range(local_refine_trials):
        dr = rng.uniform(-delta, delta, 3)
        R_delta = euler_to_R(dr[0], dr[1], dr[2])
        R_try = R_delta @ best_R
        evaluate_rotation(R_try, "local_refine")

    print(f"    best init = {best_name}")
    print(f"    粗配准 fit={best_fit*100:.2f}% | rmse={best_rmse*1000:.2f} mm")
    return best_R, best_t, {"name": best_name, "fit": best_fit, "rmse": best_rmse, "score": best_score}


# =========================================================
# 2. 阶段二：Trimmed SVD-ICP
# =========================================================
def best_fit_transform(A: np.ndarray, B: np.ndarray):
    """
    求 A -> B 的最优刚体变换 (行向量风格最终通过 apply_rt 使用)
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - centroid_A @ R.T
    return R, t


def trimmed_icp(source, target, iterations=50, trim_percent=75.0, min_pairs=30):
    print("[2] 阶段二：Trimmed SVD-ICP 收紧几何缝隙...")

    curr = np.copy(source)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)

    tgt_tree = cKDTree(target)
    history = []

    for i in range(iterations):
        dist, idx = tgt_tree.query(curr, workers=-1)

        thresh = np.percentile(dist, trim_percent)
        mask = dist <= thresh
        if np.sum(mask) < min_pairs:
            print(f"    [ICP] 提前停止，匹配点太少: {np.sum(mask)}")
            break

        src_match = curr[mask]
        tgt_match = target[idx[mask]]

        R_step, t_step = best_fit_transform(src_match, tgt_match)
        curr = apply_rt(curr, R_step, t_step)
        R_total, t_total = compose_rt(R_step, t_step, R_total, t_total)

        mean_dist = float(np.mean(dist))
        trim_dist = float(np.mean(dist[mask]))
        history.append({"iter": i, "mean_dist": mean_dist, "trim_dist": trim_dist})

        if i % 10 == 0 or i == iterations - 1:
            print(f"    [ICP {i:02d}] mean={mean_dist*1000:.2f} mm | trim={trim_dist*1000:.2f} mm")

    return R_total, t_total, history


# =========================================================
# 3. 阶段三：2.5D 可见性精修
# =========================================================
def get_projection_axes(view_axis: str):
    view_axis = view_axis.lower()
    if view_axis == "x":
        return 1, 2, 0   # 投影到 yz，深度是 x
    if view_axis == "y":
        return 0, 2, 1   # 投影到 xz，深度是 y
    return 0, 1, 2       # 默认投影到 xy，深度是 z


def extract_front_points(pts: np.ndarray, view_axis="z", front_ratio=0.45, front_mode="min"):
    _, _, d_idx = get_projection_axes(view_axis)
    depth = pts[:, d_idx]
    d_min, d_max = depth.min(), depth.max()

    if front_mode == "min":
        thr = d_min + (d_max - d_min) * front_ratio
        mask = depth <= thr
    else:
        thr = d_max - (d_max - d_min) * front_ratio
        mask = depth >= thr

    out = pts[mask]
    if out.shape[0] < 50:
        return pts, np.ones(len(pts), dtype=bool)
    return out, mask


def get_comprehensive_loss(partial_pts, comp_pts, pixel_size=0.002, view_axis="z", front_mode="min"):
    """
    双向 2D 轮廓误差 + 简化 z-buffer 穿模惩罚
    """
    x_idx, y_idx, d_idx = get_projection_axes(view_axis)

    px = partial_pts[:, x_idx]
    py = partial_pts[:, y_idx]
    pd = partial_pts[:, d_idx]

    cx = comp_pts[:, x_idx]
    cy = comp_pts[:, y_idx]
    cd = comp_pts[:, d_idx]

    x_min = min(px.min(), cx.min()) - pixel_size * 5
    y_min = min(py.min(), cy.min()) - pixel_size * 5

    pxi = ((px - x_min) / pixel_size).astype(np.int32)
    pyi = ((py - y_min) / pixel_size).astype(np.int32)
    cxi = ((cx - x_min) / pixel_size).astype(np.int32)
    cyi = ((cy - y_min) / pixel_size).astype(np.int32)

    w = max(pxi.max(), cxi.max()) + 1
    h = max(pyi.max(), cyi.max()) + 1

    mask_p = np.zeros((w, h), dtype=bool)
    mask_c = np.zeros((w, h), dtype=bool)
    mask_p[pxi, pyi] = True
    mask_c[cxi, cyi] = True

    red_uncovered = np.sum(mask_p & (~mask_c))
    blue_uncovered = np.sum(mask_c & (~mask_p))
    error_ratio = float((red_uncovered + blue_uncovered) / max(np.sum(mask_p), 1))

    if front_mode == "min":
        z_buffer_p = np.full((w, h), np.inf, dtype=np.float64)
        z_buffer_c = np.full((w, h), np.inf, dtype=np.float64)
        np.minimum.at(z_buffer_p, (pxi, pyi), pd)
        np.minimum.at(z_buffer_c, (cxi, cyi), cd)
        overlap = mask_p & mask_c
        penetration_depths = np.clip(z_buffer_p[overlap] - z_buffer_c[overlap], 0, None)
    else:
        z_buffer_p = np.full((w, h), -np.inf, dtype=np.float64)
        z_buffer_c = np.full((w, h), -np.inf, dtype=np.float64)
        np.maximum.at(z_buffer_p, (pxi, pyi), pd)
        np.maximum.at(z_buffer_c, (cxi, cyi), cd)
        overlap = mask_p & mask_c
        penetration_depths = np.clip(z_buffer_c[overlap] - z_buffer_p[overlap], 0, None)

    mean_pen = float(np.mean(penetration_depths)) if penetration_depths.size > 0 else 0.0
    return error_ratio, mean_pen


def visibility_refinement(source_pts, target_pts,
                          iterations=300,
                          pixel_size=0.002,
                          view_axis="z",
                          front_mode="min",
                          rot_range_deg=3.0,
                          trans_range=0.005,
                          pen_weight=500.0,
                          dist_weight=5.0,
                          rng_seed=42):
    print("[3] 阶段三：2.5D 投影轮廓 + Z-Buffer 防穿模精修...")

    rng = np.random.default_rng(rng_seed)
    centroid = np.mean(source_pts, axis=0)

    best_err_ratio, best_pen = get_comprehensive_loss(
        target_pts, source_pts,
        pixel_size=pixel_size,
        view_axis=view_axis,
        front_mode=front_mode
    )

    tree = cKDTree(source_pts)
    dist, _ = tree.query(target_pts, workers=-1)
    best_dist = float(np.mean(dist))

    best_loss = best_err_ratio + pen_weight * best_pen + dist_weight * best_dist
    best_R = np.eye(3, dtype=np.float64)
    best_t = np.zeros(3, dtype=np.float64)

    print(f"    [初始] 轮廓误差={best_err_ratio*100:.2f}% | 穿模={best_pen*1000:.2f} mm | 点距={best_dist*1000:.2f} mm")

    for i in range(iterations):
        anneal = 1.0 - 0.8 * (i / max(iterations, 1))
        dr = rng.uniform(-rot_range_deg, rot_range_deg, 3) * anneal
        dt = rng.uniform(-trans_range, trans_range, 3) * anneal

        R_try = euler_to_R(dr[0], dr[1], dr[2])
        t_try = np.asarray(dt, dtype=np.float64)

        test_pts = apply_rt(source_pts - centroid, R_try, np.zeros(3)) + centroid + t_try

        err_ratio, pen_depth = get_comprehensive_loss(
            target_pts, test_pts,
            pixel_size=pixel_size,
            view_axis=view_axis,
            front_mode=front_mode
        )

        tree = cKDTree(test_pts)
        dist, _ = tree.query(target_pts, workers=-1)
        mean_dist = float(np.mean(dist))

        loss = err_ratio + pen_weight * pen_depth + dist_weight * mean_dist

        if loss < best_loss:
            best_loss = loss
            best_R = R_try
            best_t = t_try
            best_err_ratio = err_ratio
            best_pen = pen_depth
            best_dist = mean_dist

            if i % 50 == 0:
                print(f"    [refine {i:03d}] 轮廓误差={best_err_ratio*100:.2f}% | 穿模={best_pen*1000:.2f} mm | 点距={best_dist*1000:.2f} mm")

    print(f"    [完成] 轮廓误差={best_err_ratio*100:.2f}% | 穿模={best_pen*1000:.2f} mm | 点距={best_dist*1000:.2f} mm")
    return best_R, best_t, centroid, {
        "error_ratio": best_err_ratio,
        "penetration_depth": best_pen,
        "mean_dist": best_dist,
        "loss": best_loss,
    }


# =========================================================
# 4. IO 与结果保存
# =========================================================
def save_point_cloud(path: str, pts: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if color is not None:
        color = np.asarray(color, dtype=np.float64).reshape(1, 3)
        colors = np.repeat(color, len(pts), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd)


def save_mesh_with_affine(path: str, mesh: o3d.geometry.TriangleMesh, A: np.ndarray, t: np.ndarray):
    mesh_out = copy.deepcopy(mesh)
    verts = np.asarray(mesh_out.vertices).astype(np.float64)
    verts = apply_affine(verts, A, t)
    mesh_out.vertices = o3d.utility.Vector3dVector(verts)
    o3d.io.write_triangle_mesh(path, mesh_out)


# =========================================================
# 5. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="完整模型到局部点云的三阶段配准")
    parser.add_argument("--model-dir", required=True, help="GT 模型目录，用于读取真实尺度 mesh")
    parser.add_argument("--sam3d", required=True, help="source: mesh 或 pointcloud")
    parser.add_argument("--partial", required=True, help="target: partial point cloud")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--meta", default="", help="可选，pickle/json。若包含 objRot，会作为初始化旋转先验")

    parser.add_argument("--sam-sample", type=int, default=40000)
    parser.add_argument("--sam-jitter-std", type=float, default=0.0005)

    parser.add_argument("--search-sample", type=int, default=600)
    parser.add_argument("--global-step", type=float, default=22.5)
    parser.add_argument("--global-dist", type=float, default=0.025)
    parser.add_argument("--local-refine-trials", type=int, default=256)

    parser.add_argument("--icp-iters", type=int, default=50)
    parser.add_argument("--icp-trim", type=float, default=75.0)

    parser.add_argument("--refine-iters", type=int, default=300)
    parser.add_argument("--pixel-size", type=float, default=0.002)
    parser.add_argument("--front-ratio", type=float, default=0.45)
    parser.add_argument("--view-axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--front-mode", choices=["min", "max"], default="min")
    parser.add_argument("--rot-range-deg", type=float, default=3.0)
    parser.add_argument("--trans-range", type=float, default=0.005)
    parser.add_argument("--pen-weight", type=float, default=500.0)
    parser.add_argument("--dist-weight", type=float, default=5.0)

    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.rng_seed)
    ensure_dir(args.out_dir)

    args.model_dir = _maybe_fix_ho3d_paths(args.model_dir)
    args.sam3d = _maybe_fix_ho3d_paths(args.sam3d)
    args.partial = _maybe_fix_ho3d_paths(args.partial)
    args.meta = _maybe_fix_ho3d_paths(args.meta)

    print("\n" + "=" * 70)
    print("[🚀] 启动：全局旋转搜索 + Trimmed ICP + 2.5D 可见性精修")
    print("=" * 70)

    # -----------------------------------------------------
    # 1) 读入 source / target / GT 尺度 / meta
    # -----------------------------------------------------
    meta = load_meta(args.meta) if args.meta else {}

    print("[0] 读取 source / target / GT mesh ...")
    sam_geom_type, sam_geom_raw, sam_pts_raw = load_source_geometry(
        args.sam3d,
        sample_points=args.sam_sample,
        jitter_std=args.sam_jitter_std
    )
    part_pts = load_point_cloud_points(args.partial)

    gt_mesh_path = find_mesh_in_model_dir(args.model_dir)
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    if len(gt_mesh.vertices) == 0:
        raise ValueError(f"GT mesh 读取失败: {gt_mesh_path}")

    # -----------------------------------------------------
    # 2) 尺度对齐（更稳的 robust bbox）
    # -----------------------------------------------------
    print("[1] 真实尺度锁定...")
    gt_diag = np.linalg.norm(gt_mesh.get_axis_aligned_bounding_box().get_extent())

    sam_center = np.mean(sam_pts_raw, axis=0)
    sam_pts_centered = sam_pts_raw - sam_center
    sam_diag = get_scale_diag(sam_pts_centered, robust=True)
    if sam_diag <= 0:
        raise ValueError("sam source 尺度异常，无法计算包围盒")

    locked_scale = gt_diag / max(sam_diag, 1e-12)
    sam_pts_scaled = sam_pts_centered * locked_scale

    part_diag = get_scale_diag(part_pts, robust=True)

    print(f"    GT diag         = {gt_diag:.8f}")
    print(f"    SAM diag        = {sam_diag:.8f}")
    print(f"    Partial diag    = {part_diag:.8f}")
    print(f"    Locked scale    = {locked_scale:.10f}")

    # -----------------------------------------------------
    # 3) 阶段一：全局旋转搜索
    # -----------------------------------------------------
    coarse_R, coarse_t, coarse_info = fast_global_search(
        sam_pts_scaled,
        part_pts,
        meta=meta,
        sample_n=args.search_sample,
        angle_step=args.global_step,
        dist_thresh=args.global_dist,
        local_refine_trials=args.local_refine_trials,
        rng_seed=args.rng_seed
    )
    sam_coarse = apply_rt(sam_pts_scaled, coarse_R, coarse_t)

    # -----------------------------------------------------
    # 4) 阶段二：Trimmed ICP
    # -----------------------------------------------------
    icp_R, icp_t, icp_history = trimmed_icp(
        sam_coarse,
        part_pts,
        iterations=args.icp_iters,
        trim_percent=args.icp_trim
    )
    sam_icp = apply_rt(sam_coarse, icp_R, icp_t)

    # 当前总变换（作用在“已缩放且居中”的 source 上）
    R_total, t_total = compose_rt(icp_R, icp_t, coarse_R, coarse_t)

    # -----------------------------------------------------
    # 5) 阶段三：只对前脸区域做 2.5D 可见性精修
    # -----------------------------------------------------
    print("[3] 提取前脸区域做可见性精修...")
    sam_front, front_mask = extract_front_points(
        sam_icp,
        view_axis=args.view_axis,
        front_ratio=args.front_ratio,
        front_mode=args.front_mode
    )
    print(f"    front points = {len(sam_front)} / {len(sam_icp)}")

    refine_R_local, refine_t_local, refine_center, refine_info = visibility_refinement(
        sam_front,
        part_pts,
        iterations=args.refine_iters,
        pixel_size=args.pixel_size,
        view_axis=args.view_axis,
        front_mode=args.front_mode,
        rot_range_deg=args.rot_range_deg,
        trans_range=args.trans_range,
        pen_weight=args.pen_weight,
        dist_weight=args.dist_weight,
        rng_seed=args.rng_seed
    )

    # 将“绕中心”的微调变成标准 RT，再并入总变换
    refine_R, refine_t = rt_about_center_to_rt(refine_R_local, refine_t_local, refine_center)
    R_total, t_total = compose_rt(refine_R, refine_t, R_total, t_total)

    # -----------------------------------------------------
    # 6) 应用最终变换
    # -----------------------------------------------------
    final_sam_pts = apply_rt(sam_pts_scaled, R_total, t_total)

    # 把变换转成“从原始 source 坐标 -> target 坐标”的仿射形式
    # raw -> centered_scaled : x -> (x - sam_center) * locked_scale
    # final: x' = x @ (locked_scale * R_total).T + [t_total - sam_center @ (locked_scale * R_total).T]
    A_total = locked_scale * R_total
    t_affine = t_total - sam_center @ A_total.T

    # -----------------------------------------------------
    # 7) 结果评估
    # -----------------------------------------------------
    tree_final = cKDTree(final_sam_pts)
    final_dist, _ = tree_final.query(part_pts, workers=-1)
    final_fit_15mm = float(np.mean(final_dist < 0.015))
    final_mean = float(np.mean(final_dist))
    final_median = float(np.median(final_dist))
    final_p90 = float(np.percentile(final_dist, 90))

    print("\n" + "=" * 60)
    print("✅ 配准完成")
    print(f"   覆盖率 (<1.5cm): {final_fit_15mm * 100:.2f}%")
    print(f"   Mean dist      : {final_mean * 1000:.2f} mm")
    print(f"   Median dist    : {final_median * 1000:.2f} mm")
    print(f"   P90 dist       : {final_p90 * 1000:.2f} mm")
    print("=" * 60)

    # -----------------------------------------------------
    # 8) 保存结果
    # -----------------------------------------------------
    print("[4] 保存结果...")

    aligned_src_ply = os.path.join(args.out_dir, "aligned_source_to_partial.ply")
    merged_ply = os.path.join(args.out_dir, "vis_merged.ply")
    transform_json = os.path.join(args.out_dir, "final_transform.json")
    stats_json = os.path.join(args.out_dir, "stats.json")

    # aligned source
    save_point_cloud(aligned_src_ply, final_sam_pts, color=[0.10, 0.45, 0.85])

    # merged vis
    merged_pts = np.vstack([final_sam_pts, part_pts])
    merged_colors = np.vstack([
        np.tile([0.10, 0.45, 0.85], (len(final_sam_pts), 1)),
        np.tile([0.85, 0.10, 0.10], (len(part_pts), 1))
    ])
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(merged_pts)
    merged.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.io.write_point_cloud(merged_ply, merged)

    # 若 source 原本是 mesh，再导出对齐后的 mesh
    if sam_geom_type == "mesh":
        aligned_mesh_path = os.path.join(args.out_dir, "aligned_source_mesh.obj")
        save_mesh_with_affine(aligned_mesh_path, sam_geom_raw, A_total, t_affine)
        print(f"   已保存对齐 mesh: {aligned_mesh_path}")

    with open(transform_json, "w", encoding="utf-8") as f:
        json.dump({
            "note": "row-vector convention: x' = x @ A.T + t",
            "source_center_before_scale": sam_center.tolist(),
            "uniform_scale": float(locked_scale),
            "R_on_scaled_centered_source": R_total.tolist(),
            "t_on_scaled_centered_source": t_total.tolist(),
            "A_on_raw_source": A_total.tolist(),
            "t_on_raw_source": t_affine.tolist(),
        }, f, indent=2)

    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump({
            "gt_diag": float(gt_diag),
            "sam_diag": float(sam_diag),
            "partial_diag": float(part_diag),
            "locked_scale": float(locked_scale),

            "coarse_search": coarse_info,
            "icp_history": icp_history,
            "refine_info": refine_info,

            "final_fit_lt_15mm": final_fit_15mm,
            "final_mean_dist": final_mean,
            "final_median_dist": final_median,
            "final_p90_dist": final_p90,
        }, f, indent=2)

    print(f"   已保存对齐点云: {aligned_src_ply}")
    print(f"   已保存可视化  : {merged_ply}")
    print(f"   已保存变换参数: {transform_json}")
    print(f"   已保存统计信息: {stats_json}")
    print("\n🎯 全流程结束。")


if __name__ == "__main__":
    main()