#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import itertools
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot


# =========================================================
# 0. 基础工具
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def sample_points(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) <= n:
        return pts.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), n, replace=False)
    return pts[idx]


def clean_points(pts: np.ndarray, dedup_decimals=6) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]

    if len(pts) == 0:
        return pts

    rounded = np.round(pts, decimals=dedup_decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    return pts[idx]


def robust_diag(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 10:
        return 1.0

    p1 = np.percentile(pts, 2, axis=0)
    p2 = np.percentile(pts, 98, axis=0)
    d = float(np.linalg.norm(p2 - p1))
    return max(d, 1e-6)


def robust_projected_diag_xy(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 10:
        return 1.0

    proj = pts[:, :2]
    p1 = np.percentile(proj, 2, axis=0)
    p2 = np.percentile(proj, 98, axis=0)
    d = float(np.linalg.norm(p2 - p1))
    return max(d, 1e-6)


def load_points_any(path: str, mesh_samples: int = 50000, cache: bool = True) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    cache_path = path + f".cache_{mesh_samples}.npy"

    if cache and os.path.exists(cache_path):
        pts = np.load(cache_path).astype(np.float64)
        pts = clean_points(pts)
        print(f"[Cache] 使用固定 SAM 点云缓存: {cache_path}, points={len(pts)}")
        return pts

    mesh = o3d.io.read_triangle_mesh(path)

    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        try:
            o3d.utility.random.seed(42)
        except Exception:
            pass

        pcd = mesh.sample_points_uniformly(number_of_points=mesh_samples)
        pts = np.asarray(pcd.points).astype(np.float64)
        pts = clean_points(pts)

        if len(pts) > 0:
            if cache:
                np.save(cache_path, pts)
                print(f"[Cache] 已保存固定 SAM 点云缓存: {cache_path}")
            return pts

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points).astype(np.float64)
    pts = clean_points(pts)

    if len(pts) == 0:
        raise RuntimeError(f"无法从 {path} 读取有效点云/网格。")

    return pts


def save_point_cloud(path, pts, color=None):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color is not None:
        c = np.asarray(color, dtype=np.float64).reshape(1, 3)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(pts), 1)))

    o3d.io.write_point_cloud(path, pcd)


def save_merged_point_cloud(
    path,
    pts_a,
    pts_b,
    color_a=(0.10, 0.45, 0.85),
    color_b=(0.85, 0.10, 0.10),
):
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 3)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([pts_a, pts_b]))
    pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([
            np.tile(np.asarray(color_a, dtype=np.float64), (len(pts_a), 1)),
            np.tile(np.asarray(color_b, dtype=np.float64), (len(pts_b), 1)),
        ])
    )

    o3d.io.write_point_cloud(path, pcd)


def euler_to_R(rx, ry, rz):
    rx, ry, rz = np.radians([rx, ry, rz])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotz(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rotate_about_center(points: np.ndarray, R: np.ndarray, center: np.ndarray):
    return (points - center.reshape(1, 3)) @ R.T + center.reshape(1, 3)


def apply_rt_np(pts: np.ndarray, R: np.ndarray, t: np.ndarray):
    return pts @ R.T + t.reshape(1, 3)


# =========================================================
# 1. 24 adapter + quaternion 候选
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
        raise RuntimeError(f"rotation_quat 长度不是 4: {raw_quat}")

    cands = []

    # raw = [w, x, y, z]
    w, x, y, z = raw_quat
    R1 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("wxyz_R", R1))
    cands.append(("wxyz_RT", R1.T))

    # raw = [x, y, z, w]
    x, y, z, w = raw_quat
    R2 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("xyzw_R", R2))
    cands.append(("xyzw_RT", R2.T))

    return cands


def compose_candidate_rotation(base_R: np.ndarray, adapter_R: np.ndarray, compose_mode: str):
    if compose_mode == "left":
        return adapter_R @ base_R

    if compose_mode == "right":
        return base_R @ adapter_R

    raise ValueError(f"未知 compose_mode: {compose_mode}")


# =========================================================
# 2. 向量化 visible shell
# =========================================================
def extract_visible_shell_camera_fast(
    pts: np.ndarray,
    grid_size: float = 0.003,
    shell_thickness: float = 0.004,
    front_mode: str = "min",
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    if len(pts) == 0:
        return pts.copy()

    xy = pts[:, :2]
    z = pts[:, 2]

    g = max(float(grid_size), 1e-9)
    xy_min = np.min(xy, axis=0)
    ij = np.floor((xy - xy_min) / g).astype(np.int64)

    i = ij[:, 0]
    j = ij[:, 1]

    j_shift = j - j.min()
    width = int(j_shift.max()) + 1
    cell = (i - i.min()) * width + j_shift

    if front_mode == "min":
        order = np.lexsort((z, cell))
    else:
        order = np.lexsort((-z, cell))

    cell_sorted = cell[order]
    z_sorted = z[order]

    unique_cell, first_idx = np.unique(cell_sorted, return_index=True)
    front_z_per_unique = z_sorted[first_idx]

    inv = np.searchsorted(unique_cell, cell)
    front_z = front_z_per_unique[inv]

    if front_mode == "min":
        mask = z <= front_z + shell_thickness
    else:
        mask = z >= front_z - shell_thickness

    out = pts[mask]

    if len(out) < 50:
        return pts.copy()

    return out


# =========================================================
# 3. 2.5D / Z-buffer loss + 双向距离 + 悬空惩罚
# =========================================================
def get_comprehensive_loss_fast(
    partial_pts,
    comp_pts,
    pixel_size=0.002,
    front_mode="min",
    max_cells=900_000,
):
    partial_pts = np.asarray(partial_pts, dtype=np.float64).reshape(-1, 3)
    comp_pts = np.asarray(comp_pts, dtype=np.float64).reshape(-1, 3)

    if len(partial_pts) < 10 or len(comp_pts) < 10:
        return 1.0, 1.0, 1.0

    px, py, pd = partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2]
    cx, cy, cd = comp_pts[:, 0], comp_pts[:, 1], comp_pts[:, 2]

    x_min = min(px.min(), cx.min()) - pixel_size * 5
    y_min = min(py.min(), cy.min()) - pixel_size * 5
    x_max = max(px.max(), cx.max()) + pixel_size * 5
    y_max = max(py.max(), cy.max()) + pixel_size * 5

    ps = float(pixel_size)

    w = int(np.ceil((x_max - x_min) / ps)) + 1
    h = int(np.ceil((y_max - y_min) / ps)) + 1

    cells = w * h

    if cells > max_cells:
        scale = np.sqrt(cells / max_cells)
        ps = ps * scale
        w = int(np.ceil((x_max - x_min) / ps)) + 1
        h = int(np.ceil((y_max - y_min) / ps)) + 1

    pxi = np.clip(((px - x_min) / ps).astype(np.int32), 0, w - 1)
    pyi = np.clip(((py - y_min) / ps).astype(np.int32), 0, h - 1)

    cxi = np.clip(((cx - x_min) / ps).astype(np.int32), 0, w - 1)
    cyi = np.clip(((cy - y_min) / ps).astype(np.int32), 0, h - 1)

    mask_p = np.zeros((w, h), dtype=bool)
    mask_c = np.zeros((w, h), dtype=bool)

    mask_p[pxi, pyi] = True
    mask_c[cxi, cyi] = True

    overlap = mask_p & mask_c

    p_area = max(int(np.sum(mask_p)), 1)
    c_area = max(int(np.sum(mask_c)), 1)

    red_uncovered = np.sum(mask_p & (~mask_c)) / p_area
    blue_uncovered = np.sum(mask_c & (~mask_p)) / c_area

    contour_error = 0.5 * float(red_uncovered + blue_uncovered)

    if np.sum(overlap) == 0:
        return contour_error, 0.02, ps

    if front_mode == "min":
        z_buffer_p = np.full((w, h), np.inf, dtype=np.float64)
        z_buffer_c = np.full((w, h), np.inf, dtype=np.float64)

        np.minimum.at(z_buffer_p, (pxi, pyi), pd)
        np.minimum.at(z_buffer_c, (cxi, cyi), cd)

        pen = np.clip(z_buffer_p[overlap] - z_buffer_c[overlap], 0, None)
    else:
        z_buffer_p = np.full((w, h), -np.inf, dtype=np.float64)
        z_buffer_c = np.full((w, h), -np.inf, dtype=np.float64)

        np.maximum.at(z_buffer_p, (pxi, pyi), pd)
        np.maximum.at(z_buffer_c, (cxi, cyi), cd)

        pen = np.clip(z_buffer_c[overlap] - z_buffer_p[overlap], 0, None)

    mean_pen = float(np.mean(pen)) if pen.size > 0 else 0.0
    return contour_error, mean_pen, ps


def trimmed_mean(x, ratio=0.95):
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if len(x) == 0:
        return 1.0

    k = max(1, int(len(x) * ratio))
    xs = np.partition(x, k - 1)[:k]
    return float(np.mean(xs))


def hybrid_score(
    partial_pts,
    shell_pts,
    pixel_size=0.002,
    front_mode="min",
    dist_clip=0.05,
    trim_ratio=0.95,
    contour_weight=1.0,
    pen_weight=120.0,
    dist_weight=15.0,
    fit_weight=0.50,
    overhang_weight=50.0,
):
    partial_pts = np.asarray(partial_pts, dtype=np.float64).reshape(-1, 3)
    shell_pts = np.asarray(shell_pts, dtype=np.float64).reshape(-1, 3)

    if len(partial_pts) < 20 or len(shell_pts) < 20:
        return 1e9, {
            "score": 1e9,
            "contour_error": 1.0,
            "penetration": 1.0,
            "dist_partial_to_shell": 1.0,
            "dist_shell_to_partial": 1.0,
            "fit_partial_to_shell_15mm": 0.0,
            "fit_shell_to_partial_15mm": 0.0,
            "overhang_ratio": 1.0,
        }

    contour_error, pen, effective_pixel = get_comprehensive_loss_fast(
        partial_pts,
        shell_pts,
        pixel_size=pixel_size,
        front_mode=front_mode,
    )

    tree_shell = cKDTree(shell_pts)
    dist_ps, _ = tree_shell.query(partial_pts, workers=-1)

    tree_part = cKDTree(partial_pts)
    dist_sp, _ = tree_part.query(shell_pts, workers=-1)

    overhang_ratio = float(np.mean(dist_ps > 0.020))

    dist_ps_clip = np.minimum(dist_ps, dist_clip)
    dist_sp_clip = np.minimum(dist_sp, dist_clip)

    d_ps = trimmed_mean(dist_ps_clip, trim_ratio)
    d_sp = trimmed_mean(dist_sp_clip, trim_ratio)

    fit_ps = float(np.mean(dist_ps < 0.015))
    fit_sp = float(np.mean(dist_sp < 0.015))

    score = (
        contour_weight * contour_error
        + pen_weight * pen
        + dist_weight * 0.5 * (d_ps + d_sp)
        + fit_weight * (1.0 - 0.5 * (fit_ps + fit_sp))
        + overhang_weight * overhang_ratio
    )

    metrics = {
        "score": float(score),
        "contour_error": float(contour_error),
        "penetration": float(pen),
        "dist_partial_to_shell": float(d_ps),
        "dist_shell_to_partial": float(d_sp),
        "fit_partial_to_shell_15mm": float(fit_ps),
        "fit_shell_to_partial_15mm": float(fit_sp),
        "overhang_ratio": float(overhang_ratio),
        "effective_pixel_size": float(effective_pixel),
    }

    return float(score), metrics


# =========================================================
# 4. 平移、尺度校准
# =========================================================
def translation_only_icp_fast(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iter: int = 12,
    trim_ratio: float = 0.90,
):
    src = np.asarray(source_pts, dtype=np.float64).copy()
    tgt = np.asarray(target_pts, dtype=np.float64)

    t_total = np.zeros(3, dtype=np.float64)

    t0 = np.zeros(3, dtype=np.float64)
    t0[0] = np.median(tgt[:, 0]) - np.median(src[:, 0])
    t0[1] = np.median(tgt[:, 1]) - np.median(src[:, 1])
    t0[2] = np.percentile(tgt[:, 2], 5) - np.percentile(src[:, 2], 5)

    src += t0.reshape(1, 3)
    t_total += t0

    for _ in range(max_iter):
        tree = cKDTree(src)
        dist, idx = tree.query(tgt, workers=-1)

        thr = np.percentile(dist, trim_ratio * 100.0)
        mask = dist <= thr

        if np.sum(mask) < 20:
            break

        delta_t = np.mean(tgt[mask] - src[idx[mask]], axis=0)

        src += delta_t
        t_total += delta_t

        if np.linalg.norm(delta_t) < 1e-6:
            break

    tree = cKDTree(src)
    final_dist, _ = tree.query(tgt, workers=-1)

    fit15 = float(np.mean(final_dist < 0.015))
    rmse = float(np.sqrt(np.mean(final_dist ** 2)))

    return t_total, fit15, rmse, src


def calibrate_scale_from_partial_fast(
    shell_pts: np.ndarray,
    partial_pts: np.ndarray,
    search_min: float = 0.94,
    search_max: float = 1.06,
    search_steps: int = 9,
    icp_iter: int = 10,
    pixel_size: float = 0.002,
    front_mode: str = "min",
):
    shell_pts = np.asarray(shell_pts, dtype=np.float64)
    partial_pts = np.asarray(partial_pts, dtype=np.float64)

    if len(shell_pts) < 30 or len(partial_pts) < 30:
        return None

    shell_center = np.mean(shell_pts, axis=0)
    shell_local = shell_pts - shell_center

    shell_diag = robust_projected_diag_xy(shell_local)
    part_diag = robust_projected_diag_xy(partial_pts)

    if shell_diag < 1e-9 or part_diag < 1e-9:
        return None

    s0 = part_diag / shell_diag

    best = None
    best_score = 1e18

    for s in s0 * np.linspace(search_min, search_max, search_steps):
        shell_scaled = shell_local * s + shell_center

        t_est, fit15, rmse, shell_aligned = translation_only_icp_fast(
            shell_scaled,
            partial_pts,
            max_iter=icp_iter,
            trim_ratio=0.90,
        )

        score, metrics = hybrid_score(
            partial_pts,
            shell_aligned,
            pixel_size=pixel_size,
            front_mode=front_mode,
        )

        if score < best_score:
            best_score = score
            best = {
                "scale": float(s),
                "translation": t_est.copy(),
                "fit15": float(fit15),
                "rmse": float(rmse),
                "score": float(score),
                "metrics": metrics,
                "shell_aligned": shell_aligned,
            }

    return best


# =========================================================
# 5. 局部随机可见性精修
# =========================================================
def local_visibility_refinement_fast(
    full_pts,
    shell_pts,
    partial_pts,
    iterations=80,
    pixel_size=0.002,
    front_mode="min",
    rot_range_deg=2.5,
    trans_range=0.004,
    rng_seed=42,
    eval_shell_samples=4000,
    eval_partial_samples=4000,
):
    rng = np.random.default_rng(rng_seed)

    full_best = np.asarray(full_pts, dtype=np.float64).copy()
    shell_best = np.asarray(shell_pts, dtype=np.float64).copy()

    partial_eval = sample_points(partial_pts, eval_partial_samples, seed=rng_seed + 11)
    shell_eval = sample_points(shell_best, eval_shell_samples, seed=rng_seed + 12)

    best_score, best_metrics = hybrid_score(
        partial_eval,
        shell_eval,
        pixel_size=pixel_size,
        front_mode=front_mode,
    )

    center = np.mean(shell_best, axis=0)

    print(
        f"    [精修初始] score={best_score:.6f} | "
        f"contour={best_metrics['contour_error']*100:.2f}% | "
        f"pen={best_metrics['penetration']*1000:.2f}mm | "
        f"overhang={best_metrics.get('overhang_ratio', 0)*100:.2f}% | "
        f"fitPS={best_metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
        f"fitSP={best_metrics['fit_shell_to_partial_15mm']*100:.2f}%"
    )

    for i in range(iterations):
        anneal = 1.0 - 0.75 * (i / max(iterations - 1, 1))

        dr = rng.uniform(-rot_range_deg, rot_range_deg, 3) * anneal
        dt = rng.uniform(-trans_range, trans_range, 3) * anneal

        R_try = euler_to_R(dr[0], dr[1], dr[2])

        shell_try = rotate_about_center(shell_best, R_try, center) + dt.reshape(1, 3)
        shell_try_eval = sample_points(shell_try, eval_shell_samples, seed=rng_seed + 100 + i)

        score, metrics = hybrid_score(
            partial_eval,
            shell_try_eval,
            pixel_size=pixel_size,
            front_mode=front_mode,
        )

        if score < best_score:
            best_score = score
            best_metrics = metrics

            full_best = rotate_about_center(full_best, R_try, center) + dt.reshape(1, 3)
            shell_best = shell_try
            center = np.mean(shell_best, axis=0)

            if i % 10 == 0 or i == iterations - 1:
                print(
                    f"    [精修 {i:03d}] score={best_score:.6f} | "
                    f"contour={metrics['contour_error']*100:.2f}% | "
                    f"pen={metrics['penetration']*1000:.2f}mm | "
                    f"overhang={metrics.get('overhang_ratio', 0)*100:.2f}% | "
                    f"fitPS={metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
                    f"fitSP={metrics['fit_shell_to_partial_15mm']*100:.2f}%"
                )

    print(
        f"    [精修完成] score={best_score:.6f} | "
        f"contour={best_metrics['contour_error']*100:.2f}% | "
        f"pen={best_metrics['penetration']*1000:.2f}mm | "
        f"overhang={best_metrics.get('overhang_ratio', 0)*100:.2f}% | "
        f"fitPS={best_metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
        f"fitSP={best_metrics['fit_shell_to_partial_15mm']*100:.2f}%"
    )

    return full_best, shell_best, {
        "score": float(best_score),
        **best_metrics,
    }


# =========================================================
# 6. 最终 SVD-KDTree 表面贴合收紧
# =========================================================
def best_fit_transform_svd(A: np.ndarray, B: np.ndarray):
    A = np.asarray(A, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(B, dtype=np.float64).reshape(-1, 3)

    ca = np.mean(A, axis=0)
    cb = np.mean(B, axis=0)

    AA = A - ca
    BB = B - cb

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cb - ca @ R.T
    return R, t


def compact_surface_refinement(
    full_pts: np.ndarray,
    shell_pts: np.ndarray,
    partial_pts: np.ndarray,
    iterations: int = 35,
    trim_ratio: float = 0.86,
    max_corr: float = 0.020,
    max_step_trans: float = 0.0025,
    max_step_rot_deg: float = 1.5,
    front_mode: str = "min",
    pixel_size: float = 0.002,
    rng_seed: int = 42,
    eval_samples: int = 5000,
):
    full_curr = np.asarray(full_pts, dtype=np.float64).copy()
    shell_curr = np.asarray(shell_pts, dtype=np.float64).copy()
    partial_pts = np.asarray(partial_pts, dtype=np.float64).reshape(-1, 3)

    partial_eval = sample_points(partial_pts, min(len(partial_pts), eval_samples), seed=rng_seed + 701)

    best_score, best_metrics = hybrid_score(
        partial_eval,
        sample_points(shell_curr, min(len(shell_curr), eval_samples), seed=rng_seed + 702),
        pixel_size=pixel_size,
        front_mode=front_mode,
    )

    best_full = full_curr.copy()
    best_shell = shell_curr.copy()

    print(
        f"    [紧贴初始] score={best_score:.6f} | "
        f"fitPS={best_metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
        f"fitSP={best_metrics['fit_shell_to_partial_15mm']*100:.2f}% | "
        f"dPS={best_metrics['dist_partial_to_shell']*1000:.2f}mm | "
        f"dSP={best_metrics['dist_shell_to_partial']*1000:.2f}mm | "
        f"overhang={best_metrics.get('overhang_ratio', 0)*100:.2f}%"
    )

    no_gain = 0

    for it in range(iterations):
        shell_eval = sample_points(shell_curr, min(len(shell_curr), eval_samples), seed=rng_seed + 800 + it)

        tree_p = cKDTree(partial_eval)
        dist_sp, idx_sp = tree_p.query(shell_eval, workers=-1)

        tree_s = cKDTree(shell_eval)
        dist_ps, idx_ps = tree_s.query(partial_eval, workers=-1)

        if len(dist_sp) < 30 or len(dist_ps) < 30:
            break

        th_sp = min(max_corr, np.percentile(dist_sp, trim_ratio * 100.0))
        th_ps = min(max_corr, np.percentile(dist_ps, trim_ratio * 100.0))

        m_sp = dist_sp <= th_sp
        m_ps = dist_ps <= th_ps

        src_pairs = []
        tgt_pairs = []

        # shell -> partial
        if np.sum(m_sp) >= 20:
            src_pairs.append(shell_eval[m_sp])
            tgt_pairs.append(partial_eval[idx_sp[m_sp]])

        # partial -> shell
        if np.sum(m_ps) >= 20:
            src_pairs.append(shell_eval[idx_ps[m_ps]])
            tgt_pairs.append(partial_eval[m_ps])

        if len(src_pairs) == 0:
            print(f"    [紧贴 {it:02d}] 有效匹配太少，停止。")
            break

        A = np.vstack(src_pairs)
        B = np.vstack(tgt_pairs)

        if len(A) < 30:
            break

        R_step, t_step = best_fit_transform_svd(A, B)

        rot_obj = SciRot.from_matrix(R_step)
        rotvec = rot_obj.as_rotvec()
        angle = np.linalg.norm(rotvec)
        max_angle = np.deg2rad(max_step_rot_deg)

        if angle > max_angle and angle > 1e-12:
            rotvec = rotvec / angle * max_angle
            R_step = SciRot.from_rotvec(rotvec).as_matrix()

        t_norm = np.linalg.norm(t_step)

        if t_norm > max_step_trans and t_norm > 1e-12:
            t_step = t_step / t_norm * max_step_trans

        shell_try = apply_rt_np(shell_curr, R_step, t_step)
        full_try = apply_rt_np(full_curr, R_step, t_step)

        try_score, try_metrics = hybrid_score(
            partial_eval,
            sample_points(shell_try, min(len(shell_try), eval_samples), seed=rng_seed + 900 + it),
            pixel_size=pixel_size,
            front_mode=front_mode,
        )

        if try_score < best_score:
            shell_curr = shell_try
            full_curr = full_try

            best_score = try_score
            best_metrics = try_metrics
            best_shell = shell_curr.copy()
            best_full = full_curr.copy()
            no_gain = 0

            print(
                f"    [紧贴 {it:02d}] score={best_score:.6f} | "
                f"fitPS={best_metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
                f"fitSP={best_metrics['fit_shell_to_partial_15mm']*100:.2f}% | "
                f"dPS={best_metrics['dist_partial_to_shell']*1000:.2f}mm | "
                f"dSP={best_metrics['dist_shell_to_partial']*1000:.2f}mm | "
                f"overhang={best_metrics.get('overhang_ratio', 0)*100:.2f}% | "
                f"step_t={np.linalg.norm(t_step)*1000:.2f}mm"
            )
        else:
            no_gain += 1
            max_step_trans *= 0.72
            max_step_rot_deg *= 0.72

            if no_gain >= 8 or max_step_trans < 0.00035:
                print(f"    [紧贴 {it:02d}] 连续无提升或步长过小，停止。")
                break

    print(
        f"    [紧贴完成] score={best_score:.6f} | "
        f"fitPS={best_metrics['fit_partial_to_shell_15mm']*100:.2f}% | "
        f"fitSP={best_metrics['fit_shell_to_partial_15mm']*100:.2f}% | "
        f"dPS={best_metrics['dist_partial_to_shell']*1000:.2f}mm | "
        f"dSP={best_metrics['dist_shell_to_partial']*1000:.2f}mm | "
        f"overhang={best_metrics.get('overhang_ratio', 0)*100:.2f}%"
    )

    return best_full, best_shell, {
        "score": float(best_score),
        **best_metrics,
    }


# =========================================================
# 7. 主配准逻辑
# =========================================================
def restore_canonical_to_rgb_pose_optimized(
    sam_pts_raw: np.ndarray,
    part_pts: np.ndarray,
    pose_json_path: str,
    front_mode: str = "min",
    pixel_size: float = 0.002,
    coarse_shell_samples: int = 2500,
    coarse_partial_samples: int = 2500,
    fine_shell_samples: int = 4000,
    fine_partial_samples: int = 4000,
    topk: int = 6,
    refine_iters: int = 80,
    rng_seed: int = 42,
    force_base_name: str = "",
    force_adapter_idx: int = -1,
    force_compose_mode: str = "",
    compact_iters: int = 35,
    compact_trim: float = 0.86,
    compact_max_corr: float = 0.020,
    compact_step_trans: float = 0.0025,
    compact_step_rot: float = 1.5,
):
    print("[1] 正在读取 pose json ...", flush=True)

    with open(pose_json_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if "rotation_quat" not in pose_data:
        raise RuntimeError(f"{pose_json_path} 里没有 rotation_quat")

    raw_quat = pose_data["rotation_quat"]
    base_rotations = decode_quaternion_candidates(raw_quat)
    adapters = get_24_rotations()

    sam_center = np.mean(sam_pts_raw, axis=0)
    sam_centered = sam_pts_raw - sam_center

    part_diag = robust_diag(part_pts)
    grid_size = part_diag / 70.0
    shell_thickness = part_diag / 120.0

    print(f"[2] quaternion 基础解释数: {len(base_rotations)}")
    print(f"[3] adapter 数: {len(adapters)}")
    print(f"[4] grid_size={grid_size:.6f}, shell_thickness={shell_thickness:.6f}")

    part_sub_coarse = sample_points(part_pts, coarse_partial_samples, seed=rng_seed + 1)

    # -----------------------------------------------------
    # Stage A: 粗搜索
    # -----------------------------------------------------
    print("[5] Stage A：粗搜索 rotation + partial 尺度校准 ...", flush=True)

    coarse_results = []
    cnt = 0
    total = len(base_rotations) * len(adapters) * 2

    for base_name, base_R in base_rotations:
        for ai, A in enumerate(adapters):
            for compose_mode in ["left", "right"]:
                cnt += 1

                if force_base_name and base_name != force_base_name:
                    continue

                if force_adapter_idx >= 0 and ai != force_adapter_idx:
                    continue

                if force_compose_mode and compose_mode != force_compose_mode:
                    continue

                R_candidate = compose_candidate_rotation(base_R, A, compose_mode)
                sam_rot = sam_centered @ R_candidate.T

                shell_rot = extract_visible_shell_camera_fast(
                    sam_rot,
                    grid_size=grid_size,
                    shell_thickness=shell_thickness,
                    front_mode=front_mode,
                )

                if len(shell_rot) < 100:
                    continue

                shell_sub = sample_points(shell_rot, coarse_shell_samples, seed=rng_seed + cnt)

                scale_res = calibrate_scale_from_partial_fast(
                    shell_sub,
                    part_sub_coarse,
                    search_min=0.94,
                    search_max=1.06,
                    search_steps=9,
                    icp_iter=10,
                    pixel_size=pixel_size,
                    front_mode=front_mode,
                )

                if scale_res is None:
                    continue

                coarse_results.append({
                    "base_name": base_name,
                    "adapter_idx": ai,
                    "compose_mode": compose_mode,
                    "R_candidate": R_candidate.copy(),
                    "scale": scale_res["scale"],
                    "translation": scale_res["translation"].copy(),
                    "score": scale_res["score"],
                    "metrics": scale_res["metrics"],
                })

                if cnt % 40 == 0 or cnt == total:
                    print(f"    [粗搜索] {cnt}/{total} 已完成", flush=True)

    if len(coarse_results) == 0:
        raise RuntimeError("粗搜索失败：没有找到有效 RGB 粗姿态。")

    coarse_results = sorted(coarse_results, key=lambda x: x["score"])[:topk]

    print(f"[6] 粗搜索 top-{len(coarse_results)}:")

    for i, c in enumerate(coarse_results, 1):
        m = c["metrics"]
        print(
            f"    #{i}: {c['base_name']} + A[{c['adapter_idx']}] + {c['compose_mode']} | "
            f"scale={c['scale']:.6f} | score={c['score']:.6f} | "
            f"contour={m['contour_error']*100:.2f}% | "
            f"overhang={m.get('overhang_ratio', 0)*100:.2f}% | "
            f"fitPS={m['fit_partial_to_shell_15mm']*100:.2f}% | "
            f"fitSP={m['fit_shell_to_partial_15mm']*100:.2f}%"
        )

    # -----------------------------------------------------
    # Stage B: top-k 上细化 scale + yaw
    # -----------------------------------------------------
    print("[7] Stage B：top-k 上做 scale + yaw 细搜索 ...", flush=True)

    best = None
    best_score = 1e18

    part_sub_fine = sample_points(part_pts, fine_partial_samples, seed=rng_seed + 300)

    for ci, cand in enumerate(coarse_results):
        R_candidate = cand["R_candidate"]

        for s in cand["scale"] * np.linspace(0.97, 1.03, 9):
            sam_rgb_base = (sam_centered * s) @ R_candidate.T
            rot_center = np.mean(sam_rgb_base, axis=0)

            for yaw_deg in np.arange(-8.0, 8.01, 1.0):
                R_yaw = rotz(yaw_deg)
                sam_rgb_yaw = rotate_about_center(sam_rgb_base, R_yaw, rot_center)

                shell = extract_visible_shell_camera_fast(
                    sam_rgb_yaw,
                    grid_size=part_diag / 80.0,
                    shell_thickness=part_diag / 140.0,
                    front_mode=front_mode,
                )

                if len(shell) < 100:
                    continue

                shell_sub = sample_points(shell, fine_shell_samples, seed=rng_seed + 400 + ci)

                t_est, fit15, rmse, shell_aligned = translation_only_icp_fast(
                    shell_sub,
                    part_sub_fine,
                    max_iter=12,
                    trim_ratio=0.90,
                )

                score, metrics = hybrid_score(
                    part_sub_fine,
                    shell_aligned,
                    pixel_size=pixel_size,
                    front_mode=front_mode,
                )

                if score < best_score:
                    best_score = score
                    best = {
                        "base_name": cand["base_name"],
                        "adapter_idx": cand["adapter_idx"],
                        "compose_mode": cand["compose_mode"],
                        "R_candidate": R_candidate.copy(),
                        "scale": float(s),
                        "yaw_deg": float(yaw_deg),
                        "translation": t_est.copy(),
                        "score": float(score),
                        "metrics": metrics,
                    }

    if best is None:
        raise RuntimeError("细搜索失败：没有得到最终 RGB 位姿。")

    print("[8] 细搜索最佳：")
    print(
        f"    {best['base_name']} + A[{best['adapter_idx']}] + {best['compose_mode']} | "
        f"scale={best['scale']:.6f} | yaw={best['yaw_deg']:.2f}° | score={best['score']:.6f}"
    )
    print(
        f"    contour={best['metrics']['contour_error']*100:.2f}% | "
        f"pen={best['metrics']['penetration']*1000:.2f}mm | "
        f"overhang={best['metrics'].get('overhang_ratio', 0)*100:.2f}% | "
        f"fitPS={best['metrics']['fit_partial_to_shell_15mm']*100:.2f}% | "
        f"fitSP={best['metrics']['fit_shell_to_partial_15mm']*100:.2f}%"
    )

    # -----------------------------------------------------
    # Stage C: 构造完整点云 + 局部随机可见性精修
    # -----------------------------------------------------
    sam_rgb = (sam_centered * best["scale"]) @ best["R_candidate"].T

    rot_center = np.mean(sam_rgb, axis=0)
    sam_rgb = rotate_about_center(sam_rgb, rotz(best["yaw_deg"]), rot_center)

    sam_rgb = sam_rgb + best["translation"].reshape(1, 3)

    shell_rgb = extract_visible_shell_camera_fast(
        sam_rgb,
        grid_size=part_diag / 85.0,
        shell_thickness=part_diag / 145.0,
        front_mode=front_mode,
    )

    print("[9] Stage C：局部 2.5D 可见性精修 ...", flush=True)

    sam_rgb_refined, shell_rgb_refined, refine_metrics = local_visibility_refinement_fast(
        full_pts=sam_rgb,
        shell_pts=shell_rgb,
        partial_pts=part_pts,
        iterations=refine_iters,
        pixel_size=pixel_size,
        front_mode=front_mode,
        rot_range_deg=2.5,
        trans_range=0.004,
        rng_seed=rng_seed,
        eval_shell_samples=fine_shell_samples,
        eval_partial_samples=fine_partial_samples,
    )

    # -----------------------------------------------------
    # Stage D: 最终确定性表面贴合收紧
    # -----------------------------------------------------
    print("[10] Stage D：SVD-KDTree 表面贴合收紧 ...", flush=True)

    sam_rgb_refined, shell_rgb_refined, compact_metrics = compact_surface_refinement(
        full_pts=sam_rgb_refined,
        shell_pts=shell_rgb_refined,
        partial_pts=part_pts,
        iterations=compact_iters,
        trim_ratio=compact_trim,
        max_corr=compact_max_corr,
        max_step_trans=compact_step_trans,
        max_step_rot_deg=compact_step_rot,
        front_mode=front_mode,
        pixel_size=pixel_size,
        rng_seed=rng_seed,
        eval_samples=max(fine_shell_samples, fine_partial_samples),
    )

    shell_final = extract_visible_shell_camera_fast(
        sam_rgb_refined,
        grid_size=part_diag / 95.0,
        shell_thickness=part_diag / 120.0,
        front_mode=front_mode,
    )

    return {
        "full_rgb_pose": sam_rgb_refined,
        "visible_shell": shell_final,
        "scale": best["scale"],
        "yaw_deg": best["yaw_deg"],
        "translation": best["translation"],
        "base_name": best["base_name"],
        "adapter_idx": best["adapter_idx"],
        "compose_mode": best["compose_mode"],
        "stage_b_score": best["score"],
        "stage_b_metrics": best["metrics"],
        "refine_metrics": refine_metrics,
        "compact_metrics": compact_metrics,
    }


# =========================================================
# 8. CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="partial 锁尺度 RGB pose 配准：候选锁定 + 悬空惩罚 + SVD-KDTree 紧贴收敛"
    )

    parser.add_argument("--sam3d", required=True, help="SAM-3D canonical 完整点云 / 网格")
    parser.add_argument("--partial", required=True, help="真实 depth partial 点云")
    parser.add_argument("--pose-json", required=True, help="sam3d_pose.json")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--sam-samples", type=int, default=50000)
    parser.add_argument("--front-mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument("--pixel-size", type=float, default=0.002)

    parser.add_argument("--coarse-shell-samples", type=int, default=2500)
    parser.add_argument("--coarse-partial-samples", type=int, default=2500)
    parser.add_argument("--fine-shell-samples", type=int, default=4000)
    parser.add_argument("--fine-partial-samples", type=int, default=4000)

    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--refine-iters", type=int, default=80)
    parser.add_argument("--rng-seed", type=int, default=42)

    # 候选锁定
    parser.add_argument("--force-base-name", type=str, default="", help="强制 quaternion 解释，例如 wxyz_R")
    parser.add_argument("--force-adapter-idx", type=int, default=-1, help="强制 adapter idx，例如 8")
    parser.add_argument("--force-compose-mode", type=str, default="", choices=["", "left", "right"])

    # 最后紧贴参数
    parser.add_argument("--compact-iters", type=int, default=35)
    parser.add_argument("--compact-trim", type=float, default=0.86)
    parser.add_argument("--compact-max-corr", type=float, default=0.020)
    parser.add_argument("--compact-step-trans", type=float, default=0.0025)
    parser.add_argument("--compact-step-rot", type=float, default=1.5)

    args = parser.parse_args()

    t0 = time.time()

    np.random.seed(args.rng_seed)

    try:
        o3d.utility.random.seed(args.rng_seed)
    except Exception:
        pass

    ensure_dir(args.out_dir)

    print("\n" + "=" * 70)
    print("[🚀] 启动：partial 锁尺度 RGB pose 配准 + SVD-KDTree 紧贴收敛")
    print("=" * 70)
    print(f"[Args] sam3d      = {args.sam3d}")
    print(f"[Args] partial    = {args.partial}")
    print(f"[Args] pose-json  = {args.pose_json}")
    print(f"[Args] out-dir    = {args.out_dir}")
    print(f"[Args] front-mode = {args.front_mode}")
    print(f"[Args] pixel-size = {args.pixel_size}")
    print(f"[Args] topk       = {args.topk}")
    print(f"[Args] refine     = {args.refine_iters}")
    print(f"[Args] compact    = {args.compact_iters} iters, corr={args.compact_max_corr}, step={args.compact_step_trans}m")

    if args.force_base_name or args.force_adapter_idx >= 0 or args.force_compose_mode:
        print(
            f"[Args] force      = base={args.force_base_name}, "
            f"adapter={args.force_adapter_idx}, compose={args.force_compose_mode}"
        )

    print("=" * 70)

    print("[0] 读取点云 ...")

    sam_pts_raw = load_points_any(args.sam3d, mesh_samples=args.sam_samples, cache=True)
    part_pts = load_points_any(args.partial, mesh_samples=args.sam_samples, cache=False)

    print(f"    SAM points     = {len(sam_pts_raw)}")
    print(f"    Partial points = {len(part_pts)}")

    result = restore_canonical_to_rgb_pose_optimized(
        sam_pts_raw=sam_pts_raw,
        part_pts=part_pts,
        pose_json_path=args.pose_json,
        front_mode=args.front_mode,
        pixel_size=args.pixel_size,
        coarse_shell_samples=args.coarse_shell_samples,
        coarse_partial_samples=args.coarse_partial_samples,
        fine_shell_samples=args.fine_shell_samples,
        fine_partial_samples=args.fine_partial_samples,
        topk=args.topk,
        refine_iters=args.refine_iters,
        rng_seed=args.rng_seed,
        force_base_name=args.force_base_name,
        force_adapter_idx=args.force_adapter_idx,
        force_compose_mode=args.force_compose_mode,
        compact_iters=args.compact_iters,
        compact_trim=args.compact_trim,
        compact_max_corr=args.compact_max_corr,
        compact_step_trans=args.compact_step_trans,
        compact_step_rot=args.compact_step_rot,
    )

    full_rgb = result["full_rgb_pose"]
    shell_rgb = result["visible_shell"]

    final_score, final_metrics = hybrid_score(
        sample_points(part_pts, 3000, seed=args.rng_seed + 900),
        sample_points(shell_rgb, 3000, seed=args.rng_seed + 901),
        pixel_size=args.pixel_size,
        front_mode=args.front_mode,
    )

    tree_full = cKDTree(full_rgb)
    dist_partial_to_full, _ = tree_full.query(part_pts, workers=-1)

    final_fit_full_15 = float(np.mean(dist_partial_to_full < 0.015))
    final_rmse_full = float(np.sqrt(np.mean(dist_partial_to_full ** 2)))

    full_path = os.path.join(args.out_dir, "full_rgb_pose.ply")
    shell_path = os.path.join(args.out_dir, "visible_rgb_shell.ply")
    merged_path = os.path.join(args.out_dir, "merged_rgb_pose.ply")
    result_json = os.path.join(args.out_dir, "pose_decode_result.json")

    save_point_cloud(full_path, full_rgb, color=[0.10, 0.45, 0.85])
    save_point_cloud(shell_path, shell_rgb, color=[0.10, 0.85, 0.20])
    save_merged_point_cloud(merged_path, full_rgb, part_pts)

    info = {
        "script": "step_pose-part.py",
        "base_name": result["base_name"],
        "adapter_idx": int(result["adapter_idx"]),
        "compose_mode": result["compose_mode"],
        "scale": float(result["scale"]),
        "yaw_deg": float(result["yaw_deg"]),
        "translation": result["translation"].tolist(),
        "stage_b_score": float(result["stage_b_score"]),
        "stage_b_metrics": result["stage_b_metrics"],
        "refine_metrics": result["refine_metrics"],
        "compact_metrics": result.get("compact_metrics", {}),
        "final_hybrid_score": float(final_score),
        "final_hybrid_metrics": final_metrics,
        "final_fit_partial_to_full_15mm": final_fit_full_15,
        "final_rmse_partial_to_full": final_rmse_full,
        "front_mode": args.front_mode,
        "pixel_size": args.pixel_size,
        "sam3d": args.sam3d,
        "partial": args.partial,
        "pose_json": args.pose_json,
        "runtime_sec": time.time() - t0,
        "note": "candidate locking + overhang penalty + final SVD-KDTree compact refinement",
    }

    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("✅ RGB 视角位姿恢复完成")
    print(f"   最佳解释: {result['base_name']} + A[{result['adapter_idx']}] + {result['compose_mode']}")
    print(f"   scale:    {result['scale']:.6f}")
    print(f"   yaw:      {result['yaw_deg']:.2f}°")
    print(f"   final hybrid score: {final_score:.6f}")
    print(f"   final contour err : {final_metrics['contour_error']*100:.2f}%")
    print(f"   final penetration : {final_metrics['penetration']*1000:.2f} mm")
    print(f"   final overhang err: {final_metrics.get('overhang_ratio', 0)*100:.2f}%")

    if result.get("compact_metrics"):
        cm = result["compact_metrics"]
        print(
            f"   compact fitPS/SP : "
            f"{cm['fit_partial_to_shell_15mm']*100:.2f}% / "
            f"{cm['fit_shell_to_partial_15mm']*100:.2f}%"
        )
        print(
            f"   compact dPS/dSP  : "
            f"{cm['dist_partial_to_shell']*1000:.2f}mm / "
            f"{cm['dist_shell_to_partial']*1000:.2f}mm"
        )

    print(f"   partial->shell fit@1.5cm: {final_metrics['fit_partial_to_shell_15mm']*100:.2f}%")
    print(f"   shell->partial fit@1.5cm: {final_metrics['fit_shell_to_partial_15mm']*100:.2f}%")
    print(f"   partial->full fit@1.5cm : {final_fit_full_15*100:.2f}%")
    print(f"   partial->full RMSE      : {final_rmse_full:.4f} m")
    print(f"   runtime: {time.time() - t0:.2f} sec")
    print(f"   full_rgb_pose:     {full_path}")
    print(f"   visible_rgb_shell: {shell_path}")
    print(f"   merged_rgb_pose:   {merged_path}")
    print(f"   result_json:       {result_json}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()