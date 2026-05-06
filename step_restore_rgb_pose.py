#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import copy
import argparse
import itertools
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot


# =========================================================
# 0. 基础工具
# =========================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def sample_points(pts: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= n:
        return pts.copy()
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


def robust_diag(pts: np.ndarray) -> float:
    p1 = np.percentile(pts, 2, axis=0)
    p2 = np.percentile(pts, 98, axis=0)
    return float(np.linalg.norm(p2 - p1))


def robust_projected_diag_xy(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    proj = pts[:, :2]
    p1 = np.percentile(proj, 2, axis=0)
    p2 = np.percentile(proj, 98, axis=0)
    return float(np.linalg.norm(p2 - p1))


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


# =========================================================
# 1. 固定 RGB 相机坐标下的 visible shell
# =========================================================
def extract_visible_shell_camera(
    pts: np.ndarray,
    grid_size: float = 0.003,
    shell_thickness: float = 0.004,
    front_mode: str = "min",
) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) == 0:
        return pts.copy()

    xy = pts[:, :2]
    z = pts[:, 2]

    xy_min = np.min(xy, axis=0)
    ij = np.floor((xy - xy_min) / max(grid_size, 1e-9)).astype(np.int64)

    front_z = {}
    for i in range(len(pts)):
        key = (int(ij[i, 0]), int(ij[i, 1]))
        d = z[i]
        if key not in front_z:
            front_z[key] = d
        else:
            if front_mode == "min":
                if d < front_z[key]:
                    front_z[key] = d
            else:
                if d > front_z[key]:
                    front_z[key] = d

    mask = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        key = (int(ij[i, 0]), int(ij[i, 1]))
        d0 = front_z[key]
        if front_mode == "min":
            if z[i] <= d0 + shell_thickness:
                mask[i] = True
        else:
            if z[i] >= d0 - shell_thickness:
                mask[i] = True

    return pts[mask]


# =========================================================
# 2. 只优化平移
# =========================================================
def translation_only_icp(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_iter: int = 35,
    trim_ratio: float = 0.80,
):
    src = np.copy(source_pts)
    t_total = np.zeros(3, dtype=np.float64)

    t0 = np.zeros(3, dtype=np.float64)
    t0[0] = np.median(target_pts[:, 0]) - np.median(src[:, 0])
    t0[1] = np.median(target_pts[:, 1]) - np.median(src[:, 1])
    t0[2] = np.percentile(target_pts[:, 2], 5) - np.percentile(src[:, 2], 5)

    src += t0.reshape(1, 3)
    t_total += t0

    for _ in range(max_iter):
        tree = cKDTree(src)
        dist, idx = tree.query(target_pts, workers=-1)

        thr = np.percentile(dist, trim_ratio * 100.0)
        mask = dist <= thr
        if np.sum(mask) < 20:
            break

        delta_t = np.mean(target_pts[mask] - src[idx[mask]], axis=0)
        src += delta_t
        t_total += delta_t

        if np.linalg.norm(delta_t) < 1e-6:
            break

    tree = cKDTree(src)
    final_dist, _ = tree.query(target_pts, workers=-1)
    fit15 = float(np.mean(final_dist < 0.015))
    rmse = float(np.sqrt(np.mean(final_dist ** 2)))
    return t_total, fit15, rmse, src


# =========================================================
# 3. yaw / 小残差变换
# =========================================================
def rotz(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rotate_about_center_z(points: np.ndarray, angle_deg: float, center: np.ndarray):
    R = rotz(angle_deg)
    return ((points - center) @ R.T) + center


def apply_se2z(points: np.ndarray, yaw_deg: float, tx: float, ty: float, tz: float, center_xy: np.ndarray):
    pts = np.asarray(points, dtype=np.float64).copy()
    a = np.radians(yaw_deg)
    c, s = np.cos(a), np.sin(a)
    R2 = np.array([[c, -s],
                   [s,  c]], dtype=np.float64)

    xy = pts[:, :2] - center_xy.reshape(1, 2)
    xy = xy @ R2.T
    xy = xy + center_xy.reshape(1, 2)
    xy[:, 0] += tx
    xy[:, 1] += ty
    pts[:, :2] = xy
    pts[:, 2] += tz
    return pts


# =========================================================
# 4. quaternion 解码候选
# =========================================================
def decode_quaternion_candidates(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64).reshape(-1)
    if len(raw_quat) != 4:
        raise RuntimeError(f"rotation_quat 长度不是 4: {raw_quat}")

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


def compose_candidate_rotation(base_R: np.ndarray, adapter_R: np.ndarray, compose_mode: str):
    if compose_mode == "left":
        return adapter_R @ base_R
    elif compose_mode == "right":
        return base_R @ adapter_R
    else:
        raise ValueError(f"未知 compose_mode: {compose_mode}")


# =========================================================
# 5. 用 partial 校准尺度
# =========================================================
def calibrate_scale_from_partial(
    shell_pts: np.ndarray,
    partial_pts: np.ndarray,
    search_min: float = 0.94,
    search_max: float = 1.06,
    search_steps: int = 11,
    icp_iter: int = 24,
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
    best_score = (-1.0, -1e9)

    for s in s0 * np.linspace(search_min, search_max, search_steps):
        shell_scaled = shell_local * s + shell_center

        t_est, fit15, rmse, _ = translation_only_icp(
            shell_scaled,
            partial_pts,
            max_iter=icp_iter,
            trim_ratio=0.80
        )

        score = (fit15, -rmse)
        if score > best_score:
            best_score = score
            best = {
                "scale": float(s),
                "translation": t_est.copy(),
                "fit15": float(fit15),
                "rmse": float(rmse),
            }

    return best


# =========================================================
# 6. 2D/深度/穿模评分
# =========================================================
def get_2d_symmetric_error(partial_xy, comp_xy, pixel_size=0.002):
    x_min = min(partial_xy[:, 0].min(), comp_xy[:, 0].min()) - 2 * pixel_size
    y_min = min(partial_xy[:, 1].min(), comp_xy[:, 1].min()) - 2 * pixel_size

    px = np.floor((partial_xy[:, 0] - x_min) / pixel_size).astype(np.int32)
    py = np.floor((partial_xy[:, 1] - y_min) / pixel_size).astype(np.int32)

    cx = np.floor((comp_xy[:, 0] - x_min) / pixel_size).astype(np.int32)
    cy = np.floor((comp_xy[:, 1] - y_min) / pixel_size).astype(np.int32)

    w = max(px.max(), cx.max()) + 1
    h = max(py.max(), cy.max()) + 1

    mask_p = np.zeros((w, h), dtype=bool)
    mask_c = np.zeros((w, h), dtype=bool)

    mask_p[px, py] = True
    mask_c[cx, cy] = True

    red_uncovered = np.sum(mask_p & (~mask_c))
    blue_uncovered = np.sum(mask_c & (~mask_p))
    total_error = red_uncovered + blue_uncovered
    denom = max(np.sum(mask_p), 1)

    return total_error / denom, red_uncovered, blue_uncovered


def compute_antipenetration_loss(partial_pts, shell_pts, pixel_size=0.0018, z_margin=0.0015, front_mode="min"):
    err2d, red_err, blue_err = get_2d_symmetric_error(
        partial_pts[:, :2], shell_pts[:, :2], pixel_size=pixel_size
    )

    tree_ps = cKDTree(shell_pts)
    d_ps, _ = tree_ps.query(partial_pts, workers=-1)
    mean_ps = np.mean(d_ps)

    tree_sp = cKDTree(partial_pts)
    d_sp, _ = tree_sp.query(shell_pts, workers=-1)
    mean_sp = np.mean(d_sp)

    all_pts = np.vstack([partial_pts, shell_pts])
    x_min = np.min(all_pts[:, 0]) - 2 * pixel_size
    y_min = np.min(all_pts[:, 1]) - 2 * pixel_size

    def depth_map_with_shared_origin(points):
        xy = points[:, :2]
        z = points[:, 2]
        ix = np.floor((xy[:, 0] - x_min) / pixel_size).astype(np.int32)
        iy = np.floor((xy[:, 1] - y_min) / pixel_size).astype(np.int32)
        dm = {}
        for i in range(len(points)):
            key = (int(ix[i]), int(iy[i]))
            d = z[i]
            if key not in dm:
                dm[key] = d
            else:
                if front_mode == "min":
                    if d < dm[key]:
                        dm[key] = d
                else:
                    if d > dm[key]:
                        dm[key] = d
        return dm

    dm_p = depth_map_with_shared_origin(partial_pts)
    dm_s = depth_map_with_shared_origin(shell_pts)

    common = set(dm_p.keys()) & set(dm_s.keys())
    if len(common) > 0:
        dp = np.array([dm_p[k] for k in common], dtype=np.float64)
        ds = np.array([dm_s[k] for k in common], dtype=np.float64)

        depth_mae = float(np.mean(np.abs(dp - ds)))
        if front_mode == "min":
            penetration_ratio = float(np.mean(ds < dp - z_margin))
            hollow_ratio = float(np.mean(ds > dp + z_margin))
        else:
            penetration_ratio = float(np.mean(ds > dp + z_margin))
            hollow_ratio = float(np.mean(ds < dp - z_margin))
    else:
        depth_mae = 1.0
        penetration_ratio = 1.0
        hollow_ratio = 1.0

    loss = (
        err2d * 1.0 +
        penetration_ratio * 4.0 +
        hollow_ratio * 1.5 +
        depth_mae * 30.0 +
        mean_ps * 4.0 +
        mean_sp * 1.5
    )

    return {
        "loss": float(loss),
        "err2d": float(err2d),
        "red_err": int(red_err),
        "blue_err": int(blue_err),
        "mean_ps": float(mean_ps),
        "mean_sp": float(mean_sp),
        "depth_mae": float(depth_mae),
        "penetration_ratio": float(penetration_ratio),
        "hollow_ratio": float(hollow_ratio),
    }


def rerank_candidate_with_geometry(sam_rgb_pts, part_pts, front_mode, pixel_size, z_margin):
    shell = extract_visible_shell_camera(
        sam_rgb_pts,
        grid_size=max(robust_diag(part_pts), 1e-6) / 80.0,
        shell_thickness=max(robust_diag(part_pts), 1e-6) / 140.0,
        front_mode=front_mode
    )
    if len(shell) < 100:
        return None

    metrics = compute_antipenetration_loss(
        sample_points(part_pts, min(3000, len(part_pts))),
        sample_points(shell, min(3000, len(shell))),
        pixel_size=pixel_size,
        z_margin=z_margin,
        front_mode=front_mode
    )
    return metrics, shell


def refine_pose_antipenetration(
    full_pts: np.ndarray,
    shell_pts: np.ndarray,
    partial_pts: np.ndarray,
    pixel_size=0.0018,
    z_margin=0.0015,
    front_mode="min",
    shell_sample_n=5000,
    partial_sample_n=5000,
):
    shell_sub = sample_points(shell_pts, min(shell_sample_n, len(shell_pts)))
    part_sub = sample_points(partial_pts, min(partial_sample_n, len(partial_pts)))

    center_xy = np.median(shell_sub[:, :2], axis=0)
    params = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # yaw, tx, ty, tz

    def transform_shell(p):
        return apply_se2z(shell_sub, p[0], p[1], p[2], p[3], center_xy)

    best_shell = transform_shell(params)
    best_metrics = compute_antipenetration_loss(
        part_sub, best_shell, pixel_size=pixel_size, z_margin=z_margin, front_mode=front_mode
    )
    best_loss = best_metrics["loss"]

    print(f"    [防穿模初始] 2D误差率={best_metrics['err2d']*100:.2f}%, "
          f"穿模率={best_metrics['penetration_ratio']*100:.2f}%, "
          f"深度MAE={best_metrics['depth_mae']:.4f}m", flush=True)

    schedules = [
        (0.30, 0.0015),
        (0.12, 0.0008),
        (0.05, 0.0004),
    ]

    for level, (yaw_step, trans_step) in enumerate(schedules, 1):
        improved = True
        while improved:
            improved = False
            step_list = [yaw_step, trans_step, trans_step, trans_step]
            for i, step in enumerate(step_list):
                for sign in (+1.0, -1.0):
                    cand = params.copy()
                    cand[i] += sign * step

                    cand_shell = transform_shell(cand)
                    cand_metrics = compute_antipenetration_loss(
                        part_sub, cand_shell,
                        pixel_size=pixel_size,
                        z_margin=z_margin,
                        front_mode=front_mode
                    )

                    if cand_metrics["loss"] < best_loss:
                        params = cand
                        best_loss = cand_metrics["loss"]
                        best_metrics = cand_metrics
                        improved = True

        print(f"    [防穿模 level{level}] yaw={params[0]:.3f}°, tx={params[1]:.4f}, "
              f"ty={params[2]:.4f}, tz={params[3]:.4f}, "
              f"穿模率={best_metrics['penetration_ratio']*100:.2f}%, "
              f"深度MAE={best_metrics['depth_mae']:.4f}m", flush=True)

    full_refined = apply_se2z(full_pts, params[0], params[1], params[2], params[3], center_xy)
    shell_refined = apply_se2z(shell_pts, params[0], params[1], params[2], params[3], center_xy)

    return {
        "full_pts": full_refined,
        "shell_pts": shell_refined,
        "yaw_refine_deg": float(params[0]),
        "tx_refine": float(params[1]),
        "ty_refine": float(params[2]),
        "tz_refine": float(params[3]),
        "metrics": best_metrics,
    }


# =========================================================
# 7. 主搜索：保留你原来成功版主干，但优化评分
# =========================================================
def restore_canonical_to_rgb_pose(
    sam_pts_raw: np.ndarray,
    part_pts: np.ndarray,
    pose_json_path: str,
    front_mode: str = "min",
    proj_pixel: float = 0.0018,
    z_margin: float = 0.0015,
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

    print(f"[2] quaternion 基础解释数: {len(base_rotations)}", flush=True)
    print(f"[3] adapter 数: {len(adapters)}", flush=True)

    coarse_results = []
    print("[4] 正在粗搜索 canonical -> RGB 粗姿态...", flush=True)

    total = len(base_rotations) * len(adapters) * 2
    cnt = 0
    part_sub = sample_points(part_pts, 1800)
    part_scale_ref = max(robust_diag(part_pts), 1e-6)

    for base_name, base_R in base_rotations:
        for ai, A in enumerate(adapters):
            for compose_mode in ["left", "right"]:
                cnt += 1
                R_candidate = compose_candidate_rotation(base_R, A, compose_mode)

                sam_rot = sam_centered @ R_candidate.T
                shell_rot = extract_visible_shell_camera(
                    sam_rot,
                    grid_size=part_scale_ref / 70.0,
                    shell_thickness=part_scale_ref / 120.0,
                    front_mode=front_mode
                )

                if len(shell_rot) < 100:
                    continue

                scale_res = calibrate_scale_from_partial(
                    sample_points(shell_rot, 1800),
                    part_sub,
                    search_min=0.92,
                    search_max=1.08,
                    search_steps=15,
                    icp_iter=22
                )

                if scale_res is None:
                    continue

                sam_rgb = (sam_centered * scale_res["scale"]) @ R_candidate.T
                sam_rgb = sam_rgb + scale_res["translation"].reshape(1, 3)

                geom = rerank_candidate_with_geometry(
                    sam_rgb, part_pts, front_mode, proj_pixel, z_margin
                )
                if geom is None:
                    continue
                metrics, _ = geom

                coarse_results.append({
                    "base_name": base_name,
                    "adapter_idx": ai,
                    "compose_mode": compose_mode,
                    "R_candidate": R_candidate.copy(),
                    "scale": scale_res["scale"],
                    "translation": scale_res["translation"].copy(),
                    "fit15": scale_res["fit15"],
                    "rmse": scale_res["rmse"],
                    "geom_loss": metrics["loss"],
                    "penetration_ratio": metrics["penetration_ratio"],
                    "depth_mae": metrics["depth_mae"],
                })

                if cnt % 40 == 0 or cnt == total:
                    print(f"    [粗搜索] {cnt}/{total} 已完成", flush=True)

    if len(coarse_results) == 0:
        raise RuntimeError("粗搜索失败：没有找到有效 RGB 粗姿态。")

    coarse_results = sorted(
        coarse_results,
        key=lambda x: (x["fit15"], -x["geom_loss"], -x["rmse"]),
        reverse=True
    )[:4]

    print("[5] 粗搜索前 4 名：", flush=True)
    for i, c in enumerate(coarse_results, 1):
        print(
            f"    #{i}: {c['base_name']} + A[{c['adapter_idx']}] + {c['compose_mode']}, "
            f"scale={c['scale']:.6f}, fit15={c['fit15']*100:.2f}%, "
            f"rmse={c['rmse']:.4f}m, geom={c['geom_loss']:.4f}",
            flush=True
        )

    best = None
    best_score = (1e9, -1.0, 1e9)

    print("[6] 对前几名做轻量 scale + yaw 精修...", flush=True)

    for ci, cand in enumerate(coarse_results, 1):
        print(f"    [细搜候选 {ci}/{len(coarse_results)}] ...", flush=True)

        R_candidate = cand["R_candidate"]
        local_best = None
        local_best_score = (1e9, -1.0, 1e9)

        scale_cands = cand["scale"] * np.linspace(0.99, 1.01, 5)
        yaw_cands = np.arange(-4.0, 4.01, 1.0)

        for s in scale_cands:
            sam_rgb = (sam_centered * s) @ R_candidate.T
            rot_center = np.mean(sam_rgb, axis=0)

            for yaw_deg in yaw_cands:
                sam_rgb_yaw = rotate_about_center_z(sam_rgb, yaw_deg, rot_center)

                shell = extract_visible_shell_camera(
                    sam_rgb_yaw,
                    grid_size=part_scale_ref / 75.0,
                    shell_thickness=part_scale_ref / 130.0,
                    front_mode=front_mode
                )
                if len(shell) < 100:
                    continue

                t_est, fit15, rmse, _ = translation_only_icp(
                    sample_points(shell, 2200),
                    sample_points(part_pts, 2200),
                    max_iter=24,
                    trim_ratio=0.80
                )

                sam_try = sam_rgb_yaw + t_est.reshape(1, 3)
                geom = rerank_candidate_with_geometry(
                    sam_try, part_pts, front_mode, proj_pixel, z_margin
                )
                if geom is None:
                    continue
                metrics, _ = geom

                score = (metrics["loss"], -fit15, rmse)
                if score < local_best_score:
                    local_best_score = score
                    local_best = {
                        "base_name": cand["base_name"],
                        "adapter_idx": cand["adapter_idx"],
                        "compose_mode": cand["compose_mode"],
                        "R_candidate": R_candidate.copy(),
                        "scale": float(s),
                        "yaw_deg": float(yaw_deg),
                        "translation": t_est.copy(),
                        "fit15": float(fit15),
                        "rmse": float(rmse),
                        "geom_metrics": metrics,
                    }

        if local_best is not None:
            score = (local_best["geom_metrics"]["loss"], -local_best["fit15"], local_best["rmse"])
            if score < best_score:
                best_score = score
                best = copy.deepcopy(local_best)

    if best is None:
        raise RuntimeError("细搜索失败：没有得到最终 RGB 位姿。")

    print("[7] 最终最佳解释：", flush=True)
    print(
        f"    {best['base_name']} + A[{best['adapter_idx']}] + {best['compose_mode']}, "
        f"scale={best['scale']:.6f}, yaw={best['yaw_deg']:.2f}°, "
        f"fit15={best['fit15']*100:.2f}%, rmse={best['rmse']:.4f}m, "
        f"geom={best['geom_metrics']['loss']:.4f}",
        flush=True
    )

    sam_rgb = (sam_centered * best["scale"]) @ best["R_candidate"].T
    rot_center = np.mean(sam_rgb, axis=0)
    sam_rgb = rotate_about_center_z(sam_rgb, best["yaw_deg"], rot_center)
    sam_rgb = sam_rgb + best["translation"].reshape(1, 3)

    shell_rgb = extract_visible_shell_camera(
        sam_rgb,
        grid_size=part_scale_ref / 80.0,
        shell_thickness=part_scale_ref / 140.0,
        front_mode=front_mode,
    )

    return {
        "full_rgb_pose": sam_rgb,
        "visible_shell": shell_rgb,
        "scale": best["scale"],
        "yaw_deg": best["yaw_deg"],
        "translation": best["translation"],
        "base_name": best["base_name"],
        "adapter_idx": best["adapter_idx"],
        "compose_mode": best["compose_mode"],
        "fit15": best["fit15"],
        "rmse": best["rmse"],
        "geom_metrics": best["geom_metrics"],
    }


# =========================================================
# 8. CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d", required=True, help="canonical 完整点云 / 网格")
    parser.add_argument("--partial", required=True, help="真实 partial 点云（RGB 视角下）")
    parser.add_argument("--pose-json", required=True, help="sam3d_pose.json")
    parser.add_argument("--model-dir", default="", help="兼容旧命令保留，不再用于尺度校准")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sam-samples", type=int, default=50000)
    parser.add_argument("--front-mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument("--proj-pixel", type=float, default=0.0018)
    parser.add_argument("--z-margin", type=float, default=0.0015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    print("\n" + "=" * 60)
    print("[🚀] 启动：优化版 RGB 视角位姿恢复（保留成功版搜索 + 几何重排 + 防穿模精修）")
    print("=" * 60, flush=True)

    sam_pts_raw = load_points_any(args.sam3d, mesh_samples=args.sam_samples)
    part_pts = load_points_any(args.partial, mesh_samples=args.sam_samples)

    result = restore_canonical_to_rgb_pose(
        sam_pts_raw=sam_pts_raw,
        part_pts=part_pts,
        pose_json_path=args.pose_json,
        front_mode=args.front_mode,
        proj_pixel=args.proj_pixel,
        z_margin=args.z_margin,
    )

    full_rgb = result["full_rgb_pose"]
    shell_rgb = result["visible_shell"]

    print("[8] 末端防穿模精修中（锁尺度，仅调 yaw + tx + ty + tz）...", flush=True)
    refine_res = refine_pose_antipenetration(
        full_pts=full_rgb,
        shell_pts=shell_rgb,
        partial_pts=part_pts,
        pixel_size=args.proj_pixel,
        z_margin=args.z_margin,
        front_mode=args.front_mode,
        shell_sample_n=5000,
        partial_sample_n=5000,
    )

    full_refined = refine_res["full_pts"]
    shell_refined = refine_res["shell_pts"]

    tree = cKDTree(full_refined)
    final_dist, _ = tree.query(part_pts, workers=-1)
    final_fit = float(np.mean(final_dist < 0.015))
    final_rmse = float(np.sqrt(np.mean(final_dist ** 2)))

    full_path = os.path.join(args.out_dir, "full_rgb_pose_refined.ply")
    shell_path = os.path.join(args.out_dir, "visible_rgb_shell_refined.ply")
    merged_path = os.path.join(args.out_dir, "merged_rgb_pose_refined.ply")
    result_json = os.path.join(args.out_dir, "pose_decode_result.json")

    save_point_cloud(full_path, full_refined, color=[0.10, 0.45, 0.85])
    save_point_cloud(shell_path, shell_refined, color=[0.10, 0.85, 0.20])
    save_merged_point_cloud(merged_path, full_refined, part_pts)

    info = {
        "base_name": result["base_name"],
        "adapter_idx": int(result["adapter_idx"]),
        "compose_mode": result["compose_mode"],
        "scale": float(result["scale"]),
        "yaw_deg_restore": float(result["yaw_deg"]),
        "translation_restore": result["translation"].tolist(),
        "fit15_internal": float(result["fit15"]),
        "rmse_internal": float(result["rmse"]),
        "geom_metrics_restore": result["geom_metrics"],

        "yaw_refine_deg": float(refine_res["yaw_refine_deg"]),
        "tx_refine": float(refine_res["tx_refine"]),
        "ty_refine": float(refine_res["ty_refine"]),
        "tz_refine": float(refine_res["tz_refine"]),
        "refine_metrics": refine_res["metrics"],

        "final_fit_partial_to_full_rgb": float(final_fit),
        "final_rmse_partial_to_full_rgb": float(final_rmse),

        "front_mode": args.front_mode,
        "sam3d": args.sam3d,
        "partial": args.partial,
        "pose_json": args.pose_json,
        "note": "optimized from previous successful version; keep adapter search; add geometry rerank and anti-penetration refinement",
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ 优化版 RGB 视角位姿恢复完成")
    print(f"   最佳解释: {result['base_name']} + A[{result['adapter_idx']}] + {result['compose_mode']}")
    print(f"   scale:    {result['scale']:.6f}")
    print(f"   恢复yaw:  {result['yaw_deg']:.2f}°")
    print(
        f"   精修残差: yaw={refine_res['yaw_refine_deg']:.3f}°, "
        f"tx={refine_res['tx_refine']:.4f}, ty={refine_res['ty_refine']:.4f}, tz={refine_res['tz_refine']:.4f}"
    )
    print(f"   最终覆盖率 (partial -> full @1.5cm): {final_fit*100:.2f}%")
    print(f"   最终 RMSE: {final_rmse:.4f}m")
    print(f"   full_rgb_pose_refined:     {full_path}")
    print(f"   visible_rgb_shell_refined: {shell_path}")
    print(f"   merged_rgb_pose_refined:   {merged_path}")
    print(f"   result_json:               {result_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()