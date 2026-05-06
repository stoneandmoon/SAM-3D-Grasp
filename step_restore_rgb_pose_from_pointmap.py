#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import copy
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot


# =========================================================
# 0. 基础工具
# =========================================================
def sample_points(pts: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= n:
        return pts.copy()
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


def robust_diag(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
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


# =========================================================
# 1. quaternion 候选解释
# =========================================================
def decode_quaternion_candidates(raw_quat):
    raw_quat = np.asarray(raw_quat, dtype=np.float64).reshape(-1)
    if len(raw_quat) != 4:
        raise RuntimeError(f"rotation_quat 长度不是 4: {raw_quat}")

    cands = []

    # case A: raw = [w, x, y, z]
    w, x, y, z = raw_quat
    R1 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("wxyz_R", R1))
    cands.append(("wxyz_RT", R1.T))

    # case B: raw = [x, y, z, w]
    x, y, z, w = raw_quat
    R2 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("xyzw_R", R2))
    cands.append(("xyzw_RT", R2.T))

    return cands


# =========================================================
# 2. 相机视角下提可见点
# =========================================================
def extract_visible_shell_camera(
    pts_cam: np.ndarray,
    grid_size: float = 0.003,
    shell_thickness: float = 0.004,
    front_mode: str = "min",
):
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    if len(pts_cam) == 0:
        return pts_cam.copy()

    xy = pts_cam[:, :2]
    z = pts_cam[:, 2]

    xy_min = np.min(xy, axis=0)
    ij = np.floor((xy - xy_min) / max(grid_size, 1e-9)).astype(np.int64)

    front_z = {}
    for i in range(len(pts_cam)):
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

    mask = np.zeros(len(pts_cam), dtype=bool)
    for i in range(len(pts_cam)):
        key = (int(ij[i, 0]), int(ij[i, 1]))
        d0 = front_z[key]
        if front_mode == "min":
            if z[i] <= d0 + shell_thickness:
                mask[i] = True
        else:
            if z[i] >= d0 - shell_thickness:
                mask[i] = True

    return pts_cam[mask]


# =========================================================
# 3. 只优化平移
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
# 4. uniform scale
# =========================================================
def calibrate_scale_from_partial(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    search_min: float = 0.94,
    search_max: float = 1.06,
    search_steps: int = 11,
    icp_iter: int = 24,
):
    source_pts = np.asarray(source_pts, dtype=np.float64)
    target_pts = np.asarray(target_pts, dtype=np.float64)

    if len(source_pts) < 30 or len(target_pts) < 30:
        return None

    src_center = np.mean(source_pts, axis=0)
    src_local = source_pts - src_center

    src_diag = robust_projected_diag_xy(src_local)
    tgt_diag = robust_projected_diag_xy(target_pts)

    if src_diag < 1e-9 or tgt_diag < 1e-9:
        return None

    s0 = tgt_diag / src_diag

    best = None
    best_score = (-1.0, -1e9)

    for s in s0 * np.linspace(search_min, search_max, search_steps):
        src_scaled = src_local * s + src_center

        t_est, fit15, rmse, _ = translation_only_icp(
            src_scaled,
            target_pts,
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
# 5. 小 3D 旋转 / 相似变换
# =========================================================
def rotx(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c]
    ], dtype=np.float64)


def roty(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float64)


def rotz(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rotate_about_center_xyz(points: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float, center: np.ndarray):
    R = rotz(rz_deg) @ roty(ry_deg) @ rotx(rx_deg)
    return ((points - center.reshape(1, 3)) @ R.T) + center.reshape(1, 3)


def apply_similarity_small_3d(points: np.ndarray,
                              scale: float,
                              rx_deg: float, ry_deg: float, rz_deg: float,
                              tx: float, ty: float, tz: float,
                              center_xyz: np.ndarray):
    pts = np.asarray(points, dtype=np.float64).copy()
    pts = (pts - center_xyz.reshape(1, 3)) * scale + center_xyz.reshape(1, 3)

    R = rotz(rz_deg) @ roty(ry_deg) @ rotx(rx_deg)
    pts = ((pts - center_xyz.reshape(1, 3)) @ R.T) + center_xyz.reshape(1, 3)

    pts[:, 0] += tx
    pts[:, 1] += ty
    pts[:, 2] += tz
    return pts


# =========================================================
# 6. 几何评分
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


def compute_geometry_loss(partial_pts, source_pts, pixel_size=0.0018, z_margin=0.0015, front_mode="min"):
    err2d, red_err, blue_err = get_2d_symmetric_error(
        partial_pts[:, :2], source_pts[:, :2], pixel_size=pixel_size
    )

    tree_ps = cKDTree(source_pts)
    d_ps, _ = tree_ps.query(partial_pts, workers=-1)
    mean_ps = np.mean(d_ps)

    tree_sp = cKDTree(partial_pts)
    d_sp, _ = tree_sp.query(source_pts, workers=-1)
    mean_sp = np.mean(d_sp)

    all_pts = np.vstack([partial_pts, source_pts])
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
    dm_s = depth_map_with_shared_origin(source_pts)

    common = set(dm_p.keys()) & set(dm_s.keys())
    if len(common) > 0:
        dp = np.array([dm_p[k] for k in common], dtype=np.float64)
        ds = np.array([dm_s[k] for k in common], dtype=np.float64)

        depth_mae = float(np.mean(np.abs(dp - ds)))
        if front_mode == "min":
            penetration_ratio = float(np.mean(ds < dp - z_margin))
        else:
            penetration_ratio = float(np.mean(ds > dp + z_margin))
    else:
        depth_mae = 1.0
        penetration_ratio = 1.0

    loss = err2d + mean_ps * 4.0 + mean_sp * 1.5 + penetration_ratio * 3.5 + depth_mae * 20.0

    return {
        "loss": float(loss),
        "err2d": float(err2d),
        "mean_ps": float(mean_ps),
        "mean_sp": float(mean_sp),
        "penetration_ratio": float(penetration_ratio),
        "depth_mae": float(depth_mae),
        "red_err": int(red_err),
        "blue_err": int(blue_err),
    }


# =========================================================
# 6.5 末端精修：scale + rx + ry + rz + tx + ty + tz
# =========================================================
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

    center_xyz = np.mean(shell_sub, axis=0)

    # scale, rx, ry, rz, tx, ty, tz
    params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def transform_shell(p):
        return apply_similarity_small_3d(
            shell_sub,
            scale=p[0],
            rx_deg=p[1],
            ry_deg=p[2],
            rz_deg=p[3],
            tx=p[4],
            ty=p[5],
            tz=p[6],
            center_xyz=center_xyz
        )

    best_shell = transform_shell(params)
    best_metrics = compute_geometry_loss(
        part_sub, best_shell,
        pixel_size=pixel_size,
        z_margin=z_margin,
        front_mode=front_mode
    )
    best_loss = best_metrics["loss"]

    print(
        f"    [末端精修初始] 2D误差率={best_metrics['err2d']*100:.2f}%, "
        f"穿模率={best_metrics['penetration_ratio']*100:.2f}%, "
        f"深度MAE={best_metrics['depth_mae']:.4f}m",
        flush=True
    )

    schedules = [
        (0.0020, 0.60, 0.0015),
        (0.0010, 0.25, 0.0008),
        (0.0005, 0.10, 0.0004),
    ]

    for level, (scale_step, rot_step, trans_step) in enumerate(schedules, 1):
        improved = True
        while improved:
            improved = False

            proposals = [
                (0, scale_step),
                (1, rot_step),
                (2, rot_step),
                (3, rot_step),
                (4, trans_step),
                (5, trans_step),
                (6, trans_step),
            ]

            for idx, step in proposals:
                for sign in (+1.0, -1.0):
                    cand = params.copy()
                    cand[idx] += sign * step

                    if idx == 0 and (cand[0] < 0.97 or cand[0] > 1.03):
                        continue

                    cand_shell = transform_shell(cand)
                    cand_metrics = compute_geometry_loss(
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

        print(
            f"    [末端精修 level{level}] scale={params[0]:.5f}, "
            f"rx={params[1]:.3f}°, ry={params[2]:.3f}°, rz={params[3]:.3f}°, "
            f"tx={params[4]:.4f}, ty={params[5]:.4f}, tz={params[6]:.4f}, "
            f"穿模率={best_metrics['penetration_ratio']*100:.2f}%, "
            f"深度MAE={best_metrics['depth_mae']:.4f}m",
            flush=True
        )

    full_refined = apply_similarity_small_3d(
        full_pts,
        scale=params[0],
        rx_deg=params[1],
        ry_deg=params[2],
        rz_deg=params[3],
        tx=params[4],
        ty=params[5],
        tz=params[6],
        center_xyz=center_xyz
    )

    shell_refined = apply_similarity_small_3d(
        shell_pts,
        scale=params[0],
        rx_deg=params[1],
        ry_deg=params[2],
        rz_deg=params[3],
        tx=params[4],
        ty=params[5],
        tz=params[6],
        center_xyz=center_xyz
    )

    return {
        "full_pts": full_refined,
        "shell_pts": shell_refined,
        "scale_refine": float(params[0]),
        "rx_refine_deg": float(params[1]),
        "ry_refine_deg": float(params[2]),
        "rz_refine_deg": float(params[3]),
        "tx_refine": float(params[4]),
        "ty_refine": float(params[5]),
        "tz_refine": float(params[6]),
        "metrics": best_metrics,
    }


# =========================================================
# 7. Step A: canonical full -> sam_partial_rgb
# =========================================================
def align_full_to_sam_partial(
    sam_full_raw: np.ndarray,
    sam_partial_rgb: np.ndarray,
    pose_json_path: str,
    front_mode_arg: str = "auto",
    proj_pixel: float = 0.0018,
    z_margin: float = 0.0015,
):
    print("[1] 正在读取 pose json ...", flush=True)
    with open(pose_json_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if "rotation_quat" not in pose_data:
        raise RuntimeError(f"{pose_json_path} 里没有 rotation_quat")

    raw_quat = pose_data["rotation_quat"]
    quat_cands = decode_quaternion_candidates(raw_quat)

    full_center = np.mean(sam_full_raw, axis=0)
    full_centered = sam_full_raw - full_center

    front_modes = ["min", "max"] if front_mode_arg == "auto" else [front_mode_arg]
    sam_partial_scale_ref = max(robust_diag(sam_partial_rgb), 1e-6)

    print(f"[2] quaternion 候选数: {len(quat_cands)}", flush=True)
    print(f"[3] front_mode 候选: {front_modes}", flush=True)

    coarse_results = []
    print("[4] StepA：full_canonical -> sam_partial_rgb 粗搜索...", flush=True)

    total = len(quat_cands) * len(front_modes)
    done = 0
    sam_partial_sub = sample_points(sam_partial_rgb, 1800)

    for qname, Rcand in quat_cands:
        for front_mode in front_modes:
            done += 1

            full_cam = full_centered @ Rcand.T

            shell = extract_visible_shell_camera(
                full_cam,
                grid_size=sam_partial_scale_ref / 75.0,
                shell_thickness=sam_partial_scale_ref / 130.0,
                front_mode=front_mode
            )
            if len(shell) < 100:
                print(f"    [粗搜索] {done}/{total} 已完成 ({qname}, mode={front_mode}) shell过少", flush=True)
                continue

            scale_res = calibrate_scale_from_partial(
                sample_points(shell, 1800),
                sam_partial_sub,
                search_min=0.92,
                search_max=1.08,
                search_steps=15,
                icp_iter=22
            )
            if scale_res is None:
                print(f"    [粗搜索] {done}/{total} 已完成 ({qname}, mode={front_mode}) scale失败", flush=True)
                continue

            local_best = None
            local_best_score = (1e9, -1.0, 1e9)

            for yaw_deg in np.arange(-6.0, 6.01, 2.0):
                full_try = (full_centered * scale_res["scale"]) @ Rcand.T
                full_try = rotate_about_center_xyz(
                    full_try,
                    rx_deg=0.0,
                    ry_deg=0.0,
                    rz_deg=yaw_deg,
                    center=np.zeros(3, dtype=np.float64)
                )
                full_try = full_try + scale_res["translation"].reshape(1, 3)

                shell_try = extract_visible_shell_camera(
                    full_try,
                    grid_size=sam_partial_scale_ref / 75.0,
                    shell_thickness=sam_partial_scale_ref / 130.0,
                    front_mode=front_mode
                )
                if len(shell_try) < 100:
                    continue

                geom = compute_geometry_loss(
                    sample_points(sam_partial_rgb, min(3000, len(sam_partial_rgb))),
                    sample_points(shell_try, min(3000, len(shell_try))),
                    pixel_size=proj_pixel,
                    z_margin=z_margin,
                    front_mode=front_mode
                )

                t_est, fit15, rmse, _ = translation_only_icp(
                    sample_points(shell_try, 2200),
                    sample_points(sam_partial_rgb, 2200),
                    max_iter=20,
                    trim_ratio=0.80
                )

                score = (geom["loss"], -fit15, rmse)
                if score < local_best_score:
                    local_best_score = score
                    local_best = {
                        "quat_name": qname,
                        "front_mode": front_mode,
                        "R_candidate": Rcand.copy(),
                        "scale": float(scale_res["scale"]),
                        "yaw_deg": float(yaw_deg),
                        "translation": (scale_res["translation"] + t_est).copy(),
                        "fit15": float(fit15),
                        "rmse": float(rmse),
                        "geom_metrics": geom,
                    }

            if local_best is not None:
                coarse_results.append(local_best)

            print(f"    [粗搜索] {done}/{total} 已完成 ({qname}, mode={front_mode})", flush=True)

    if len(coarse_results) == 0:
        raise RuntimeError("StepA 粗搜索失败：没有有效候选。")

    coarse_results = sorted(
        coarse_results,
        key=lambda x: (x["geom_metrics"]["loss"], -x["fit15"], x["rmse"])
    )[:4]

    print("[5] StepA 粗搜索前 4 名：", flush=True)
    for i, c in enumerate(coarse_results, 1):
        print(
            f"    #{i}: {c['quat_name']}, mode={c['front_mode']}, "
            f"scale={c['scale']:.6f}, yaw={c['yaw_deg']:.2f}°, "
            f"fit15={c['fit15']*100:.2f}%, rmse={c['rmse']:.4f}m, "
            f"geom={c['geom_metrics']['loss']:.4f}",
            flush=True
        )

    best = None
    best_score = (1e9, -1.0, 1e9)

    print("[6] StepA 细搜索...", flush=True)

    for ci, cand in enumerate(coarse_results, 1):
        print(f"    [细搜候选 {ci}/{len(coarse_results)}] ...", flush=True)

        Rcand = cand["R_candidate"]
        front_mode = cand["front_mode"]

        local_best = None
        local_best_score = (1e9, -1.0, 1e9)

        scale_cands = cand["scale"] * np.linspace(0.99, 1.01, 5)
        yaw_cands = np.arange(cand["yaw_deg"] - 2.0, cand["yaw_deg"] + 2.01, 0.5)

        for s in scale_cands:
            full_try = (full_centered * s) @ Rcand.T

            for yaw_deg in yaw_cands:
                full_try_yaw = rotate_about_center_xyz(
                    full_try,
                    rx_deg=0.0,
                    ry_deg=0.0,
                    rz_deg=yaw_deg,
                    center=np.zeros(3, dtype=np.float64)
                )

                shell_try = extract_visible_shell_camera(
                    full_try_yaw,
                    grid_size=sam_partial_scale_ref / 80.0,
                    shell_thickness=sam_partial_scale_ref / 140.0,
                    front_mode=front_mode
                )
                if len(shell_try) < 100:
                    continue

                t_est, fit15, rmse, _ = translation_only_icp(
                    sample_points(shell_try, 2200),
                    sample_points(sam_partial_rgb, 2200),
                    max_iter=24,
                    trim_ratio=0.80
                )

                full_eval = full_try_yaw + t_est.reshape(1, 3)
                shell_eval = extract_visible_shell_camera(
                    full_eval,
                    grid_size=sam_partial_scale_ref / 80.0,
                    shell_thickness=sam_partial_scale_ref / 140.0,
                    front_mode=front_mode
                )

                geom = compute_geometry_loss(
                    sample_points(sam_partial_rgb, min(3000, len(sam_partial_rgb))),
                    sample_points(shell_eval, min(3000, len(shell_eval))),
                    pixel_size=proj_pixel,
                    z_margin=z_margin,
                    front_mode=front_mode
                )

                score = (geom["loss"], -fit15, rmse)
                if score < local_best_score:
                    local_best_score = score
                    local_best = {
                        "quat_name": cand["quat_name"],
                        "front_mode": front_mode,
                        "R_candidate": Rcand.copy(),
                        "scale": float(s),
                        "yaw_deg": float(yaw_deg),
                        "translation": t_est.copy(),
                        "fit15": float(fit15),
                        "rmse": float(rmse),
                        "geom_metrics": geom,
                    }

        if local_best is not None:
            score = (local_best["geom_metrics"]["loss"], -local_best["fit15"], local_best["rmse"])
            if score < best_score:
                best_score = score
                best = copy.deepcopy(local_best)

    if best is None:
        raise RuntimeError("StepA 细搜索失败：没有得到最终解。")

    print("[7] StepA 最终最佳解释：", flush=True)
    print(
        f"    {best['quat_name']}, mode={best['front_mode']}, "
        f"scale={best['scale']:.6f}, yaw={best['yaw_deg']:.2f}°, "
        f"fit15={best['fit15']*100:.2f}%, rmse={best['rmse']:.4f}m, "
        f"geom={best['geom_metrics']['loss']:.4f}",
        flush=True
    )

    full_rgb = (full_centered * best["scale"]) @ best["R_candidate"].T
    full_rgb = rotate_about_center_xyz(
        full_rgb,
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=best["yaw_deg"],
        center=np.zeros(3, dtype=np.float64)
    )
    full_rgb = full_rgb + best["translation"].reshape(1, 3)

    shell_rgb = extract_visible_shell_camera(
        full_rgb,
        grid_size=sam_partial_scale_ref / 85.0,
        shell_thickness=sam_partial_scale_ref / 145.0,
        front_mode=best["front_mode"]
    )

    return {
        "full_rgb_pose": full_rgb,
        "visible_shell_rgb": shell_rgb,
        "sam_partial_rgb": sam_partial_rgb,
        "scale": best["scale"],
        "yaw_deg": best["yaw_deg"],
        "translation": best["translation"],
        "quat_name": best["quat_name"],
        "front_mode": best["front_mode"],
        "fit15": best["fit15"],
        "rmse": best["rmse"],
        "geom_metrics": best["geom_metrics"],
    }


# =========================================================
# 8. Step B: sam_partial_rgb -> real_partial_rgb residual
#    这里改成小范围 3 轴残差旋转
# =========================================================
def align_sam_partial_to_real_partial(
    sam_partial_rgb: np.ndarray,
    real_partial_rgb: np.ndarray,
    front_mode: str = "min",
    proj_pixel: float = 0.0018,
    z_margin: float = 0.0015,
):
    print("[8] StepB：sam_partial_rgb -> real_partial_rgb residual 配准...", flush=True)

    center = np.mean(sam_partial_rgb, axis=0)

    scale_res = calibrate_scale_from_partial(
        sample_points(sam_partial_rgb, min(2200, len(sam_partial_rgb))),
        sample_points(real_partial_rgb, min(2200, len(real_partial_rgb))),
        search_min=0.97,
        search_max=1.03,
        search_steps=9,
        icp_iter=20
    )
    if scale_res is None:
        raise RuntimeError("StepB 初始尺度估计失败。")

    best = None
    best_score = (1e9, -1.0, 1e9)

    scale_cands = scale_res["scale"] * np.linspace(0.995, 1.005, 5)
    rx_cands = np.arange(-4.0, 4.01, 1.0)
    ry_cands = np.arange(-4.0, 4.01, 1.0)
    rz_cands = np.arange(-6.0, 6.01, 1.0)

    for s in scale_cands:
        for rx_deg in rx_cands:
            for ry_deg in ry_cands:
                for rz_deg in rz_cands:
                    src = apply_similarity_small_3d(
                        sam_partial_rgb,
                        scale=s,
                        rx_deg=rx_deg,
                        ry_deg=ry_deg,
                        rz_deg=rz_deg,
                        tx=0.0, ty=0.0, tz=0.0,
                        center_xyz=center
                    )

                    t_est, fit15, rmse, _ = translation_only_icp(
                        sample_points(src, min(2200, len(src))),
                        sample_points(real_partial_rgb, min(2200, len(real_partial_rgb))),
                        max_iter=24,
                        trim_ratio=0.80
                    )

                    src_eval = src + t_est.reshape(1, 3)
                    geom = compute_geometry_loss(
                        sample_points(real_partial_rgb, min(3000, len(real_partial_rgb))),
                        sample_points(src_eval, min(3000, len(src_eval))),
                        pixel_size=proj_pixel,
                        z_margin=z_margin,
                        front_mode=front_mode
                    )

                    score = (geom["loss"], -fit15, rmse)
                    if score < best_score:
                        best_score = score
                        best = {
                            "scale": float(s),
                            "rx_deg": float(rx_deg),
                            "ry_deg": float(ry_deg),
                            "rz_deg": float(rz_deg),
                            "translation": t_est.copy(),
                            "fit15": float(fit15),
                            "rmse": float(rmse),
                            "geom_metrics": geom,
                        }

    if best is None:
        raise RuntimeError("StepB residual 搜索失败。")

    print(
        f"    StepB 最优: scale={best['scale']:.6f}, "
        f"rx={best['rx_deg']:.2f}°, ry={best['ry_deg']:.2f}°, rz={best['rz_deg']:.2f}°, "
        f"fit15={best['fit15']*100:.2f}%, rmse={best['rmse']:.4f}m, "
        f"geom={best['geom_metrics']['loss']:.4f}",
        flush=True
    )

    return best


# =========================================================
# 9. 主流程
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d", required=True, help="SAM-3D 完整 canonical 点云/网格")
    parser.add_argument("--sam-partial", required=True, help="由 pointmap 导出的 sam_partial_rgb.ply")
    parser.add_argument("--real-partial", required=True, help="真实深度提取的 partial_rgb.ply")
    parser.add_argument("--pose-json", required=True, help="sam3d_pose.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sam-samples", type=int, default=50000)
    parser.add_argument("--front-mode", type=str, default="auto", choices=["auto", "min", "max"])
    parser.add_argument("--proj-pixel", type=float, default=0.0018)
    parser.add_argument("--z-margin", type=float, default=0.0015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    print("\n" + "=" * 60)
    print("[🚀] 启动：pointmap 版 RGB 视角恢复 + partial-to-partial 配准")
    print("=" * 60, flush=True)

    sam_full = load_points_any(args.sam3d, mesh_samples=args.sam_samples)
    sam_partial_rgb = load_points_any(args.sam_partial, mesh_samples=args.sam_samples)
    real_partial_rgb = load_points_any(args.real_partial, mesh_samples=args.sam_samples)

    # Step A
    stepA = align_full_to_sam_partial(
        sam_full_raw=sam_full,
        sam_partial_rgb=sam_partial_rgb,
        pose_json_path=args.pose_json,
        front_mode_arg=args.front_mode,
        proj_pixel=args.proj_pixel,
        z_margin=args.z_margin,
    )

    full_rgb = stepA["full_rgb_pose"]
    shell_rgb = stepA["visible_shell_rgb"]

    # Step B
    stepB = align_sam_partial_to_real_partial(
        sam_partial_rgb=stepA["sam_partial_rgb"],
        real_partial_rgb=real_partial_rgb,
        front_mode=stepA["front_mode"],
        proj_pixel=args.proj_pixel,
        z_margin=args.z_margin,
    )

    center_res = np.mean(stepA["sam_partial_rgb"], axis=0)

    full_rgb_refined = apply_similarity_small_3d(
        full_rgb,
        scale=stepB["scale"],
        rx_deg=stepB["rx_deg"],
        ry_deg=stepB["ry_deg"],
        rz_deg=stepB["rz_deg"],
        tx=stepB["translation"][0],
        ty=stepB["translation"][1],
        tz=stepB["translation"][2],
        center_xyz=center_res
    )

    shell_rgb_refined = apply_similarity_small_3d(
        shell_rgb,
        scale=stepB["scale"],
        rx_deg=stepB["rx_deg"],
        ry_deg=stepB["ry_deg"],
        rz_deg=stepB["rz_deg"],
        tx=stepB["translation"][0],
        ty=stepB["translation"][1],
        tz=stepB["translation"][2],
        center_xyz=center_res
    )

    sam_partial_refined = apply_similarity_small_3d(
        stepA["sam_partial_rgb"],
        scale=stepB["scale"],
        rx_deg=stepB["rx_deg"],
        ry_deg=stepB["ry_deg"],
        rz_deg=stepB["rz_deg"],
        tx=stepB["translation"][0],
        ty=stepB["translation"][1],
        tz=stepB["translation"][2],
        center_xyz=center_res
    )

    # 末端小残差精修
    print("[9] 末端 residual 精修中...", flush=True)
    refine_res = refine_pose_antipenetration(
        full_pts=full_rgb_refined,
        shell_pts=shell_rgb_refined,
        partial_pts=real_partial_rgb,
        pixel_size=args.proj_pixel,
        z_margin=args.z_margin,
        front_mode=stepA["front_mode"],
        shell_sample_n=5000,
        partial_sample_n=5000,
    )

    full_final = refine_res["full_pts"]
    shell_final = refine_res["shell_pts"]

    tree = cKDTree(full_final)
    final_dist, _ = tree.query(real_partial_rgb, workers=-1)
    final_fit = float(np.mean(final_dist < 0.015))
    final_rmse = float(np.sqrt(np.mean(final_dist ** 2)))

    full_path = os.path.join(args.out_dir, "full_rgb_pose_from_pointmap_refined.ply")
    shell_path = os.path.join(args.out_dir, "visible_rgb_shell_from_full_refined.ply")
    sam_partial_path = os.path.join(args.out_dir, "sam_partial_rgb_to_real_refined.ply")
    merged_path = os.path.join(args.out_dir, "merged_rgb_pose_from_pointmap_refined.ply")
    result_json = os.path.join(args.out_dir, "pointmap_pose_result.json")

    save_point_cloud(full_path, full_final, color=[0.10, 0.45, 0.85])
    save_point_cloud(shell_path, shell_final, color=[0.10, 0.85, 0.20])
    save_point_cloud(sam_partial_path, sam_partial_refined, color=[0.85, 0.85, 0.10])
    save_merged_point_cloud(merged_path, full_final, real_partial_rgb)

    info = {
        "stepA": {
            "quat_name": stepA["quat_name"],
            "front_mode": stepA["front_mode"],
            "scale": float(stepA["scale"]),
            "yaw_deg": float(stepA["yaw_deg"]),
            "translation": stepA["translation"].tolist(),
            "fit15": float(stepA["fit15"]),
            "rmse": float(stepA["rmse"]),
            "geom_metrics": stepA["geom_metrics"],
        },
        "stepB": {
            "scale": float(stepB["scale"]),
            "rx_deg": float(stepB["rx_deg"]),
            "ry_deg": float(stepB["ry_deg"]),
            "rz_deg": float(stepB["rz_deg"]),
            "translation": stepB["translation"].tolist(),
            "fit15": float(stepB["fit15"]),
            "rmse": float(stepB["rmse"]),
            "geom_metrics": stepB["geom_metrics"],
        },
        "refine": {
            "scale_refine": float(refine_res["scale_refine"]),
            "rx_refine_deg": float(refine_res["rx_refine_deg"]),
            "ry_refine_deg": float(refine_res["ry_refine_deg"]),
            "rz_refine_deg": float(refine_res["rz_refine_deg"]),
            "tx_refine": float(refine_res["tx_refine"]),
            "ty_refine": float(refine_res["ty_refine"]),
            "tz_refine": float(refine_res["tz_refine"]),
            "metrics": refine_res["metrics"],
        },
        "final_fit_partial_to_full_rgb": float(final_fit),
        "final_rmse_partial_to_full_rgb": float(final_rmse),
        "sam3d": args.sam3d,
        "sam_partial": args.sam_partial,
        "real_partial": args.real_partial,
        "pose_json": args.pose_json,
        "note": "pointmap-driven alignment with small 3D residual rotations",
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ pointmap 版配准完成")
    print(f"   StepA 最佳解释: {stepA['quat_name']}, mode={stepA['front_mode']}")
    print(f"   StepA: scale={stepA['scale']:.6f}, yaw={stepA['yaw_deg']:.2f}°")
    print(
        f"   StepB: scale={stepB['scale']:.6f}, "
        f"rx={stepB['rx_deg']:.2f}°, ry={stepB['ry_deg']:.2f}°, rz={stepB['rz_deg']:.2f}°"
    )
    print(
        f"   末端精修: scale={refine_res['scale_refine']:.5f}, "
        f"rx={refine_res['rx_refine_deg']:.3f}°, "
        f"ry={refine_res['ry_refine_deg']:.3f}°, "
        f"rz={refine_res['rz_refine_deg']:.3f}°, "
        f"tx={refine_res['tx_refine']:.4f}, "
        f"ty={refine_res['ty_refine']:.4f}, "
        f"tz={refine_res['tz_refine']:.4f}"
    )
    print(f"   最终覆盖率 (real_partial -> full @1.5cm): {final_fit*100:.2f}%")
    print(f"   最终 RMSE: {final_rmse:.4f}m")
    print(f"   full:      {full_path}")
    print(f"   shell:     {shell_path}")
    print(f"   samPartial:{sam_partial_path}")
    print(f"   merged:    {merged_path}")
    print(f"   result:    {result_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()