#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# =========================================================
# I/O
# =========================================================
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


def sample_points(pts: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) <= n:
        return pts.copy()
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


# =========================================================
# 几何
# =========================================================
def robust_diag(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    p1 = np.percentile(pts, 2, axis=0)
    p2 = np.percentile(pts, 98, axis=0)
    return float(np.linalg.norm(p2 - p1))


def robust_projected_diag_xy(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    xy = pts[:, :2]
    p1 = np.percentile(xy, 2, axis=0)
    p2 = np.percentile(xy, 98, axis=0)
    return float(np.linalg.norm(p2 - p1))


def build_xy_mask_keys(points: np.ndarray, pixel_size: float, x_min=None, y_min=None):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.empty((0,), dtype=np.int64), 0.0, 0.0

    if x_min is None:
        x_min = np.min(pts[:, 0]) - 2 * pixel_size
    if y_min is None:
        y_min = np.min(pts[:, 1]) - 2 * pixel_size

    ix = np.floor((pts[:, 0] - x_min) / pixel_size).astype(np.int64)
    iy = np.floor((pts[:, 1] - y_min) / pixel_size).astype(np.int64)
    base = int(max(iy.max() + 5, 10))
    keys = ix * base + iy
    return np.unique(keys), float(x_min), float(y_min)


# =========================================================
# 旋转 / 变换
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


def euler_matrix(rx_deg, ry_deg, rz_deg):
    return rotz(rz_deg) @ roty(ry_deg) @ rotx(rx_deg)


def apply_similarity_about_center(points: np.ndarray,
                                  center_src: np.ndarray,
                                  scale: float,
                                  R: np.ndarray,
                                  target_center: np.ndarray,
                                  delta_t: np.ndarray = None):
    pts = np.asarray(points, dtype=np.float64)
    center_src = np.asarray(center_src, dtype=np.float64).reshape(1, 3)
    target_center = np.asarray(target_center, dtype=np.float64).reshape(1, 3)
    if delta_t is None:
        delta_t = np.zeros(3, dtype=np.float64)
    delta_t = np.asarray(delta_t, dtype=np.float64).reshape(1, 3)
    return scale * ((pts - center_src) @ R.T) + target_center + delta_t


# =========================================================
# 壳层
# =========================================================
def extract_visible_shell_camera(
    pts_cam: np.ndarray,
    grid_size: float,
    shell_thickness: float,
    front_mode: str = "min",
):
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    if len(pts_cam) == 0:
        return pts_cam.copy()

    xy = pts_cam[:, :2]
    z = pts_cam[:, 2]

    xy_min = np.min(xy, axis=0)
    ij = np.floor((xy - xy_min) / max(grid_size, 1e-9)).astype(np.int64)

    uniq, inv = np.unique(ij, axis=0, return_inverse=True)

    if front_mode == "min":
        front = np.full(len(uniq), np.inf, dtype=np.float64)
        np.minimum.at(front, inv, z)
        mask = z <= front[inv] + shell_thickness
    else:
        front = np.full(len(uniq), -np.inf, dtype=np.float64)
        np.maximum.at(front, inv, z)
        mask = z >= front[inv] - shell_thickness

    return pts_cam[mask]


def crop_shell_xy(shell_pts: np.ndarray, ref_pts: np.ndarray, margin=0.02):
    shell_pts = np.asarray(shell_pts, dtype=np.float64)
    ref_pts = np.asarray(ref_pts, dtype=np.float64)

    x0, y0 = np.min(ref_pts[:, :2], axis=0) - margin
    x1, y1 = np.max(ref_pts[:, :2], axis=0) + margin

    mask = (
        (shell_pts[:, 0] >= x0) & (shell_pts[:, 0] <= x1) &
        (shell_pts[:, 1] >= y0) & (shell_pts[:, 1] <= y1)
    )
    cropped = shell_pts[mask]
    if len(cropped) < 50:
        return shell_pts
    return cropped


# =========================================================
# 评分
# =========================================================
def compute_partial_metrics(src_pts: np.ndarray,
                            tgt_pts: np.ndarray,
                            dist_thresh=0.015,
                            pixel_size=0.003):
    src_pts = np.asarray(src_pts, dtype=np.float64)
    tgt_pts = np.asarray(tgt_pts, dtype=np.float64)

    tree_tgt = cKDTree(tgt_pts)
    d_st, _ = tree_tgt.query(src_pts, workers=-1)

    tree_src = cKDTree(src_pts)
    d_ts, _ = tree_src.query(tgt_pts, workers=-1)

    fit_src_to_tgt = float(np.mean(d_st < dist_thresh))
    fit_tgt_to_src = float(np.mean(d_ts < dist_thresh))
    rmse_src_to_tgt = float(np.sqrt(np.mean(d_st ** 2)))
    rmse_tgt_to_src = float(np.sqrt(np.mean(d_ts ** 2)))

    all_xy = np.vstack([src_pts[:, :2], tgt_pts[:, :2]])
    x_min = np.min(all_xy[:, 0]) - 2 * pixel_size
    y_min = np.min(all_xy[:, 1]) - 2 * pixel_size

    k_s, _, _ = build_xy_mask_keys(src_pts[:, :2], pixel_size, x_min, y_min)
    k_t, _, _ = build_xy_mask_keys(tgt_pts[:, :2], pixel_size, x_min, y_min)

    set_s = set(k_s.tolist())
    set_t = set(k_t.tolist())
    union_num = max(len(set_s | set_t), 1)
    inter_num = len(set_s & set_t)
    iou2d = float(inter_num / union_num)

    score = (
        4.0 * fit_src_to_tgt
        + 4.0 * fit_tgt_to_src
        + 3.0 * iou2d
        - 10.0 * rmse_src_to_tgt
        - 10.0 * rmse_tgt_to_src
    )

    return {
        "score": float(score),
        "fit_src_to_tgt": fit_src_to_tgt,
        "fit_tgt_to_src": fit_tgt_to_src,
        "rmse_src_to_tgt": rmse_src_to_tgt,
        "rmse_tgt_to_src": rmse_tgt_to_src,
        "iou2d": iou2d,
    }


def compute_shell_metrics(real_partial_pts: np.ndarray,
                          shell_pts: np.ndarray,
                          dist_thresh=0.015,
                          pixel_size=0.003):
    real_partial_pts = np.asarray(real_partial_pts, dtype=np.float64)
    shell_pts = np.asarray(shell_pts, dtype=np.float64)

    shell_eval = crop_shell_xy(shell_pts, real_partial_pts, margin=0.025)
    shell_eval = sample_points(
        shell_eval,
        min(max(len(real_partial_pts) * 2, 4000), len(shell_eval))
    )

    tree_shell = cKDTree(shell_eval)
    d_rs, _ = tree_shell.query(real_partial_pts, workers=-1)

    tree_real = cKDTree(real_partial_pts)
    d_sr, _ = tree_real.query(shell_eval, workers=-1)

    fit_real_to_shell = float(np.mean(d_rs < dist_thresh))
    fit_shell_to_real = float(np.mean(d_sr < dist_thresh))
    rmse_real_to_shell = float(np.sqrt(np.mean(d_rs ** 2)))
    rmse_shell_to_real = float(np.sqrt(np.mean(d_sr ** 2)))

    all_xy = np.vstack([real_partial_pts[:, :2], shell_eval[:, :2]])
    x_min = np.min(all_xy[:, 0]) - 2 * pixel_size
    y_min = np.min(all_xy[:, 1]) - 2 * pixel_size

    k_r, _, _ = build_xy_mask_keys(real_partial_pts[:, :2], pixel_size, x_min, y_min)
    k_s, _, _ = build_xy_mask_keys(shell_eval[:, :2], pixel_size, x_min, y_min)

    set_r = set(k_r.tolist())
    set_s = set(k_s.tolist())
    union_num = max(len(set_r | set_s), 1)
    inter_num = len(set_r & set_s)
    iou2d = float(inter_num / union_num)

    score = (
        3.0 * fit_real_to_shell
        + 5.0 * fit_shell_to_real
        + 4.0 * iou2d
        - 8.0 * rmse_real_to_shell
        - 10.0 * rmse_shell_to_real
    )

    return {
        "score": float(score),
        "fit_real_to_shell": fit_real_to_shell,
        "fit_shell_to_real": fit_shell_to_real,
        "rmse_real_to_shell": rmse_real_to_shell,
        "rmse_shell_to_real": rmse_shell_to_real,
        "iou2d": iou2d,
    }


# =========================================================
# 平移 ICP
# =========================================================
def translation_only_icp(source_pts: np.ndarray,
                         target_pts: np.ndarray,
                         max_iter=16,
                         trim_ratio=0.88):
    src = np.asarray(source_pts, dtype=np.float64).copy()
    tgt = np.asarray(target_pts, dtype=np.float64)

    t_total = np.mean(tgt, axis=0) - np.mean(src, axis=0)
    src += t_total.reshape(1, 3)

    for _ in range(max_iter):
        tree = cKDTree(src)
        dist, idx = tree.query(tgt, workers=-1)

        thr = np.percentile(dist, trim_ratio * 100.0)
        mask = dist <= thr
        if np.sum(mask) < 20:
            break

        delta = np.mean(tgt[mask] - src[idx[mask]], axis=0)
        src += delta
        t_total += delta

        if np.linalg.norm(delta) < 1e-5:
            break

    return t_total, src


# =========================================================
# Stage A: sam_partial -> real_partial
# =========================================================
def evaluate_partial_bridge(sam_pts, real_pts, sam_center, real_center,
                            scale, rx, ry, rz, dist_thresh=0.015):
    R = euler_matrix(rx, ry, rz)
    src_try = apply_similarity_about_center(
        sam_pts, sam_center, scale, R, real_center
    )

    delta_t, src_icp = translation_only_icp(
        src_try, real_pts, max_iter=18, trim_ratio=0.88
    )
    metrics = compute_partial_metrics(
        src_icp, real_pts,
        dist_thresh=dist_thresh,
        pixel_size=max(robust_projected_diag_xy(real_pts) / 120.0, 0.002)
    )
    return {
        "scale": float(scale),
        "rx": float(rx),
        "ry": float(ry),
        "rz": float(rz),
        "R": R,
        "delta_t": delta_t,
        "metrics": metrics,
    }


def align_sam_partial_to_real(sam_partial_pts, real_partial_pts, dist_thresh=0.015):
    sam_center = np.mean(sam_partial_pts, axis=0)
    real_center = np.mean(real_partial_pts, axis=0)

    sam_sub = sample_points(sam_partial_pts, min(3000, len(sam_partial_pts)))
    real_sub = sample_points(real_partial_pts, min(3000, len(real_partial_pts)))

    base_scale = robust_diag(real_sub) / max(robust_diag(sam_sub), 1e-8)
    print(f"[StageA] base_scale={base_scale:.6f}")

    coarse = []
    for s_mul in [0.95, 0.98, 1.00, 1.02, 1.05]:
        for rx in [-8, -4, 0, 4, 8]:
            for ry in [-8, -4, 0, 4, 8]:
                for rz in [-8, -4, 0, 4, 8]:
                    rec = evaluate_partial_bridge(
                        sam_sub, real_sub,
                        sam_center, real_center,
                        base_scale * s_mul, rx, ry, rz,
                        dist_thresh=dist_thresh
                    )
                    m = rec["metrics"]
                    if m["fit_src_to_tgt"] < 0.55:
                        continue
                    if m["fit_tgt_to_src"] < 0.55:
                        continue
                    if m["iou2d"] < 0.30:
                        continue
                    coarse.append(rec)

    if len(coarse) == 0:
        raise RuntimeError("StageA 没找到满足约束的候选。")

    coarse = sorted(coarse, key=lambda x: x["metrics"]["score"], reverse=True)[:6]

    print("[StageA] 粗搜前 6 名：")
    for i, c in enumerate(coarse, 1):
        m = c["metrics"]
        print(
            f"    #{i}: s={c['scale']:.6f}, rx={c['rx']:.1f}, ry={c['ry']:.1f}, rz={c['rz']:.1f}, "
            f"fitS2R={m['fit_src_to_tgt']*100:.2f}%, fitR2S={m['fit_tgt_to_src']*100:.2f}%, "
            f"IoU2D={m['iou2d']*100:.2f}%, score={m['score']:.4f}"
        )

    best = None
    best_score = -1e18
    for idx, c in enumerate(coarse, 1):
        print(f"    [StageA fine {idx}/{len(coarse)}]", flush=True)
        for s_mul in [0.99, 1.00, 1.01]:
            for drx in [-2, -1, 0, 1, 2]:
                for dry in [-2, -1, 0, 1, 2]:
                    for drz in [-2, -1, 0, 1, 2]:
                        rec = evaluate_partial_bridge(
                            sam_sub, real_sub,
                            sam_center, real_center,
                            c["scale"] * s_mul,
                            c["rx"] + drx,
                            c["ry"] + dry,
                            c["rz"] + drz,
                            dist_thresh=dist_thresh
                        )
                        m = rec["metrics"]
                        if m["fit_src_to_tgt"] < 0.65:
                            continue
                        if m["fit_tgt_to_src"] < 0.60:
                            continue
                        if m["iou2d"] < 0.35:
                            continue
                        if m["score"] > best_score:
                            best = rec
                            best_score = m["score"]

    if best is None:
        raise RuntimeError("StageA 细搜后没有满足硬约束的结果。")

    print("[StageA] 最终最佳：")
    m = best["metrics"]
    print(
        f"    s={best['scale']:.6f}, rx={best['rx']:.2f}, ry={best['ry']:.2f}, rz={best['rz']:.2f}, "
        f"fitS2R={m['fit_src_to_tgt']*100:.2f}%, fitR2S={m['fit_tgt_to_src']*100:.2f}%, "
        f"IoU2D={m['iou2d']*100:.2f}%, rmseS2R={m['rmse_src_to_tgt']:.4f}"
    )

    return {
        "best": best,
        "sam_center": sam_center.tolist(),
        "real_center": real_center.tolist(),
    }


def apply_stageA(points, sam_center, real_center, stageA):
    return apply_similarity_about_center(
        points,
        sam_center,
        stageA["scale"],
        stageA["R"],
        real_center,
        stageA["delta_t"]
    )


# =========================================================
# Stage B: full shell local refine（只允许小平移）
# =========================================================
def refine_full_locally(full_init, real_partial_pts, dist_thresh=0.015):
    center_full = np.mean(full_init, axis=0)
    full_sub = sample_points(full_init, min(9000, len(full_init)))
    real_sub = sample_points(real_partial_pts, min(3500, len(real_partial_pts)))

    grid_size = max(robust_projected_diag_xy(real_sub) / 120.0, 0.0020)
    shell_thickness = max(robust_diag(real_sub) / 200.0, 0.0015)

    candidates = []
    for front_mode in ["min", "max"]:
        shell = extract_visible_shell_camera(
            full_sub,
            grid_size=grid_size,
            shell_thickness=shell_thickness,
            front_mode=front_mode
        )
        shell = crop_shell_xy(shell, real_sub, margin=0.02)
        if len(shell) < 50:
            continue

        delta_t, shell_icp = translation_only_icp(
            shell, real_sub, max_iter=18, trim_ratio=0.90
        )
        m = compute_shell_metrics(
            real_partial_pts=real_sub,
            shell_pts=shell_icp,
            dist_thresh=dist_thresh,
            pixel_size=grid_size
        )

        candidates.append({
            "front_mode": front_mode,
            "delta_t": delta_t,
            "metrics": m
        })

    if len(candidates) == 0:
        raise RuntimeError("StageB 失败：两个 front_mode 都没有得到有效壳层。")

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["metrics"]["fit_shell_to_real"],
            x["metrics"]["iou2d"],
            x["metrics"]["fit_real_to_shell"],
            -x["metrics"]["rmse_shell_to_real"]
        ),
        reverse=True
    )

    best0 = candidates[0]
    print(
        f"[StageB] 初选 mode={best0['front_mode']}, "
        f"fitR2S={best0['metrics']['fit_real_to_shell']*100:.2f}%, "
        f"fitS2R={best0['metrics']['fit_shell_to_real']*100:.2f}%, "
        f"IoU2D={best0['metrics']['iou2d']*100:.2f}%"
    )

    best = best0
    best_score = (
        best["metrics"]["fit_shell_to_real"] * 5.0 +
        best["metrics"]["iou2d"] * 4.0 +
        best["metrics"]["fit_real_to_shell"] * 3.0 -
        best["metrics"]["rmse_shell_to_real"] * 10.0
    )

    for dx in np.linspace(-0.006, 0.006, 7):
        for dy in np.linspace(-0.006, 0.006, 7):
            for dz in np.linspace(-0.006, 0.006, 7):
                delta_try = best0["delta_t"] + np.array([dx, dy, dz], dtype=np.float64)

                full_try = full_init + delta_try.reshape(1, 3)
                shell_try = extract_visible_shell_camera(
                    full_try,
                    grid_size=grid_size,
                    shell_thickness=shell_thickness,
                    front_mode=best0["front_mode"]
                )
                shell_try = crop_shell_xy(shell_try, real_sub, margin=0.02)
                if len(shell_try) < 50:
                    continue

                m = compute_shell_metrics(
                    real_partial_pts=real_sub,
                    shell_pts=shell_try,
                    dist_thresh=dist_thresh,
                    pixel_size=grid_size
                )

                if m["fit_shell_to_real"] < 0.25:
                    continue
                if m["iou2d"] < 0.10:
                    continue

                score = (
                    m["fit_shell_to_real"] * 5.0 +
                    m["iou2d"] * 4.0 +
                    m["fit_real_to_shell"] * 3.0 -
                    m["rmse_shell_to_real"] * 10.0
                )
                if score > best_score:
                    best_score = score
                    best = {
                        "front_mode": best0["front_mode"],
                        "delta_t": delta_try,
                        "metrics": m
                    }

    print("[StageB] 最终采用：")
    m = best["metrics"]
    print(
        f"    mode={best['front_mode']}, "
        f"fitR2S={m['fit_real_to_shell']*100:.2f}%, "
        f"fitS2R={m['fit_shell_to_real']*100:.2f}%, "
        f"IoU2D={m['iou2d']*100:.2f}%, "
        f"rmseR2S={m['rmse_real_to_shell']:.4f}"
    )

    full_final = full_init + best["delta_t"].reshape(1, 3)
    shell_final = extract_visible_shell_camera(
        full_final,
        grid_size=grid_size,
        shell_thickness=shell_thickness,
        front_mode=best["front_mode"]
    )
    shell_final = crop_shell_xy(shell_final, real_partial_pts, margin=0.02)

    return {
        "best": {
            "scale_local": 1.0,
            "rx": 0.0,
            "ry": 0.0,
            "rz": 0.0,
            "delta_t": best["delta_t"],
            "metrics": best["metrics"],
        },
        "front_mode": best["front_mode"],
        "grid_size": grid_size,
        "shell_thickness": shell_thickness,
        "full_final": full_final,
        "shell_final": shell_final,
    }


# =========================================================
# 主程序
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--sam-partial", required=True)
    parser.add_argument("--real-partial", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--full-samples", type=int, default=50000)
    parser.add_argument("--partial-samples", type=int, default=20000)
    parser.add_argument("--dist-thresh", type=float, default=0.015)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    print("\n" + "=" * 64)
    print("[🚀] 启动：sam_partial 桥梁稳定版")
    print("=" * 64)

    full_pts = load_points_any(args.full, mesh_samples=args.full_samples)
    sam_partial_pts = load_points_any(args.sam_partial, mesh_samples=args.partial_samples)
    real_partial_pts = load_points_any(args.real_partial, mesh_samples=args.partial_samples)

    # Stage A
    stageA_pack = align_sam_partial_to_real(
        sam_partial_pts=sam_partial_pts,
        real_partial_pts=real_partial_pts,
        dist_thresh=args.dist_thresh
    )
    stageA = stageA_pack["best"]
    sam_center = np.asarray(stageA_pack["sam_center"], dtype=np.float64)
    real_center = np.asarray(stageA_pack["real_center"], dtype=np.float64)

    sam_partial_aligned = apply_stageA(sam_partial_pts, sam_center, real_center, stageA)
    full_init = apply_stageA(full_pts, sam_center, real_center, stageA)

    # Stage B
    stageB_pack = refine_full_locally(
        full_init=full_init,
        real_partial_pts=real_partial_pts,
        dist_thresh=args.dist_thresh
    )

    shell_metrics = compute_shell_metrics(
        real_partial_pts=real_partial_pts,
        shell_pts=stageB_pack["shell_final"],
        dist_thresh=args.dist_thresh,
        pixel_size=stageB_pack["grid_size"]
    )

    tree_full = cKDTree(stageB_pack["full_final"])
    d_rf, _ = tree_full.query(real_partial_pts, workers=-1)
    full_ref_fit = float(np.mean(d_rf < args.dist_thresh))
    full_ref_rmse = float(np.sqrt(np.mean(d_rf ** 2)))

    save_point_cloud(os.path.join(args.out_dir, "stageA_sam_partial_to_real.ply"), sam_partial_aligned, color=[0.85, 0.85, 0.10])
    save_merged_point_cloud(os.path.join(args.out_dir, "stageA_merged_sam_partial_vs_real.ply"), sam_partial_aligned, real_partial_pts)

    save_point_cloud(os.path.join(args.out_dir, "final_full_aligned_to_real.ply"), stageB_pack["full_final"], color=[0.10, 0.45, 0.85])
    save_point_cloud(os.path.join(args.out_dir, "final_visible_shell_aligned_to_real.ply"), stageB_pack["shell_final"], color=[0.10, 0.85, 0.20])
    save_merged_point_cloud(os.path.join(args.out_dir, "final_merged_full_vs_real.ply"), stageB_pack["full_final"], real_partial_pts)
    save_merged_point_cloud(os.path.join(args.out_dir, "final_merged_shell_vs_real.ply"), stageB_pack["shell_final"], real_partial_pts)

    result = {
        "stageA": {
            "sam_center": stageA_pack["sam_center"],
            "real_center": stageA_pack["real_center"],
            "best": {
                "scale": stageA["scale"],
                "rx": stageA["rx"],
                "ry": stageA["ry"],
                "rz": stageA["rz"],
                "delta_t": stageA["delta_t"].tolist(),
                "metrics": stageA["metrics"],
            }
        },
        "stageB": {
            "front_mode": stageB_pack["front_mode"],
            "best": {
                "scale_local": stageB_pack["best"]["scale_local"],
                "rx": stageB_pack["best"]["rx"],
                "ry": stageB_pack["best"]["ry"],
                "rz": stageB_pack["best"]["rz"],
                "delta_t": stageB_pack["best"]["delta_t"].tolist(),
                "metrics": stageB_pack["best"]["metrics"],
            }
        },
        "final_shell_metrics": shell_metrics,
        "final_full_reference_only": {
            "real_partial_to_full_fit15": full_ref_fit,
            "real_partial_to_full_rmse": full_ref_rmse
        }
    }

    with open(os.path.join(args.out_dir, "bridge_registration_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 56)
    print("✅ 稳定版完成")
    print(f"   StageA best: s={stageA['scale']:.6f}, "
          f"rx={stageA['rx']:.2f}, ry={stageA['ry']:.2f}, rz={stageA['rz']:.2f}, "
          f"fitS2R={stageA['metrics']['fit_src_to_tgt']*100:.2f}%, "
          f"fitR2S={stageA['metrics']['fit_tgt_to_src']*100:.2f}%, "
          f"IoU2D={stageA['metrics']['iou2d']*100:.2f}%")
    print(f"   StageB best: s={stageB_pack['best']['scale_local']:.6f}, "
          f"rx={stageB_pack['best']['rx']:.2f}, ry={stageB_pack['best']['ry']:.2f}, rz={stageB_pack['best']['rz']:.2f}, "
          f"fitR2S={stageB_pack['best']['metrics']['fit_real_to_shell']*100:.2f}%, "
          f"fitS2R={stageB_pack['best']['metrics']['fit_shell_to_real']*100:.2f}%, "
          f"IoU2D={stageB_pack['best']['metrics']['iou2d']*100:.2f}%")
    print(f"   Final SHELL real->shell: {shell_metrics['fit_real_to_shell']*100:.2f}%")
    print(f"   Final SHELL shell->real: {shell_metrics['fit_shell_to_real']*100:.2f}%")
    print(f"   Final SHELL 2D IoU:      {shell_metrics['iou2d']*100:.2f}%")
    print(f"   Final SHELL RMSE:        {shell_metrics['rmse_real_to_shell']:.4f} m")
    print(f"   输出目录: {args.out_dir}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()