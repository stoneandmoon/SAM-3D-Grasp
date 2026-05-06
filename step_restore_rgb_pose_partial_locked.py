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
    """
    支持 mesh 或 point cloud。
    """
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
#    默认图像平面 = XY，深度轴 = Z
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
    max_iter: int = 40,
    trim_ratio: float = 0.80,
):
    """
    source -> target
    只优化平移
    """
    src = np.copy(source_pts)
    t_total = np.zeros(3, dtype=np.float64)

    # 初始：xy 中位数对齐 + z 前表面对齐
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
# 3. 只绕 RGB 相机 z 轴做 in-plane yaw
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


# =========================================================
# 4. quaternion 解码候选
# =========================================================
def decode_quaternion_candidates(raw_quat):
    """
    从原始四元数生成 4 种基础解释：
    1) raw 视为 [w,x,y,z]，取 R
    2) raw 视为 [w,x,y,z]，取 R.T
    3) raw 视为 [x,y,z,w]，取 R
    4) raw 视为 [x,y,z,w]，取 R.T
    """
    raw_quat = np.asarray(raw_quat, dtype=np.float64).reshape(-1)
    if len(raw_quat) != 4:
        raise RuntimeError(f"rotation_quat 长度不是 4: {raw_quat}")

    cands = []

    # raw = [w,x,y,z]
    w, x, y, z = raw_quat
    R1 = SciRot.from_quat([x, y, z, w]).as_matrix()
    cands.append(("wxyz_R", R1))
    cands.append(("wxyz_RT", R1.T))

    # raw = [x,y,z,w]
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
    search_min: float = 0.90,
    search_max: float = 1.10,
    search_steps: int = 31,
    icp_iter: int = 30,
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

        t_est, fit15, rmse, shell_aligned = translation_only_icp(
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
# 6. 主搜索：canonical -> RGB pose
#    关键改动：scale 来自 partial，不来自 GT
# =========================================================
def restore_canonical_to_rgb_pose(
    sam_pts_raw: np.ndarray,
    part_pts: np.ndarray,
    pose_json_path: str,
    front_mode: str = "min",
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

    # -----------------------------------------------------
    # Stage A: 粗搜索旋转解释（先不锁精确尺度）
    # 用 partial 的 2D 投影与 shell 的 2D 投影估一个粗尺度
    # -----------------------------------------------------
    coarse_results = []
    print("[4] 正在粗搜索 canonical -> RGB 粗姿态...", flush=True)

    total = len(base_rotations) * len(adapters) * 2
    cnt = 0

    part_sub = sample_points(part_pts, 1800)

    for base_name, base_R in base_rotations:
        for ai, A in enumerate(adapters):
            for compose_mode in ["left", "right"]:
                cnt += 1
                R_candidate = compose_candidate_rotation(base_R, A, compose_mode)

                sam_rot = sam_centered @ R_candidate.T
                shell_rot = extract_visible_shell_camera(
                    sam_rot,
                    grid_size=robust_diag(part_pts) / 70.0,
                    shell_thickness=robust_diag(part_pts) / 120.0,
                    front_mode=front_mode
                )

                if len(shell_rot) < 100:
                    continue

                scale_res = calibrate_scale_from_partial(
                    sample_points(shell_rot, 1800),
                    part_sub,
                    search_min=0.92,
                    search_max=1.08,
                    search_steps=21,
                    icp_iter=24
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
                    "fit15": scale_res["fit15"],
                    "rmse": scale_res["rmse"],
                })

                if cnt % 40 == 0 or cnt == total:
                    print(f"    [粗搜索] {cnt}/{total} 已完成", flush=True)

    if len(coarse_results) == 0:
        raise RuntimeError("粗搜索失败：没有找到有效 RGB 粗姿态。")

    coarse_results = sorted(
        coarse_results,
        key=lambda x: (x["fit15"], -x["rmse"]),
        reverse=True
    )[:6]

    print("[5] 粗搜索前 6 名：", flush=True)
    for i, c in enumerate(coarse_results, 1):
        print(
            f"    #{i}: {c['base_name']} + A[{c['adapter_idx']}] + {c['compose_mode']}, "
            f"scale={c['scale']:.6f}, fit15={c['fit15']*100:.2f}%, rmse={c['rmse']:.4f}m",
            flush=True
        )

    # -----------------------------------------------------
    # Stage B: 对前几名做更细的 scale + yaw 精修
    # -----------------------------------------------------
    best = None
    best_score = (-1.0, -1e9)

    print("[6] 对前几名做 scale + yaw 精修...", flush=True)

    for cand in coarse_results:
        R_candidate = cand["R_candidate"]

        local_best = None
        local_best_score = (-1.0, -1e9)

        # 围绕 coarse scale 再做小范围精修
        for s in cand["scale"] * np.linspace(0.97, 1.03, 9):
            sam_rgb = (sam_centered * s) @ R_candidate.T

            for yaw_deg in np.arange(-8.0, 8.01, 1.0):
                rot_center = np.mean(sam_rgb, axis=0)
                sam_rgb_yaw = rotate_about_center_z(sam_rgb, yaw_deg, rot_center)

                shell = extract_visible_shell_camera(
                    sam_rgb_yaw,
                    grid_size=robust_diag(part_pts) / 80.0,
                    shell_thickness=robust_diag(part_pts) / 140.0,
                    front_mode=front_mode
                )
                if len(shell) < 100:
                    continue

                scale_res = calibrate_scale_from_partial(
                    sample_points(shell, 2200),
                    sample_points(part_pts, 2200),
                    search_min=0.985,
                    search_max=1.015,
                    search_steps=11,
                    icp_iter=24
                )
                if scale_res is None:
                    continue

                total_scale = s * scale_res["scale"]

                # 注意：上面 shell 已经在 sam_rgb_yaw 空间里，所以这里只接受
                # 非常接近 1 的二次微调 scale。把它同步到完整点云时一起乘进去。
                score = (scale_res["fit15"], -scale_res["rmse"])
                if score > local_best_score:
                    local_best_score = score
                    local_best = {
                        "base_name": cand["base_name"],
                        "adapter_idx": cand["adapter_idx"],
                        "compose_mode": cand["compose_mode"],
                        "R_candidate": R_candidate.copy(),
                        "pre_scale": float(s),
                        "post_scale": float(scale_res["scale"]),
                        "scale": float(total_scale),
                        "yaw_deg": float(yaw_deg),
                        "translation": scale_res["translation"].copy(),
                        "fit15": float(scale_res["fit15"]),
                        "rmse": float(scale_res["rmse"]),
                    }

        if local_best is not None:
            score = (local_best["fit15"], -local_best["rmse"])
            if score > best_score:
                best_score = score
                best = copy.deepcopy(local_best)

    if best is None:
        raise RuntimeError("细搜索失败：没有得到最终 RGB 位姿。")

    print("[7] 最终最佳解释：", flush=True)
    print(
        f"    {best['base_name']} + A[{best['adapter_idx']}] + {best['compose_mode']}, "
        f"scale={best['scale']:.6f}, yaw={best['yaw_deg']:.2f}°, "
        f"fit15={best['fit15']*100:.2f}%, rmse={best['rmse']:.4f}m",
        flush=True
    )

    # -----------------------------------------------------
    # 构造最终 full_rgb_pose
    # -----------------------------------------------------
    sam_rgb = (sam_centered * best["scale"]) @ best["R_candidate"].T
    rot_center = np.mean(sam_rgb, axis=0)
    sam_rgb = rotate_about_center_z(sam_rgb, best["yaw_deg"], rot_center)
    sam_rgb = sam_rgb + best["translation"].reshape(1, 3)

    shell_rgb = extract_visible_shell_camera(
        sam_rgb,
        grid_size=robust_diag(part_pts) / 85.0,
        shell_thickness=robust_diag(part_pts) / 145.0,
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
    }


# =========================================================
# 7. CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d", required=True, help="canonical 完整点云 / 网格")
    parser.add_argument("--partial", required=True, help="真实 partial 点云（RGB 视角下）")
    parser.add_argument("--pose-json", required=True, help="sam3d_pose.json")
    parser.add_argument("--model-dir", default="", help="为了兼容旧命令保留，但这个版本不再用它锁尺度")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sam-samples", type=int, default=50000)
    parser.add_argument("--front-mode", type=str, default="min", choices=["min", "max"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    print("\n" + "=" * 60)
    print("[🚀] 启动：canonical 完整点云恢复到 RGB 视角位姿（partial 锁尺度版）")
    print("=" * 60, flush=True)

    sam_pts_raw = load_points_any(args.sam3d, mesh_samples=args.sam_samples)
    part_pts = load_points_any(args.partial, mesh_samples=args.sam_samples)

    result = restore_canonical_to_rgb_pose(
        sam_pts_raw=sam_pts_raw,
        part_pts=part_pts,
        pose_json_path=args.pose_json,
        front_mode=args.front_mode,
    )

    full_rgb = result["full_rgb_pose"]
    shell_rgb = result["visible_shell"]

    # 最终评测：partial -> full_rgb
    tree = cKDTree(full_rgb)
    final_dist, _ = tree.query(part_pts, workers=-1)
    final_fit = float(np.mean(final_dist < 0.015))
    final_rmse = float(np.sqrt(np.mean(final_dist ** 2)))

    # 输出文件
    full_path = os.path.join(args.out_dir, "full_rgb_pose.ply")
    shell_path = os.path.join(args.out_dir, "visible_rgb_shell.ply")
    merged_path = os.path.join(args.out_dir, "merged_rgb_pose.ply")
    result_json = os.path.join(args.out_dir, "pose_decode_result.json")

    save_point_cloud(full_path, full_rgb, color=[0.10, 0.45, 0.85])
    save_point_cloud(shell_path, shell_rgb, color=[0.10, 0.85, 0.20])
    save_merged_point_cloud(merged_path, full_rgb, part_pts)

    info = {
        "base_name": result["base_name"],
        "adapter_idx": int(result["adapter_idx"]),
        "compose_mode": result["compose_mode"],
        "scale": float(result["scale"]),
        "yaw_deg": float(result["yaw_deg"]),
        "translation": result["translation"].tolist(),
        "fit15_internal": float(result["fit15"]),
        "rmse_internal": float(result["rmse"]),
        "final_fit_partial_to_full_rgb": float(final_fit),
        "final_rmse_partial_to_full_rgb": float(final_rmse),
        "front_mode": args.front_mode,
        "sam3d": args.sam3d,
        "partial": args.partial,
        "pose_json": args.pose_json,
        "note": "scale is calibrated from partial, not from GT model",
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("✅ RGB 视角位姿恢复完成（partial 锁尺度版）")
    print(f"   最佳解释: {result['base_name']} + A[{result['adapter_idx']}] + {result['compose_mode']}")
    print(f"   scale:    {result['scale']:.6f}")
    print(f"   yaw:      {result['yaw_deg']:.2f}°")
    print(f"   最终覆盖率 (partial -> full @1.5cm): {final_fit*100:.2f}%")
    print(f"   最终 RMSE: {final_rmse:.4f}m")
    print(f"   full_rgb_pose:     {full_path}")
    print(f"   visible_rgb_shell: {shell_path}")
    print(f"   merged_rgb_pose:   {merged_path}")
    print(f"   result_json:       {result_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()