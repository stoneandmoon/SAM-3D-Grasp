#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
yazishuzhizhousuanfa_feature.py

作用：
  用 FPFH + RANSAC 特征配准，替代旧的 single ↔ joint_duck_candidate 配准。

解决的问题：
  旧版 06_axis_transfer 中 single_aligned_to_joint_candidate.ply 明显姿态错误，
  导致 normal_table_single 错，继而 Step 9 / Step 10 的尺度和位姿都被带偏。

核心流程：
  1. 读取 single SAM3D 点云
  2. 读取 joint SAM3D 点云
  3. 从 joint 中拟合桌面平面
  4. 去掉桌面，得到 joint duck candidate
  5. 对 single 和 joint duck 做归一化
  6. 用 FPFH + RANSAC 做全局特征配准
  7. 用 ICP 微调
  8. 把 joint 桌面法向转回 single 坐标系
  9. 输出 table_normal_in_single_duck_frame.json

输出兼容旧 pipeline：
  out_dir/
    joint_table_points.ply
    joint_non_table_candidate.ply
    joint_above_table_duck_candidate.ply
    single_aligned_to_joint_candidate.ply
    compare_single_joint_candidate.ply
    joint_table_plane.json
    single_to_joint_transform.json
    table_normal_in_single_duck_frame.json
"""

import os
import json
import argparse
import numpy as np
import open3d as o3d


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def robust_bbox(points, q0=2.0, q1=98.0):
    points = np.asarray(points, dtype=np.float64)
    lo = np.percentile(points, q0, axis=0)
    hi = np.percentile(points, q1, axis=0)
    return lo, hi, hi - lo, float(np.linalg.norm(hi - lo))


def np_to_pcd(points, color=None):
    points = np.asarray(points, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None and len(points) > 0:
        color = np.asarray(color, dtype=np.float64).reshape(1, 3)
        colors = np.repeat(color, len(points), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_pcd(points, path, color=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    pcd = np_to_pcd(points, color=color)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def save_compare(single_aligned, joint_duck, table_points, path):
    p1 = np_to_pcd(single_aligned, color=[0.1, 0.35, 1.0])
    p2 = np_to_pcd(joint_duck, color=[1.0, 0.05, 0.02])
    p3 = np_to_pcd(table_points, color=[0.1, 0.9, 0.2])
    ok = o3d.io.write_point_cloud(path, p1 + p2 + p3)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}")


def load_geometry_points(path, sample_points=180000):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    print(f"[Load] {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"  mesh vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        return np.asarray(pcd.points, dtype=np.float64)

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        print(f"  point cloud points={len(pcd.points)}")
        return np.asarray(pcd.points, dtype=np.float64)

    raise RuntimeError(f"无法读取 geometry: {path}")


def fit_joint_table(points, args):
    points = np.asarray(points, dtype=np.float64)
    _, _, _, diag = robust_bbox(points)

    if args.table_dist > 0:
        table_dist = args.table_dist
    else:
        table_dist = max(0.004 * diag, 0.003)

    pcd = np_to_pcd(points)

    print("\n[Joint Table RANSAC]")
    print(f"  robust diag = {diag:.6f}")
    print(f"  table_dist  = {table_dist:.6f}")

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=table_dist,
        ransac_n=3,
        num_iterations=args.ransac_iters,
    )

    a, b, c, d = [float(x) for x in plane_model]
    n = normalize([a, b, c])
    d = d / max(np.linalg.norm([a, b, c]), 1e-12)

    h = points @ n + d

    # 让非桌面主体位于正侧
    non_inlier_mask = np.ones(len(points), dtype=bool)
    non_inlier_mask[np.asarray(inliers, dtype=np.int64)] = False
    non_table_init = points[non_inlier_mask]

    if len(non_table_init) > 100:
        h_non = non_table_init @ n + d
        if np.median(h_non) < 0:
            n = -n
            d = -d
            h = -h
            print("[Orient] 翻转 joint table normal，使物体位于正侧。")

    table_shell = np.abs(h) < table_dist * args.table_shell_mult
    above_thresh = table_dist * args.above_mult

    table_points = points[table_shell]
    non_table_points = points[~table_shell]
    joint_duck = points[h > above_thresh]

    if len(joint_duck) < 500:
        print("[WARN] joint_duck 点太少，退回使用非桌面正侧点。")
        joint_duck = non_table_points[(non_table_points @ n + d) > 0]

    print("\n[Joint Split]")
    print(f"  all points       = {len(points)}")
    print(f"  table points     = {len(table_points)}")
    print(f"  non-table points = {len(non_table_points)}")
    print(f"  joint duck       = {len(joint_duck)}")
    print(f"  normal           = {n.tolist()}")
    print(f"  d                = {d:.8f}")
    print(f"  h p01/p50/p99    = {np.percentile(h,1):.6f}, {np.percentile(h,50):.6f}, {np.percentile(h,99):.6f}")

    return n, float(d), table_points, non_table_points, joint_duck, table_dist


def filter_joint_duck(joint_duck, n, d, percentile=10.0):
    h = joint_duck @ n + d

    if percentile <= 0:
        return joint_duck

    th = np.percentile(h, percentile)
    out = joint_duck[h >= th]

    if len(out) < 500:
        print("[WARN] filter 后 joint_duck 太少，使用原始 joint_duck。")
        return joint_duck

    print("\n[Filter Joint Duck]")
    print(f"  remove bottom percentile = {percentile}")
    print(f"  before = {len(joint_duck)}")
    print(f"  after  = {len(out)}")
    print(f"  h_th   = {th:.6f}")

    return out


def normalize_points_for_feature(points):
    points = np.asarray(points, dtype=np.float64)

    lo, hi, ext, diag = robust_bbox(points)
    center = (lo + hi) / 2.0

    if diag < 1e-12:
        raise RuntimeError("点云 robust diag 过小，无法归一化。")

    points_norm = (points - center.reshape(1, 3)) / diag

    return points_norm, center, diag


def preprocess_feature(points_norm, voxel_size):
    pcd = np_to_pcd(points_norm)

    pcd_down = pcd.voxel_down_sample(voxel_size)

    if len(pcd_down.points) < 100:
        print("[WARN] voxel downsample 后点太少，使用原始点云估特征。")
        pcd_down = pcd

    radius_normal = voxel_size * 2.5
    radius_feature = voxel_size * 5.0

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=30,
        )
    )

    pcd_down.orient_normals_consistent_tangent_plane(30)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100,
        ),
    )

    return pcd_down, fpfh


def run_feature_registration(single_points, joint_duck_points, args):
    """
    先把两个点云分别归一化，再做 FPFH/RANSAC/ICP。
    返回原始坐标系下的 similarity:
      p_joint = scale * R @ p_single + t
    """
    src_norm, src_center, src_diag = normalize_points_for_feature(single_points)
    tgt_norm, tgt_center, tgt_diag = normalize_points_for_feature(joint_duck_points)

    voxel_size = args.voxel_size
    if voxel_size <= 0:
        voxel_size = 0.035

    print("\n[Feature Registration]")
    print(f"  src_diag = {src_diag:.6f}")
    print(f"  tgt_diag = {tgt_diag:.6f}")
    print(f"  voxel    = {voxel_size:.6f}")

    src_down, src_fpfh = preprocess_feature(src_norm, voxel_size)
    tgt_down, tgt_fpfh = preprocess_feature(tgt_norm, voxel_size)

    print(f"  src_down points = {len(src_down.points)}")
    print(f"  tgt_down points = {len(tgt_down.points)}")

    distance_threshold = voxel_size * args.ransac_dist_mult

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            args.ransac_max_iter,
            args.ransac_confidence,
        ),
    )

    print("\n[RANSAC]")
    print(f"  fitness = {result_ransac.fitness:.6f}")
    print(f"  rmse    = {result_ransac.inlier_rmse:.6f}")
    print(f"  T_norm  =\n{result_ransac.transformation}")

    if args.icp_iters <= 0:
        print("\n[ICP] skipped because --icp-iters <= 0")
        Tn = result_ransac.transformation
        R = Tn[:3, :3].copy()
        t_norm = Tn[:3, 3].copy()

        # 正交化 R
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1.0
            R = U @ Vt

        scale = float(tgt_diag / src_diag)
        t = tgt_center + tgt_diag * t_norm - scale * (R @ src_center)

        return {
            "scale": scale,
            "R": R,
            "t": t,
            "src_center": src_center,
            "tgt_center": tgt_center,
            "src_diag": src_diag,
            "tgt_diag": tgt_diag,
            "ransac_fitness": float(result_ransac.fitness),
            "ransac_rmse": float(result_ransac.inlier_rmse),
            "icp_fitness": float(result_ransac.fitness),
            "icp_rmse": float(result_ransac.inlier_rmse),
            "T_norm": Tn.tolist(),
            "voxel_size": float(voxel_size),
        }

    # ICP 微调，仍然在归一化空间
    src_pcd = np_to_pcd(src_norm)
    tgt_pcd = np_to_pcd(tgt_norm)

    src_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.5,
            max_nn=30,
        )
    )
    tgt_pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2.5,
            max_nn=30,
        )
    )

    icp_threshold = voxel_size * args.icp_dist_mult

    if args.icp_point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    result_icp = o3d.pipelines.registration.registration_icp(
        src_pcd,
        tgt_pcd,
        icp_threshold,
        result_ransac.transformation,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=args.icp_iters,
        ),
    )

    print("\n[ICP]")
    print(f"  fitness = {result_icp.fitness:.6f}")
    print(f"  rmse    = {result_icp.inlier_rmse:.6f}")
    print(f"  T_norm  =\n{result_icp.transformation}")

    Tn = result_icp.transformation
    R = Tn[:3, :3].copy()
    t_norm = Tn[:3, 3].copy()

    # 重新正交化 R，避免数值误差
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    # 从归一化空间恢复到原始 joint 坐标：
    # q_norm = R @ p_norm + t_norm
    # (q - tgt_center) / tgt_diag = R @ ((p - src_center) / src_diag) + t_norm
    # q = (tgt_diag/src_diag) * R @ p + tgt_center + tgt_diag*t_norm - (tgt_diag/src_diag)*R@src_center
    scale = float(tgt_diag / src_diag)
    t = tgt_center + tgt_diag * t_norm - scale * (R @ src_center)

    return {
        "scale": scale,
        "R": R,
        "t": t,
        "src_center": src_center,
        "tgt_center": tgt_center,
        "src_diag": src_diag,
        "tgt_diag": tgt_diag,
        "ransac_fitness": float(result_ransac.fitness),
        "ransac_rmse": float(result_ransac.inlier_rmse),
        "icp_fitness": float(result_icp.fitness),
        "icp_rmse": float(result_icp.inlier_rmse),
        "T_norm": result_icp.transformation.tolist(),
        "voxel_size": float(voxel_size),
    }


def apply_similarity(points, scale, R, t):
    return float(scale) * (np.asarray(points) @ np.asarray(R).T) + np.asarray(t).reshape(1, 3)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--single", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--joint-duck-override", default="", help="可选：直接指定干净的 joint duck candidate ply，跳过自动 h>above_thresh 提取")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--sample-points", type=int, default=180000)

    parser.add_argument("--table-dist", type=float, default=-1.0)
    parser.add_argument("--ransac-iters", type=int, default=3000)
    parser.add_argument("--table-shell-mult", type=float, default=1.8)
    parser.add_argument("--above-mult", type=float, default=3.0)
    parser.add_argument("--joint-bottom-filter-percentile", type=float, default=10.0)

    parser.add_argument("--voxel-size", type=float, default=0.035)
    parser.add_argument("--ransac-dist-mult", type=float, default=1.5)
    parser.add_argument("--ransac-max-iter", type=int, default=800000)
    parser.add_argument("--ransac-confidence", type=float, default=0.999)
    parser.add_argument("--icp-dist-mult", type=float, default=1.2)
    parser.add_argument("--icp-iters", type=int, default=80)
    parser.add_argument("--icp-point-to-plane", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] FPFH feature registration for axis transfer")
    print("=" * 80)

    single = load_geometry_points(args.single, sample_points=args.sample_points)
    joint = load_geometry_points(args.joint, sample_points=args.sample_points)

    n_joint, d_joint, table_points, non_table, joint_duck, table_dist = fit_joint_table(joint, args)

    if args.joint_duck_override and args.joint_duck_override.strip():
        print("\n[Override] 使用手动指定的 clean joint duck candidate:")
        print(f"  {args.joint_duck_override}")
        joint_duck_override = load_geometry_points(args.joint_duck_override, sample_points=args.sample_points)
        joint_duck = joint_duck_override
        joint_duck_for_align = joint_duck_override
    else:
        joint_duck_for_align = filter_joint_duck(
            joint_duck,
            n_joint,
            d_joint,
            percentile=args.joint_bottom_filter_percentile,
        )

    save_pcd(table_points, os.path.join(args.out_dir, "joint_table_points.ply"), color=[0.1, 0.9, 0.2])
    save_pcd(non_table, os.path.join(args.out_dir, "joint_non_table_candidate.ply"), color=[1.0, 0.2, 0.05])
    save_pcd(joint_duck, os.path.join(args.out_dir, "joint_above_table_duck_candidate.ply"), color=[1.0, 0.05, 0.02])
    save_pcd(joint_duck_for_align, os.path.join(args.out_dir, "joint_duck_for_feature_alignment.ply"), color=[1.0, 0.45, 0.05])

    reg = run_feature_registration(single, joint_duck_for_align, args)

    scale = reg["scale"]
    R = reg["R"]
    t = reg["t"]

    single_aligned = apply_similarity(single, scale, R, t)

    normal_table_single = normalize(R.T @ n_joint)

    out_single = os.path.join(args.out_dir, "single_aligned_to_joint_candidate.ply")
    out_compare = os.path.join(args.out_dir, "compare_single_joint_candidate.ply")
    out_plane = os.path.join(args.out_dir, "joint_table_plane.json")
    out_transform = os.path.join(args.out_dir, "single_to_joint_transform.json")
    out_normal = os.path.join(args.out_dir, "table_normal_in_single_duck_frame.json")

    save_pcd(single_aligned, out_single, color=[0.1, 0.35, 1.0])
    save_compare(single_aligned, joint_duck, table_points, out_compare)

    with open(out_plane, "w", encoding="utf-8") as f:
        json.dump({
            "plane_form": "normal dot x + d = 0",
            "normal_table_joint": n_joint.tolist(),
            "d_table_joint": float(d_joint),
            "table_dist_used": float(table_dist),
            "source_joint": os.path.abspath(args.joint),
        }, f, indent=2, ensure_ascii=False)

    with open(out_transform, "w", encoding="utf-8") as f:
        json.dump({
            "definition": "p_joint = scale * R_single_to_joint @ p_single + t_single_to_joint",
            "note": "This scale is only for single-to-joint visualization/alignment. Do not use it as physical RGB-D scale.",
            "scale": float(scale),
            "R_single_to_joint": R.tolist(),
            "t_single_to_joint": t.tolist(),
            "normal_table_joint": n_joint.tolist(),
            "normal_table_single": normal_table_single.tolist(),
            "feature_registration": {
                "ransac_fitness": reg["ransac_fitness"],
                "ransac_rmse": reg["ransac_rmse"],
                "icp_fitness": reg["icp_fitness"],
                "icp_rmse": reg["icp_rmse"],
                "voxel_size": reg["voxel_size"],
                "src_diag": reg["src_diag"],
                "tgt_diag": reg["tgt_diag"],
                "T_norm": reg["T_norm"],
            }
        }, f, indent=2, ensure_ascii=False)

    with open(out_normal, "w", encoding="utf-8") as f:
        json.dump({
            "definition": "normal_table_single = R_single_to_joint.T @ normal_table_joint",
            "normal_table_single": normal_table_single.tolist(),
            "normal_single": normal_table_single.tolist(),
            "normal_table_joint": n_joint.tolist(),
            "d_table_joint": float(d_joint),
            "source_single": os.path.abspath(args.single),
            "source_joint": os.path.abspath(args.joint),
            "source_transform": os.path.abspath(out_transform),
            "warning": "single-to-joint scale is not physical RGB-D scale. Final physical scale should still be estimated by table height after this normal is correct.",
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "█" * 80)
    print("✅ 完成：FPFH feature axis transfer")
    print("重点输出：")
    print(f"  compare : {os.path.abspath(out_compare)}")
    print(f"  normal  : {os.path.abspath(out_normal)}")
    print(f"  transform: {os.path.abspath(out_transform)}")
    print("█" * 80)

    print("\n关键结果：")
    print(f"  single_to_joint_visual_scale = {scale:.6f}")
    print(f"  normal_table_single          = {normal_table_single.tolist()}")
    print(f"  ransac_fitness               = {reg['ransac_fitness']:.6f}")
    print(f"  ransac_rmse                  = {reg['ransac_rmse']:.6f}")
    print(f"  icp_fitness                  = {reg['icp_fitness']:.6f}")
    print(f"  icp_rmse                     = {reg['icp_rmse']:.6f}")


if __name__ == "__main__":
    main()
