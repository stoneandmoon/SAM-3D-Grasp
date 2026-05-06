#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step_extract_joint_table_and_transfer_normal.py

用途：
  从 SAM3D 联合生成结果中提取桌面平面，并尝试把桌面法向迁移回
  “单独生成鸭子”的 SAM3D 坐标系。

输入：
  1. 单独鸭子点云 / mesh:
       sam3d_duck_clean.ply

  2. 联合生成点云 / mesh:
       sam3d_duck_table_joint_clean.ply

输出：
  out_dir/
    joint_table_points.ply
    joint_non_table_candidate.ply
    joint_above_table_duck_candidate.ply
    joint_table_plane.json
    single_aligned_to_joint_candidate.ply
    compare_single_joint_candidate.ply
    single_to_joint_transform.json
    table_normal_in_single_duck_frame.json

核心思想：
  联合模型里：
    - 桌面用于拟合 n_table_joint
    - 去桌面后的鸭子主体用于估计 single duck -> joint duck 的旋转

  如果得到：
    p_joint = s * R * p_single + t

  那么 joint 桌面法向转回 single 坐标系：
    n_table_single = R.T @ n_table_joint

  joint 平面方程：
    n_joint · x_joint + d_joint = 0

  转到 single 坐标系：
    n_single · x_single + d_single = 0

  其中：
    n_single = R.T @ n_joint
    d_single = (n_joint · t + d_joint) / s

注意：
  joint_duck_candidate 不是完美鸭子，只是用来估计 single 与 joint 的相对旋转。
"""

import os
import json
import argparse
import itertools
import copy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ============================================================
# 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_vec(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def geometry_to_pcd(path, sample_points=120000):
    """
    兼容读取：
      - mesh ply / obj
      - point cloud ply

    如果是 mesh，则采样成点云。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")

    print(f"[Load] {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"  读取为 mesh: vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(sample_points))
        return pcd

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        print(f"  读取为 point cloud: points={len(pcd.points)}")
        return pcd

    raise RuntimeError(f"Open3D 无法读取有效 geometry: {path}")


def np_to_pcd(points, color=None):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None and len(points) > 0:
        colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def save_pcd(points, path, color=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    pcd = np_to_pcd(points, color=color)
    ok = o3d.io.write_point_cloud(path, pcd)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def get_points(pcd):
    return np.asarray(pcd.points, dtype=np.float64)


def bbox_diag(points):
    points = np.asarray(points, dtype=np.float64)
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def downsample_points(points, max_points=60000, seed=0):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n <= max_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


# ============================================================
# 多平面候选 + 桌面平面选择
# ============================================================

def plane_project_area(points, normal):
    """
    估计平面内点分布面积：
      把点投影到平面内两个 PCA 轴，取二维 bbox 面积。
    """
    if len(points) < 10:
        return 0.0, [0.0, 0.0]

    normal = normalize_vec(normal)
    c = points.mean(axis=0)
    X = points - c

    # 构造平面内正交基
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    u = normalize_vec(np.cross(normal, tmp))
    v = normalize_vec(np.cross(normal, u))

    pu = X @ u
    pv = X @ v

    extent_u = float(np.percentile(pu, 98) - np.percentile(pu, 2))
    extent_v = float(np.percentile(pv, 98) - np.percentile(pv, 2))
    area = max(extent_u, 1e-12) * max(extent_v, 1e-12)

    return area, [extent_u, extent_v]


def segment_table_plane_multi(
    points,
    ransac_dist_ratio=0.006,
    max_planes=6,
    min_inlier_ratio=0.04,
    ransac_n=3,
    iterations=2500,
    seed=0,
):
    """
    从联合点云里找桌面平面。

    做法：
      1. 对点云迭代分割多个平面候选
      2. 对每个候选计算：
          - inlier 数量
          - 平面内展开面积
      3. 选择 score 最大的平面

    一般桌面顶面是最大、最平、展开面积最大的候选。
    """
    rng = np.random.default_rng(seed)

    pts_full = np.asarray(points, dtype=np.float64)
    diag = bbox_diag(pts_full)
    dist_thresh = float(ransac_dist_ratio * diag)

    pts_work = downsample_points(pts_full, max_points=80000, seed=seed)
    pcd_work = np_to_pcd(pts_work)

    candidates = []
    remaining = pcd_work

    print("\n" + "=" * 80)
    print("[Plane] 开始多平面 RANSAC")
    print(f"[Plane] bbox_diag={diag:.6f}")
    print(f"[Plane] ransac distance threshold={dist_thresh:.6f}")
    print("=" * 80)

    for k in range(max_planes):
        if len(remaining.points) < 1000:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=ransac_n,
            num_iterations=iterations,
        )

        inliers = np.asarray(inliers, dtype=np.int64)
        rem_pts = get_points(remaining)

        if len(inliers) < max(100, int(min_inlier_ratio * len(pts_work))):
            print(f"[Plane {k}] inliers 太少，停止。inliers={len(inliers)}")
            break

        a, b, c, d = [float(x) for x in plane_model]
        n = normalize_vec([a, b, c])
        d = d / max(np.linalg.norm([a, b, c]), 1e-12)

        plane_pts = rem_pts[inliers]
        area, extents = plane_project_area(plane_pts, n)

        score = len(inliers) * np.sqrt(max(area, 1e-12))

        candidates.append({
            "k": k,
            "normal": n,
            "d": d,
            "inlier_count": int(len(inliers)),
            "area": float(area),
            "extents": extents,
            "score": float(score),
        })

        print(
            f"[Plane {k}] "
            f"inliers={len(inliers)}, "
            f"area={area:.6f}, "
            f"extents={extents}, "
            f"score={score:.6f}, "
            f"normal={n.tolist()}, d={d:.6f}"
        )

        # 删除当前平面内点，继续找下一个平面
        remaining = remaining.select_by_index(inliers.tolist(), invert=True)

    if len(candidates) == 0:
        raise RuntimeError("没有找到任何有效平面，请调大 --ransac-dist-ratio 或检查联合点云。")

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    print("\n[Plane] 选择 score 最大的候选作为桌面:")
    print(json.dumps({
        "k": best["k"],
        "normal": best["normal"].tolist(),
        "d": best["d"],
        "inlier_count": best["inlier_count"],
        "area": best["area"],
        "extents": best["extents"],
        "score": best["score"],
    }, indent=2, ensure_ascii=False))

    return best, candidates, dist_thresh, diag


def orient_table_normal_to_duck_side(points, normal, d):
    """
    RANSAC 平面法向有正负歧义。
    这里把法向调整到“更像鸭子主体所在的一侧”。

    经验规则：
      计算所有点到平面的 signed distance。
      比较上侧 99% 分位和下侧 1% 分位的绝对值。
      鸭子高度通常比桌板厚度更大，因此让更高的一侧为正。
    """
    points = np.asarray(points, dtype=np.float64)
    normal = normalize_vec(normal)
    dist = points @ normal + float(d)

    p99 = float(np.percentile(dist, 99))
    p01 = float(np.percentile(dist, 1))

    if abs(p01) > abs(p99):
        normal = -normal
        d = -d
        dist = -dist
        print("[Plane] 法向已翻转，使鸭子主体方向为正。")
    else:
        print("[Plane] 法向保持不变。")

    print(f"[Plane] signed dist percentile: p01={np.percentile(dist,1):.6f}, "
          f"p50={np.percentile(dist,50):.6f}, p99={np.percentile(dist,99):.6f}")

    return normal, float(d), dist


# ============================================================
# 点云分离
# ============================================================

def keep_largest_dbscan_cluster(points, eps, min_points=20):
    """
    可选：对 above-table duck candidate 做 DBSCAN，只保留最大簇。
    防止桌面边缘、噪声被带进去。
    """
    if len(points) < min_points:
        return points

    pcd = np_to_pcd(points)
    labels = np.asarray(
        pcd.cluster_dbscan(eps=float(eps), min_points=int(min_points), print_progress=False)
    )

    valid = labels >= 0
    if valid.sum() == 0:
        print("[DBSCAN] 没有有效簇，跳过。")
        return points

    unique, counts = np.unique(labels[valid], return_counts=True)
    best_label = unique[np.argmax(counts)]
    kept = points[labels == best_label]

    print(f"[DBSCAN] clusters={len(unique)}, keep_label={best_label}, "
          f"points {len(points)} -> {len(kept)}")

    return kept


def separate_table_and_duck(
    points,
    normal,
    d,
    diag,
    strict_ratio=0.006,
    remove_ratio=0.018,
    above_ratio=0.035,
    use_largest_cluster=True,
    cluster_eps_ratio=0.035,
):
    """
    根据桌面平面把联合点云拆成几部分：

    table_strict:
      高置信桌面点，主要用于保存和可视化。

    non_table:
      删除桌面附近后的剩余点。
      注意：可能包含桌子侧面、底部，也可能包含鸭子。

    above_duck:
      明显位于桌面上方的点。
      主要用作 joint_duck_candidate。
    """
    points = np.asarray(points, dtype=np.float64)

    dist = points @ normal + float(d)

    strict_thresh = float(strict_ratio * diag)
    remove_thresh = float(remove_ratio * diag)
    above_thresh = float(above_ratio * diag)

    table_strict_mask = np.abs(dist) < strict_thresh
    non_table_mask = np.abs(dist) > remove_thresh
    above_mask = dist > above_thresh

    table_pts = points[table_strict_mask]
    non_table_pts = points[non_table_mask]
    above_pts = points[above_mask]

    print("\n" + "=" * 80)
    print("[Separate] 点云分离统计")
    print(f"strict_thresh={strict_thresh:.6f}")
    print(f"remove_thresh={remove_thresh:.6f}")
    print(f"above_thresh ={above_thresh:.6f}")
    print(f"all points   ={len(points)}")
    print(f"table_strict ={len(table_pts)}")
    print(f"non_table    ={len(non_table_pts)}")
    print(f"above_duck   ={len(above_pts)}")
    print("=" * 80)

    if use_largest_cluster and len(above_pts) > 100:
        eps = float(cluster_eps_ratio * diag)
        above_pts = keep_largest_dbscan_cluster(above_pts, eps=eps, min_points=20)

    return table_pts, non_table_pts, above_pts, dist


# ============================================================
# 相似变换估计：single duck -> joint duck candidate
# ============================================================

def pca_frame(points):
    """
    返回 PCA 主轴矩阵，列向量为主轴。
    """
    points = np.asarray(points, dtype=np.float64)
    c = points.mean(axis=0)
    X = points - c
    C = X.T @ X / max(len(X), 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    F = eigvecs[:, order]

    # 保证右手系
    if np.linalg.det(F) < 0:
        F[:, 2] *= -1.0

    return F, c, eigvals[order]


def umeyama_similarity(src, dst, with_scale=True):
    """
    求 p_dst ≈ s * R * p_src + t

    src, dst: Nx3 对应点
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    assert src.shape == dst.shape
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)

    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1.0

    R = U @ D @ Vt

    if with_scale:
        var_src = np.sum(X ** 2) / n
        s = np.trace(np.diag(S) @ D) / max(var_src, 1e-12)
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)

    return float(s), R, t


def apply_similarity(points, s, R, t):
    points = np.asarray(points, dtype=np.float64)
    return float(s) * (points @ R.T) + np.asarray(t, dtype=np.float64).reshape(1, 3)


def random_sample_pair(src, tgt, max_src=50000, max_tgt=50000, seed=0):
    return (
        downsample_points(src, max_src, seed=seed),
        downsample_points(tgt, max_tgt, seed=seed + 1),
    )


def trimmed_similarity_icp(
    src_points,
    tgt_points,
    init_s,
    init_R,
    init_t,
    iterations=40,
    trim_ratio=0.65,
    with_scale=True,
    max_corr_ratio=0.06,
    seed=0,
):
    """
    简单 trimmed ICP：
      - source 变换后找 target 最近邻
      - 只保留距离最小的 trim_ratio 对
      - 用 Umeyama 更新 similarity

    适合：
      single duck -> joint_duck_candidate 的粗标定。
    """
    src, tgt = random_sample_pair(src_points, tgt_points, seed=seed)

    diag_tgt = bbox_diag(tgt)
    max_corr_dist = float(max_corr_ratio * diag_tgt)

    tree = cKDTree(tgt)

    s = float(init_s)
    R = np.asarray(init_R, dtype=np.float64)
    t = np.asarray(init_t, dtype=np.float64)

    last_rmse = None

    for it in range(iterations):
        src_tf = apply_similarity(src, s, R, t)
        dists, idx = tree.query(src_tf, k=1, workers=-1)

        valid = np.isfinite(dists)
        if max_corr_dist > 0:
            valid = valid & (dists < max_corr_dist)

        if valid.sum() < 50:
            # 放宽：如果阈值太严，就只用最近的那部分
            valid = np.isfinite(dists)

        src_valid = src[valid]
        tgt_valid = tgt[idx[valid]]
        dist_valid = dists[valid]

        if len(src_valid) < 50:
            break

        keep_n = max(50, int(trim_ratio * len(src_valid)))
        order = np.argsort(dist_valid)
        keep = order[:keep_n]

        s_new, R_new, t_new = umeyama_similarity(
            src_valid[keep],
            tgt_valid[keep],
            with_scale=with_scale,
        )

        s, R, t = s_new, R_new, t_new

        src_tf_keep = apply_similarity(src_valid[keep], s, R, t)
        rmse = float(np.sqrt(np.mean(np.sum((src_tf_keep - tgt_valid[keep]) ** 2, axis=1))))

        if last_rmse is not None and abs(last_rmse - rmse) < 1e-7:
            break

        last_rmse = rmse

    # final score
    src_tf = apply_similarity(src, s, R, t)
    dists, idx = tree.query(src_tf, k=1, workers=-1)
    order = np.argsort(dists)
    keep_n = max(50, int(trim_ratio * len(order)))
    keep = order[:keep_n]

    rmse = float(np.sqrt(np.mean(dists[keep] ** 2)))
    coverage = float(np.mean(dists < max_corr_dist)) if max_corr_dist > 0 else 0.0

    return {
        "s": float(s),
        "R": R,
        "t": t,
        "rmse": rmse,
        "coverage": coverage,
        "max_corr_dist": max_corr_dist,
    }


def generate_pca_initial_rotations(src_points, tgt_points):
    """
    基于 PCA 生成多个初始旋转候选。

    R 方向定义：
      p_tgt ≈ R @ p_src

    使用所有 det=+1 的符号翻转，处理 PCA 轴方向歧义。
    """
    Fs, cs, es = pca_frame(src_points)
    Ft, ct, et = pca_frame(tgt_points)

    rotations = []

    sign_mats = []
    for signs in itertools.product([-1.0, 1.0], repeat=3):
        S = np.diag(signs)
        if np.linalg.det(S) > 0:
            sign_mats.append(S)

    for S in sign_mats:
        R = Ft @ S @ Fs.T
        if np.linalg.det(R) > 0:
            rotations.append(R)

    # 额外加一个单位旋转候选
    rotations.append(np.eye(3))

    # 去重
    uniq = []
    for R in rotations:
        duplicate = False
        for U in uniq:
            if np.linalg.norm(R - U) < 1e-6:
                duplicate = True
                break
        if not duplicate:
            uniq.append(R)

    return uniq


def rotation_angle_deg(R):
    R = np.asarray(R, dtype=np.float64)
    val = (np.trace(R) - 1.0) / 2.0
    val = float(np.clip(val, -1.0, 1.0))
    return float(np.degrees(np.arccos(val)))


def estimate_single_to_joint_similarity(
    single_points,
    joint_candidate_points,
    with_scale=True,
    seed=0,
):
    """
    估计：
      p_joint = s * R * p_single + t
    """
    src = downsample_points(single_points, max_points=80000, seed=seed)
    tgt = downsample_points(joint_candidate_points, max_points=80000, seed=seed + 10)

    if len(src) < 100 or len(tgt) < 100:
        raise RuntimeError("single 或 joint_candidate 点数太少，无法配准。")

    src_center = src.mean(axis=0)
    tgt_center = tgt.mean(axis=0)

    src_diag = bbox_diag(src)
    tgt_diag = bbox_diag(tgt)

    init_scale = tgt_diag / max(src_diag, 1e-12) if with_scale else 1.0

    rotations = generate_pca_initial_rotations(src, tgt)

    print("\n" + "=" * 80)
    print("[Register] single duck -> joint duck candidate")
    print(f"src points={len(src)}, tgt points={len(tgt)}")
    print(f"src_diag={src_diag:.6f}, tgt_diag={tgt_diag:.6f}, init_scale={init_scale:.6f}")
    print(f"PCA rotation candidates={len(rotations)}")
    print("=" * 80)

    best = None

    for i, R0 in enumerate(rotations):
        t0 = tgt_center - init_scale * (R0 @ src_center)

        res = trimmed_similarity_icp(
            src,
            tgt,
            init_s=init_scale,
            init_R=R0,
            init_t=t0,
            iterations=50,
            trim_ratio=0.65,
            with_scale=with_scale,
            max_corr_ratio=0.08,
            seed=seed + i * 17,
        )

        print(
            f"[Candidate {i:02d}] "
            f"rmse={res['rmse']:.6f}, "
            f"coverage={res['coverage']:.4f}, "
            f"scale={res['s']:.6f}, "
            f"rot_angle_from_I={rotation_angle_deg(res['R']):.2f} deg"
        )

        if best is None or res["rmse"] < best["rmse"]:
            best = res

    print("\n[Register] best:")
    print(f"  rmse       = {best['rmse']:.6f}")
    print(f"  coverage   = {best['coverage']:.4f}")
    print(f"  scale      = {best['s']:.6f}")
    print(f"  R angle    = {rotation_angle_deg(best['R']):.2f} deg")

    return best


# ============================================================
# 可视化合并
# ============================================================

def save_compare_pcd(single_aligned, joint_candidate, table_points, path):
    """
    合并可视化：
      蓝色：single_aligned
      红色：joint duck candidate
      绿色：table points
    """
    p1 = np_to_pcd(single_aligned, color=[0.1, 0.35, 1.0])
    p2 = np_to_pcd(joint_candidate, color=[1.0, 0.15, 0.1])
    p3 = np_to_pcd(table_points, color=[0.1, 0.9, 0.2])

    combined = p1 + p2 + p3
    ok = o3d.io.write_point_cloud(path, combined)
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--joint",
        required=True,
        help="联合生成的 SAM3D 点云/mesh，例如 sam3d_duck_table_joint_clean.ply"
    )
    parser.add_argument(
        "--single",
        default="",
        help="单独生成的鸭子点云/mesh，例如 sam3d_duck_clean.ply。若不提供，则只做桌面提取。"
    )
    parser.add_argument(
        "--out-dir",
        default="./output_joint_table_extract",
        help="输出目录"
    )

    parser.add_argument("--sample-points", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=0)

    # 平面拟合参数
    parser.add_argument("--ransac-dist-ratio", type=float, default=0.006)
    parser.add_argument("--max-planes", type=int, default=6)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.04)

    # 分离参数
    parser.add_argument("--strict-ratio", type=float, default=0.006)
    parser.add_argument("--remove-ratio", type=float, default=0.018)
    parser.add_argument("--above-ratio", type=float, default=0.035)

    parser.add_argument("--no-largest-cluster", action="store_true")
    parser.add_argument("--cluster-eps-ratio", type=float, default=0.035)

    # 配准参数
    parser.add_argument("--no-scale", action="store_true", help="配准 single->joint 时禁用尺度估计")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print("=" * 80)
    print("[Start] 从 SAM3D 联合点云中提取桌面，并迁移桌面法向")
    print("=" * 80)

    # ------------------------------------------------------------
    # 1. 读取 joint
    # ------------------------------------------------------------
    joint_pcd = geometry_to_pcd(args.joint, sample_points=args.sample_points)
    joint_points = get_points(joint_pcd)

    # ------------------------------------------------------------
    # 2. 多平面 RANSAC，选桌面
    # ------------------------------------------------------------
    best_plane, all_planes, ransac_dist_thresh, diag = segment_table_plane_multi(
        joint_points,
        ransac_dist_ratio=args.ransac_dist_ratio,
        max_planes=args.max_planes,
        min_inlier_ratio=args.min_inlier_ratio,
        seed=args.seed,
    )

    n_joint = np.asarray(best_plane["normal"], dtype=np.float64)
    d_joint = float(best_plane["d"])

    n_joint, d_joint, signed_dist = orient_table_normal_to_duck_side(
        joint_points,
        n_joint,
        d_joint,
    )

    # ------------------------------------------------------------
    # 3. 分离桌面和鸭子候选
    # ------------------------------------------------------------
    table_pts, non_table_pts, above_duck_pts, signed_dist = separate_table_and_duck(
        joint_points,
        n_joint,
        d_joint,
        diag=diag,
        strict_ratio=args.strict_ratio,
        remove_ratio=args.remove_ratio,
        above_ratio=args.above_ratio,
        use_largest_cluster=(not args.no_largest_cluster),
        cluster_eps_ratio=args.cluster_eps_ratio,
    )

    joint_table_path = os.path.join(args.out_dir, "joint_table_points.ply")
    joint_non_table_path = os.path.join(args.out_dir, "joint_non_table_candidate.ply")
    joint_above_path = os.path.join(args.out_dir, "joint_above_table_duck_candidate.ply")
    joint_plane_json = os.path.join(args.out_dir, "joint_table_plane.json")

    save_pcd(table_pts, joint_table_path, color=[0.1, 0.9, 0.2])
    save_pcd(non_table_pts, joint_non_table_path, color=[1.0, 0.3, 0.1])
    save_pcd(above_duck_pts, joint_above_path, color=[1.0, 0.15, 0.1])

    plane_info = {
        "source_joint": os.path.abspath(args.joint),
        "plane_form": "normal dot x + d = 0",
        "normal_joint": n_joint.tolist(),
        "d_joint": float(d_joint),
        "bbox_diag": float(diag),
        "ransac_dist_thresh": float(ransac_dist_thresh),
        "strict_ratio": float(args.strict_ratio),
        "remove_ratio": float(args.remove_ratio),
        "above_ratio": float(args.above_ratio),
        "num_points_joint": int(len(joint_points)),
        "num_points_table": int(len(table_pts)),
        "num_points_non_table": int(len(non_table_pts)),
        "num_points_above_duck": int(len(above_duck_pts)),
        "best_plane_raw": {
            "k": int(best_plane["k"]),
            "normal": best_plane["normal"].tolist(),
            "d": float(best_plane["d"]),
            "inlier_count": int(best_plane["inlier_count"]),
            "area": float(best_plane["area"]),
            "extents": best_plane["extents"],
            "score": float(best_plane["score"]),
        },
        "all_plane_candidates": [
            {
                "k": int(p["k"]),
                "normal": p["normal"].tolist(),
                "d": float(p["d"]),
                "inlier_count": int(p["inlier_count"]),
                "area": float(p["area"]),
                "extents": p["extents"],
                "score": float(p["score"]),
            }
            for p in all_planes
        ],
    }

    with open(joint_plane_json, "w", encoding="utf-8") as f:
        json.dump(plane_info, f, indent=2, ensure_ascii=False)

    print(f"[Save] {joint_plane_json}")

    # ------------------------------------------------------------
    # 4. 如果提供 single，则估计 single -> joint
    # ------------------------------------------------------------
    if args.single.strip() == "":
        print("\n[Done] 未提供 --single，只完成联合点云桌面提取。")
        return

    single_pcd = geometry_to_pcd(args.single, sample_points=args.sample_points)
    single_points = get_points(single_pcd)

    if len(above_duck_pts) < 300:
        print("\n[WARN] above_duck_candidate 点数很少，配准可能不可靠。")
        print("可以尝试减小 --above-ratio，例如 0.025，或者加 --no-largest-cluster。")

    reg = estimate_single_to_joint_similarity(
        single_points,
        above_duck_pts,
        with_scale=(not args.no_scale),
        seed=args.seed,
    )

    s = float(reg["s"])
    R = np.asarray(reg["R"], dtype=np.float64)
    t = np.asarray(reg["t"], dtype=np.float64)

    # ------------------------------------------------------------
    # 5. 转换桌面法向到 single duck 坐标系
    # ------------------------------------------------------------
    n_single = normalize_vec(R.T @ n_joint)

    # joint 平面：n_j · x_j + d_j = 0
    # x_j = s R x_s + t
    # => n_j · (s R x_s + t) + d_j = 0
    # => (R.T n_j) · x_s + (n_j·t + d_j)/s = 0
    d_single = float((np.dot(n_joint, t) + d_joint) / max(s, 1e-12))

    single_aligned = apply_similarity(single_points, s, R, t)

    single_aligned_path = os.path.join(args.out_dir, "single_aligned_to_joint_candidate.ply")
    compare_path = os.path.join(args.out_dir, "compare_single_joint_candidate.ply")
    transform_json = os.path.join(args.out_dir, "single_to_joint_transform.json")
    normal_single_json = os.path.join(args.out_dir, "table_normal_in_single_duck_frame.json")

    save_pcd(single_aligned, single_aligned_path, color=[0.1, 0.35, 1.0])
    save_compare_pcd(single_aligned, above_duck_pts, table_pts, compare_path)

    transform_info = {
        "definition": "p_joint = scale * R @ p_single + t",
        "source_single": os.path.abspath(args.single),
        "source_joint": os.path.abspath(args.joint),
        "scale": s,
        "R_single_to_joint": R.tolist(),
        "t_single_to_joint": t.tolist(),
        "rmse": float(reg["rmse"]),
        "coverage": float(reg["coverage"]),
        "max_corr_dist": float(reg["max_corr_dist"]),
        "rotation_angle_from_identity_deg": rotation_angle_deg(R),
    }

    with open(transform_json, "w", encoding="utf-8") as f:
        json.dump(transform_info, f, indent=2, ensure_ascii=False)

    normal_info = {
        "explanation": "This is the joint table plane converted into the coordinate frame of the single SAM3D duck.",
        "plane_form": "normal_single dot x_single + d_single = 0",
        "normal_table_joint": n_joint.tolist(),
        "d_table_joint": float(d_joint),
        "normal_table_single": n_single.tolist(),
        "d_table_single": float(d_single),
        "transform_used": "p_joint = scale * R @ p_single + t",
        "scale": s,
        "R_single_to_joint": R.tolist(),
        "t_single_to_joint": t.tolist(),
        "registration_rmse": float(reg["rmse"]),
        "registration_coverage": float(reg["coverage"]),
        "warning": (
            "If joint_above_table_duck_candidate.ply is too incomplete or compare_single_joint_candidate.ply is misaligned, "
            "do not trust this converted normal."
        ),
    }

    with open(normal_single_json, "w", encoding="utf-8") as f:
        json.dump(normal_info, f, indent=2, ensure_ascii=False)

    print("\n" + "█" * 80)
    print("✅ 完成")
    print("重点检查这些文件：")
    print(f"  1. 联合模型桌面点        : {os.path.abspath(joint_table_path)}")
    print(f"  2. 去桌面后的候选        : {os.path.abspath(joint_non_table_path)}")
    print(f"  3. 桌面上方鸭子候选      : {os.path.abspath(joint_above_path)}")
    print(f"  4. 桌面平面参数          : {os.path.abspath(joint_plane_json)}")
    print(f"  5. single 对齐到 joint   : {os.path.abspath(single_aligned_path)}")
    print(f"  6. 三者对比可视化        : {os.path.abspath(compare_path)}")
    print(f"  7. single->joint 变换    : {os.path.abspath(transform_json)}")
    print(f"  8. single 坐标系桌面法向 : {os.path.abspath(normal_single_json)}")
    print("█" * 80)

    print("\n颜色说明 compare_single_joint_candidate.ply：")
    print("  蓝色 = 单独生成鸭子对齐后")
    print("  红色 = 联合模型里去桌面后的鸭子候选")
    print("  绿色 = 联合模型里拟合出来的桌面点")

    print("\n判断标准：")
    print("  如果蓝色和红色鸭子大致重合，则 normal_table_single 可以继续用于后续桌面约束配准。")
    print("  如果蓝色和红色明显错位，说明 joint 里的鸭子候选太差，不能信任这个迁移出来的桌面法向。")


if __name__ == "__main__":
    main()