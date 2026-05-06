#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import open3d as o3d


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def np_to_pcd(points, color=None):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None and len(points) > 0:
        colors = np.repeat(np.asarray(color, dtype=np.float64).reshape(1, 3), len(points), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_pcd(points, path, color=None):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    ok = o3d.io.write_point_cloud(path, np_to_pcd(points, color))
    if not ok:
        raise RuntimeError(f"写出失败: {path}")
    print(f"[Save] {path}  points={len(points)}")


def load_points(path, sample_points=180000):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        print(f"[Load mesh] {path} vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=sample_points)
        return np.asarray(pcd.points, dtype=np.float64)

    pcd = o3d.io.read_point_cloud(path)
    if pcd is not None and len(pcd.points) > 0:
        print(f"[Load pcd] {path} points={len(pcd.points)}")
        return np.asarray(pcd.points, dtype=np.float64)

    raise RuntimeError(f"无法读取: {path}")


def robust_bbox_diag(points):
    lo = np.percentile(points, 2, axis=0)
    hi = np.percentile(points, 98, axis=0)
    return float(np.linalg.norm(hi - lo))


def fit_plane(points, dist, iters):
    pcd = np_to_pcd(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist,
        ransac_n=3,
        num_iterations=iters,
    )
    a, b, c, d = [float(x) for x in plane_model]
    norm = np.linalg.norm([a, b, c])
    n = np.asarray([a, b, c], dtype=np.float64) / norm
    d = d / norm
    inliers = np.asarray(inliers, dtype=np.int64)
    return n, d, inliers


def orient_plane(points, n, d, inliers):
    mask = np.ones(len(points), dtype=bool)
    mask[inliers] = False
    non = points[mask]
    if len(non) > 100:
        h = non @ n + d
        if np.median(h) < 0:
            n = -n
            d = -d
            print("[Orient] flip normal so object side is positive")
    return n, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-points", type=int, default=180000)
    parser.add_argument("--table-dist", type=float, default=-1.0)
    parser.add_argument("--ransac-iters", type=int, default=3000)

    # 多组阈值一次性输出，方便肉眼比较
    parser.add_argument("--above-list", default="6,8,10,12,15,20,25,30")
    parser.add_argument("--table-shell-mult", type=float, default=1.8)

    # DBSCAN 去平面大簇，可选
    parser.add_argument("--dbscan", action="store_true")
    parser.add_argument("--eps-ratio", type=float, default=0.018)
    parser.add_argument("--min-points", type=int, default=30)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    points = load_points(args.joint, args.sample_points)
    diag = robust_bbox_diag(points)

    table_dist = args.table_dist
    if table_dist <= 0:
        table_dist = max(0.004 * diag, 0.003)

    print("=" * 80)
    print("[Joint duck candidate extraction debug]")
    print(f"points={len(points)}")
    print(f"robust diag={diag:.6f}")
    print(f"table_dist={table_dist:.6f}")
    print("=" * 80)

    n, d, inliers = fit_plane(points, table_dist, args.ransac_iters)
    n, d = orient_plane(points, n, d, inliers)

    h = points @ n + d
    print(f"[Plane] n={n.tolist()}, d={d:.8f}")
    for q in [0, 1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98, 99, 100]:
        print(f"  h p{q:03d} = {np.percentile(h, q):.6f}")

    table_mask = np.abs(h) < table_dist * args.table_shell_mult
    table_pts = points[table_mask]
    save_pcd(table_pts, os.path.join(args.out_dir, "joint_table_points.ply"), [0.1, 0.9, 0.2])

    above_vals = [float(x.strip()) for x in args.above_list.split(",") if x.strip()]

    summary = {
        "joint": os.path.abspath(args.joint),
        "table_dist": table_dist,
        "normal": n.tolist(),
        "d": float(d),
        "height_percentiles": {str(q): float(np.percentile(h, q)) for q in [0,1,5,10,20,30,50,70,80,90,95,98,99,100]},
        "candidates": []
    }

    for mult in above_vals:
        th = table_dist * mult
        cand = points[h > th]

        name = f"joint_duck_h_above_mult_{int(mult) if mult.is_integer() else mult}.ply"
        save_pcd(cand, os.path.join(args.out_dir, name), [1.0, 0.05, 0.02])

        print(f"[Candidate] mult={mult}, th={th:.6f}, points={len(cand)}")

        if args.dbscan and len(cand) > 0:
            eps = diag * args.eps_ratio
            pcd = np_to_pcd(cand)
            labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=args.min_points, print_progress=False))
            valid = labels >= 0
            if np.any(valid):
                ids, counts = np.unique(labels[valid], return_counts=True)
                order = np.argsort(-counts)
                # 保存前 3 个簇，通常最大簇可能是桌面，第二/第三才可能是鸭子，方便看
                for rank, idx in enumerate(order[:3]):
                    lab = ids[idx]
                    cluster = cand[labels == lab]
                    cname = f"joint_duck_h_above_mult_{int(mult) if mult.is_integer() else mult}_cluster{rank}_n{len(cluster)}.ply"
                    save_pcd(cluster, os.path.join(args.out_dir, cname), [1.0, 0.2 + 0.2 * rank, 0.02])
            else:
                print(f"[DBSCAN] mult={mult}: no cluster")

        summary["candidates"].append({
            "mult": mult,
            "threshold": float(th),
            "points": int(len(cand)),
            "file": name,
        })

    with open(os.path.join(args.out_dir, "joint_extract_debug.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n✅ 完成。先逐个打开 joint_duck_h_above_mult_*.ply，看哪个最像纯鸭子。")


if __name__ == "__main__":
    main()
