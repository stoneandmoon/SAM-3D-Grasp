#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

p = Path("yazishuzhizhousuanfa_feature.py")
text = p.read_text(encoding="utf-8")

# 1. 增加 --joint-duck-override 参数
old = 'parser.add_argument("--joint", required=True)'
new = '''parser.add_argument("--joint", required=True)
    parser.add_argument("--joint-duck-override", default="", help="可选：直接指定干净的 joint duck candidate ply，跳过自动 h>above_thresh 提取")'''

if new not in text:
    if old not in text:
        raise RuntimeError("找不到 --joint 参数插入位置")
    text = text.replace(old, new)
    print("[Patch] added --joint-duck-override")
else:
    print("[Skip] --joint-duck-override already exists")


# 2. 替换 joint_duck_for_align 的生成逻辑
old = '''    n_joint, d_joint, table_points, non_table, joint_duck, table_dist = fit_joint_table(joint, args)
    joint_duck_for_align = filter_joint_duck(
        joint_duck,
        n_joint,
        d_joint,
        percentile=args.joint_bottom_filter_percentile,
    )'''

new = '''    n_joint, d_joint, table_points, non_table, joint_duck, table_dist = fit_joint_table(joint, args)

    if args.joint_duck_override and args.joint_duck_override.strip():
        print("\\n[Override] 使用手动指定的 clean joint duck candidate:")
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
        )'''

if old not in text:
    print("[WARN] 没找到 joint_duck_for_align 原始片段，可能已经改过，请手动检查。")
else:
    text = text.replace(old, new)
    print("[Patch] added joint_duck_override logic")


# 3. 在 ICP 前加入跳过逻辑
marker = '''    # ICP 微调，仍然在归一化空间
    src_pcd = np_to_pcd(src_norm)'''

insert = '''    if args.icp_iters <= 0:
        print("\\n[ICP] skipped because --icp-iters <= 0")
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
    src_pcd = np_to_pcd(src_norm)'''

if insert not in text:
    if marker not in text:
        print("[WARN] 没找到 ICP marker，请手动检查。")
    else:
        text = text.replace(marker, insert)
        print("[Patch] added ICP skip logic")
else:
    print("[Skip] ICP skip logic already exists")

p.write_text(text, encoding="utf-8")

print("\\n✅ patch complete")
print("Run:")
print("  python -m py_compile yazishuzhizhousuanfa_feature.py")
