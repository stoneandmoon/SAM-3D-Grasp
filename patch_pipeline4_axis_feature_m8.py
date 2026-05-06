#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
patch_pipeline4_axis_feature_m8.py

作用：
  修改 pipeline4.py 的 Step 8：
    旧版：
      yazishuzhizhousuanfa.py

    新版：
      yazishuzhizhousuanfa_feature.py
      --above-mult 8
      --joint-bottom-filter-percentile 0
      --voxel-size 0.035
      --icp-iters 0

原因：
  你已经验证 joint_duck_h_above_mult_8.ply 是正确的 joint duck candidate。
  旧版 Step 8 默认阈值会把桌面混进 joint duck，导致竖直轴迁移错误。
"""

from pathlib import Path
import re


p = Path("pipeline4.py")

if not p.exists():
    raise FileNotFoundError("找不到 pipeline4.py，请在 /root/SAM-3D-Grasp 下运行。")

bak = Path("pipeline4.py.bak_axis_feature_m8")
if not bak.exists():
    bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[Backup] pipeline4.py -> {bak}")
else:
    print(f"[Backup] 已存在，不重复备份: {bak}")

text = p.read_text(encoding="utf-8")

old_block = '''        run_cmd([
            sys.executable, str(resolve_path("yazishuzhizhousuanfa.py", must_exist=True)),
            "--single", str(single_sam3d_ply),
            "--joint", str(joint_ply),
            "--out-dir", str(dirs["06_axis_transfer"]),
        ])'''

new_block = '''        run_cmd([
            sys.executable, str(resolve_path("yazishuzhizhousuanfa_feature.py", must_exist=True)),
            "--single", str(single_sam3d_ply),
            "--joint", str(joint_ply),
            "--out-dir", str(dirs["06_axis_transfer"]),

            # 关键修复：
            # 你已经验证 above-mult=8 提取出来的 joint duck candidate 是正确的。
            # 默认阈值太低会把桌面混进 joint duck，导致 single->joint 配准和竖直轴迁移错误。
            "--above-mult", "8",

            # 这个场景下 mult=8 的候选已经比较干净，不再额外裁掉底部。
            "--joint-bottom-filter-percentile", "0",

            # FPFH 特征尺度，已验证可用。
            "--voxel-size", "0.035",

            # 暂时跳过 ICP，避免 Open3D ICP/法向阶段 segfault；
            # 这里主要需要正确旋转来迁移桌面法向，RANSAC 特征配准已经够用。
            "--icp-iters", "0",
        ])'''

if old_block in text:
    text = text.replace(old_block, new_block)
    print("[Patch] 已替换 Step 8 为 yazishuzhizhousuanfa_feature.py + above-mult=8")
else:
    # 如果你之前已经手动改过脚本名，这里做更宽松的修复
    if "yazishuzhizhousuanfa_feature.py" in text:
        print("[Info] pipeline4.py 里已经在使用 yazishuzhizhousuanfa_feature.py。检查参数是否存在。")

        # 找到 feature 调用块，如果缺参数则补在 --out-dir 后
        if '"--above-mult", "8"' not in text:
            text = text.replace(
                '"--out-dir", str(dirs["06_axis_transfer"]),',
                '"--out-dir", str(dirs["06_axis_transfer"]),\n'
                '            "--above-mult", "8",\n'
                '            "--joint-bottom-filter-percentile", "0",\n'
                '            "--voxel-size", "0.035",\n'
                '            "--icp-iters", "0",'
            )
            print("[Patch] 已给 feature 调用补充 above-mult=8 等参数。")
        else:
            print("[Skip] above-mult=8 参数已存在。")
    else:
        raise RuntimeError(
            "没有找到原始 Step 8 调用块。请运行：grep -n \"yazishuzhizhousuanfa\" pipeline4.py"
        )

p.write_text(text, encoding="utf-8")

print("\n✅ pipeline4.py 修改完成。")
print("建议检查：")
print("  grep -n \"yazishuzhizhousuanfa\" pipeline4.py")
print("  grep -n \"above-mult\" pipeline4.py")
print("  python -m py_compile pipeline4.py")
