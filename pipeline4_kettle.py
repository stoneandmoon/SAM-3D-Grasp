#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline4.py

目标：
  将旧 pipeline 中的真实 depth partial + step_restore_rgb_pose_partial_locked.py
  替换为：

    1. SAM3 分割目标
    2. SAM3D 生成 single object
    3. 真实 RGB-D 拟合桌面
    4. 真实 RGB-D 提取 raw camera partial
    5. 生成 object + local table joint mask
    6. 使用同一个 SAM3D engine 继续生成 joint object+table
    7. joint 桌面轴迁移到 single SAM3D 坐标系
    8. 桌面高度估计 scale
    9. 桌面轴约束初始配准
    10. visible-shell 精配准
    11. 导出纯净 SAM3D 点云和 RGB-D 坐标系位姿

最终输出：
  output_pipeline4/final/sam3d_pure_in_rgbd.ply
  output_pipeline4/final/sam3d_pose_in_rgbd.json

最终变换定义：
  p_rgbd = scale * R_sam_to_rgbd @ p_sam + t_sam_to_rgbd

抓取使用：
  center_rgbd = scale * R_sam_to_rgbd @ center_sam + t_sam_to_rgbd
  R_grasp_rgbd = R_sam_to_rgbd @ R_grasp_sam
  width_rgbd = scale * width_sam

重要：
  modules/sam3d_engine.py 里的 SAM3DEngine.generate_3d 必须支持 keep_loaded 参数：
    generate_3d(image_path, mask, keep_loaded=True/False)
"""

import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path

# =========================================================
# 环境护盾
# =========================================================
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ.setdefault("TORCH_HOME", "/root/.cache/torch")

REPO_ROOT = Path(__file__).resolve().parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from modules.asr_sensevoice import SenseVoiceASR
from modules.slm_parser import SemanticParser

# 推荐你新建了 modules/sam3_segment_pipeline4.py 就用新版本；
# 如果没有，也会退回原来的 modules.sam3_segment。
try:
    from modules.sam3_segment_pipeline4 import SAM3Segmenter
except Exception:
    from modules.sam3_segment import SAM3Segmenter

from modules.sam3d_engine_v2 import SAM3DEngine


# =========================================================
# 路径工具
# =========================================================

def resolve_path(path_like, must_exist=False):
    """
    兼容：
      SAM-3D-Grasp/data/...
      data/...
      ./data/...
      /root/SAM-3D-Grasp/data/...
    """
    p = Path(path_like)

    if p.is_absolute():
        q = p
    else:
        raw = str(p).replace("\\", "/")
        repo_prefix = REPO_ROOT.name + "/"

        if raw.startswith(repo_prefix):
            q = REPO_ROOT / raw[len(repo_prefix):]
        else:
            q = REPO_ROOT / p

    if must_exist and not q.exists():
        raise FileNotFoundError(f"找不到文件: {q}")

    return q


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None, capture=False):
    print("\n" + "=" * 100)
    print("[RUN]")
    print(" ".join(str(x) for x in cmd))
    print("=" * 100)

    if capture:
        result = subprocess.run(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"命令失败，returncode={result.returncode}")
        return result.stdout

    result = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"命令失败，returncode={result.returncode}")

    return ""


def parse_estimated_scale(text):
    vals = re.findall(r"estimated_scale\s*=\s*([0-9.]+)", text)

    if not vals:
        vals = re.findall(r"--fixed-scale\s+([0-9.]+)", text)

    if not vals:
        raise RuntimeError("无法从 chidujiaozhun-height.py 输出中解析 estimated_scale")

    return float(vals[0])


# =========================================================
# Pipeline4
# =========================================================

class Pipeline4TableAxisVisibleShell:
    def __init__(self, ds_key, force_target_en=None):
        self.asr = SenseVoiceASR()
        self.slm = SemanticParser(api_key=ds_key)
        self.sam = SAM3Segmenter()
        self.sam3d = SAM3DEngine()
        self.force_target_en = force_target_en

    def run(
        self,
        audio_file,
        image_file,
        depth_file,

        fx,
        fy,
        cx,
        cy,
        width=640,
        height=480,

        out_root="./output_pipeline4_kettle",

        # 如果已经有 single SAM3D 点云，可以填路径跳过 single 生成
        reuse_single_sam3d_ply=None,

        # 如果已经有 joint SAM3D 点云，可以填路径跳过 joint 生成
        reuse_joint_ply=None,

        # table plane 参数
        table_bbox_pad=120,
        table_ransac_dist=0.015,

        # joint mask 参数
        joint_bbox_pad=120,
        joint_plane_thresh=0.018,
        joint_max_table_ratio=3.0,

        # scale
        fixed_scale=-1.0,
        height_percentile=98.0,

        # init align
        yaw_step_deg=5.0,
        fine_yaw_step_deg=1.0,

        # visible shell
        visible_scale_min=-1.0,
        visible_scale_max=-1.0,
        visible_scale_steps=5,
        visible_yaw_range_deg=25.0,
        visible_depth_margin=0.006,
        camera_z_mode="min",
    ):
        print("\n" + "█" * 80)
        print("🚀 Pipeline4：桌面轴约束 + visible-shell SAM3D→RGB-D 配准")
        print("█" * 80)

        # ---------------------------------------------------------
        # 路径解析
        # ---------------------------------------------------------
        audio_file = resolve_path(audio_file, must_exist=True)
        image_file = resolve_path(image_file, must_exist=True)
        depth_file = resolve_path(depth_file, must_exist=True)

        out_root = resolve_path(out_root)
        ensure_dir(out_root)

        dirs = {
            "00_mask": out_root / "00_mask",
            "01_sam3d_single": out_root / "01_sam3d_single",
            "02_real_table": out_root / "02_real_table",
            "03_real_partial_raw": out_root / "03_real_partial_raw",
            "04_joint_mask": out_root / "04_joint_mask",
            "05_joint_sam3d": out_root / "05_joint_sam3d",
            "06_axis_transfer": out_root / "06_axis_transfer",
            "07_init_align": out_root / "07_table_axis_init_align",
            "08_visible_shell": out_root / "08_visible_shell_align",
            "final": out_root / "final",
        }

        for d in dirs.values():
            ensure_dir(d)

        mask_png_path = dirs["00_mask"] / "target_mask.png"
        mask_npy_path = dirs["00_mask"] / "target_mask.npy"
        mask_vis_path = dirs["00_mask"] / "target_2d_result.jpg"

        single_sam3d_ply = dirs["01_sam3d_single"] / "sam3d_single_clean.ply"
        single_pose_json = dirs["01_sam3d_single"] / "sam3d_pose.json"

        table_plane_json = dirs["02_real_table"] / "table_plane.json"
        raw_partial_ply = dirs["03_real_partial_raw"] / "duck_partial_raw_camera.ply"

        joint_mask_path = dirs["04_joint_mask"] / "duck_plus_local_table_mask.png"
        joint_ply = dirs["05_joint_sam3d"] / "sam3d_object_table_joint_clean.ply"

        single_table_normal_json = dirs["06_axis_transfer"] / "table_normal_in_single_duck_frame.json"
        init_transform_json = dirs["07_init_align"] / "table_axis_constrained_transform.json"
        visible_transform_json = dirs["08_visible_shell"] / "visible_shell_transform.json"

        visible_aligned_ply = dirs["08_visible_shell"] / "aligned_sam3d_duck_visible_refined.ply"
        final_pure_ply = dirs["final"] / "sam3d_pure_in_rgbd.ply"
        final_pose_json = dirs["final"] / "sam3d_pose_in_rgbd.json"

        print(f"[Input] audio : {audio_file}")
        print(f"[Input] rgb   : {image_file}")
        print(f"[Input] depth : {depth_file}")
        print(f"[Output root] {out_root}")

        # ---------------------------------------------------------
        # Step 1: ASR
        # ---------------------------------------------------------
        print("\n[Step 1/10] 🎙️ 语音识别...")
        raw_speech = self.asr.transcribe(str(audio_file))

        if not raw_speech:
            print("❌ 语音识别失败，pipeline4 终止。")
            return

        print(f"[ASR] {raw_speech}")

        # ---------------------------------------------------------
        # Step 2: 语义解析
        # ---------------------------------------------------------
        print("\n[Step 2/10] 🧠 语义解析目标物体...")

        if self.force_target_en is not None:
            target_en = self.force_target_en
            print(f"[SLM] 已启用强制目标: {target_en}")
        else:
            target_en = self.slm.extract_target(raw_speech)

        if not target_en:
            print("❌ 语义解析失败，pipeline4 终止。")
            return

        print(f"[Target] {target_en}")

        # ---------------------------------------------------------
        # Step 3: SAM3 分割
        # ---------------------------------------------------------
        print(f"\n[Step 3/10] 👁️ SAM3 分割目标: {target_en}")
        mask = self.sam.segment_by_text(str(image_file), target_en)

        if mask is None:
            print(f"❌ 未找到目标: {target_en}")
            return

        mask = self._normalize_mask(mask)
        self._save_mask(mask, mask_png_path, mask_npy_path)
        self._visualize_pro(str(image_file), mask, target_en, str(mask_vis_path))

        print("\n[Memory] 卸载 SAM3 分割模型...")
        try:
            self.sam._unload_model()
        except Exception as e:
            print(f"⚠️ SAM 模型卸载失败，但继续执行: {e}")

        # ---------------------------------------------------------
        # Step 4: SAM3D 单独生成完整物体
        # ---------------------------------------------------------
        print("\n[Step 4/10] 🧊 SAM3D 单独生成完整目标点云...")

        if reuse_single_sam3d_ply is not None:
            src = resolve_path(reuse_single_sam3d_ply, must_exist=True)
            shutil.copyfile(str(src), str(single_sam3d_ply))
            print(f"[Reuse] 使用已有 single SAM3D: {src}")

            # 如果直接复用 single，但是后面还需要 joint，
            # 那么 Step 7 会单独用 self.sam3d 加载一次并生成 joint。
        else:
            keep_loaded_for_joint = (reuse_joint_ply is None)

            self._generate_sam3d_and_copy(
                image_file=image_file,
                mask_array=mask,
                dst_ply=single_sam3d_ply,
                dst_pose_json=single_pose_json,
                keep_loaded=keep_loaded_for_joint,
                tag="single",
            )

        if not single_sam3d_ply.exists():
            print(f"❌ single SAM3D 点云不存在: {single_sam3d_ply}")
            return

        # ---------------------------------------------------------
        # Step 5: 从真实 RGB-D 拟合桌面
        # ---------------------------------------------------------
        print("\n[Step 5/10] 🟩 真实 RGB-D 桌面平面拟合...")

        run_cmd([
            sys.executable, str(resolve_path("make_real_table_plane.py", must_exist=True)),
            "--rgb", str(image_file),
            "--depth", str(depth_file),
            "--duck-mask", str(mask_png_path),
            "--fx", str(fx),
            "--fy", str(fy),
            "--cx", str(cx),
            "--cy", str(cy),
            "--bbox-pad", str(table_bbox_pad),
            "--ransac-dist", str(table_ransac_dist),
            "--out-dir", str(dirs["02_real_table"]),
        ])

        if not table_plane_json.exists():
            print(f"❌ 真实桌面平面生成失败: {table_plane_json}")
            return

        # ---------------------------------------------------------
        # Step 6: 从真实 RGB-D 直接提取 raw camera partial
        # ---------------------------------------------------------
        print("\n[Step 6/10] 📸 提取 raw camera 真实残缺点云...")

        run_cmd([
            sys.executable, str(resolve_path("extract_real_duck_raw_camera.py", must_exist=True)),
            "--rgb", str(image_file),
            "--depth", str(depth_file),
            "--mask", str(mask_png_path),
            "--fx", str(fx),
            "--fy", str(fy),
            "--cx", str(cx),
            "--cy", str(cy),
            "--table-plane", str(table_plane_json),
            "--out-dir", str(dirs["03_real_partial_raw"]),
        ])

        if not raw_partial_ply.exists():
            print(f"❌ raw partial 生成失败: {raw_partial_ply}")
            return

        # ---------------------------------------------------------
        # Step 7: joint mask + SAM3D joint 生成
        # ---------------------------------------------------------
        print("\n[Step 7/10] 🧩 生成 joint mask 并生成 SAM3D 物体+桌面联合点云...")

        if reuse_joint_ply is not None:
            src = resolve_path(reuse_joint_ply, must_exist=True)
            shutil.copyfile(str(src), str(joint_ply))
            print(f"[Reuse] 使用已有 joint SAM3D: {src}")
        else:
            run_cmd([
                sys.executable, str(resolve_path("make_duck_table_joint_mask.py", must_exist=True)),
                "--rgb", str(image_file),
                "--depth", str(depth_file),
                "--duck-mask", str(mask_png_path),
                "--table-plane", str(table_plane_json),
                "--fx", str(fx),
                "--fy", str(fy),
                "--cx", str(cx),
                "--cy", str(cy),
                "--bbox-pad", str(joint_bbox_pad),
                "--plane-thresh", str(joint_plane_thresh),
                "--max-table-ratio", str(joint_max_table_ratio),
                "--no-bottom-prefer",
                "--out-dir", str(dirs["04_joint_mask"]),
            ])

            if not joint_mask_path.exists():
                print(f"❌ joint mask 生成失败: {joint_mask_path}")
                return

            joint_mask_img = cv2.imread(str(joint_mask_path), cv2.IMREAD_GRAYSCALE)
            if joint_mask_img is None:
                print(f"❌ 读取 joint mask 失败: {joint_mask_path}")
                return

            joint_mask = (joint_mask_img > 127).astype(np.uint8)

            # 关键改动：
            # 不再 subprocess 调 test_sam3d_joint_table.py；
            # 直接使用同一个 self.sam3d 在当前进程继续生成 joint。
            self._generate_sam3d_and_copy(
                image_file=image_file,
                mask_array=joint_mask,
                dst_ply=joint_ply,
                dst_pose_json=dirs["05_joint_sam3d"] / "sam3d_pose.json",
                keep_loaded=False,
                tag="joint",
            )

            try:
                shutil.copyfile(
                    str(joint_mask_path),
                    str(dirs["05_joint_sam3d"] / "input_joint_mask.png"),
                )
            except Exception as e:
                print(f"⚠️ 备份 joint mask 失败，但不影响继续执行: {e}")

        if not joint_ply.exists():
            print(f"❌ joint SAM3D 点云不存在: {joint_ply}")
            return

        # ---------------------------------------------------------
        # Step 8: joint 桌面轴迁移到 single SAM3D 坐标系
        # ---------------------------------------------------------
        print("\n[Step 8/10] 🧭 joint 桌面法向迁移到 single SAM3D 坐标系...")

        run_cmd([
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
        ])

        if not single_table_normal_json.exists():
            print(f"❌ single-table-normal 生成失败: {single_table_normal_json}")
            return

        # ---------------------------------------------------------
        # Step 9: 尺度估计 + 桌面轴约束初始配准
        # ---------------------------------------------------------
        print("\n[Step 9/10] 📏 高度尺度估计 + 桌面轴约束初始配准...")

        if fixed_scale > 0:
            scale = float(fixed_scale)
            print(f"[Scale] 使用手动 fixed scale = {scale:.8f}")
        else:
            scale_text = run_cmd([
                sys.executable, str(resolve_path("chidujiaozhun-height.py", must_exist=True)),
                "--sam3d-single", str(single_sam3d_ply),
                "--real-partial", str(raw_partial_ply),
                "--real-table-plane", str(table_plane_json),
                "--single-table-normal", str(single_table_normal_json),
                "--real-height-percentile", str(height_percentile),
            ], capture=True)

            scale = parse_estimated_scale(scale_text)
            print(f"[Scale] 自动估计 scale = {scale:.8f}")

        run_cmd([
            sys.executable, str(resolve_path("step_table_axis_constrained_align.py", must_exist=True)),
            "--sam3d-single", str(single_sam3d_ply),
            "--real-partial", str(raw_partial_ply),
            "--real-table-plane", str(table_plane_json),
            "--single-table-normal", str(single_table_normal_json),
            "--out-dir", str(dirs["07_init_align"]),
            "--fixed-scale", f"{scale:.8f}",
            "--scale-min-mult", "1.0",
            "--scale-max-mult", "1.0",
            "--scale-steps", "1",
            "--yaw-step-deg", str(yaw_step_deg),
            "--fine-yaw-step-deg", str(fine_yaw_step_deg),
            "--p2s-trim", "0.95",
            "--s2p-trim", "0.10",
            "--translation-refine-iters", "16",
        ])

        if not init_transform_json.exists():
            print(f"❌ 初始配准 transform 生成失败: {init_transform_json}")
            return

        # ---------------------------------------------------------
        # Step 10: visible-shell 精配准
        # ---------------------------------------------------------
        print("\n[Step 10/10] 👁️ visible-shell 精配准...")

        if visible_scale_min <= 0:
            visible_scale_min = scale * 0.98
        if visible_scale_max <= 0:
            visible_scale_max = scale * 1.02

        run_cmd([
            sys.executable, str(resolve_path("step_table_axis_visible_shell_align.py", must_exist=True)),
            "--sam3d-single", str(single_sam3d_ply),
            "--real-partial", str(raw_partial_ply),
            "--real-table-plane", str(table_plane_json),
            "--init-transform", str(init_transform_json),
            "--fx", str(fx),
            "--fy", str(fy),
            "--cx", str(cx),
            "--cy", str(cy),
            "--width", str(width),
            "--height", str(height),
            "--scale-min", f"{visible_scale_min:.8f}",
            "--scale-max", f"{visible_scale_max:.8f}",
            "--scale-steps", str(visible_scale_steps),
            "--yaw-range-deg", str(visible_yaw_range_deg),
            "--yaw-step-deg", str(yaw_step_deg),
            "--fine-yaw-step-deg", str(fine_yaw_step_deg),
            "--depth-margin", str(visible_depth_margin),
            "--camera-z-mode", str(camera_z_mode),
            "--out-dir", str(dirs["08_visible_shell"]),
        ])

        if not visible_transform_json.exists():
            print(f"❌ visible-shell transform 生成失败: {visible_transform_json}")
            return

        if not visible_aligned_ply.exists():
            print(f"❌ visible-shell 纯净点云生成失败: {visible_aligned_ply}")
            return

        # ---------------------------------------------------------
        # Final: 导出纯净 SAM3D 点云和抓取用 pose JSON
        # ---------------------------------------------------------
        print("\n[Final] 📦 导出纯净 SAM3D 点云与抓取用位姿...")

        shutil.copyfile(str(visible_aligned_ply), str(final_pure_ply))
        self._export_final_pose_json(
            transform_json=visible_transform_json,
            aligned_ply=final_pure_ply,
            out_json=final_pose_json,
        )

        print("\n" + "█" * 80)
        print("🎉 Pipeline4 已完成！")
        print(f"🖼️  2D 目标定位图: {mask_vis_path}")
        print(f"🎭 目标 mask: {mask_png_path}")
        print(f"🧊 single SAM3D 点云: {single_sam3d_ply}")
        print(f"🟩 真实桌面平面: {table_plane_json}")
        print(f"📸 raw partial: {raw_partial_ply}")
        print(f"🧩 joint SAM3D 点云: {joint_ply}")
        print(f"🧭 single 坐标系桌面法向: {single_table_normal_json}")
        print(f"📏 初始配准 transform: {init_transform_json}")
        print(f"👁️ visible-shell transform: {visible_transform_json}")
        print(f"✅ 最终纯净 SAM3D 点云 RGB-D 坐标系: {final_pure_ply}")
        print(f"✅ 最终抓取用位姿 JSON: {final_pose_json}")
        print("█" * 80)

        print("\n[抓取使用]")
        print("center_rgbd = scale * R_sam_to_rgbd @ center_sam + t_sam_to_rgbd")
        print("R_grasp_rgbd = R_sam_to_rgbd @ R_grasp_sam")
        print("width_rgbd = scale * width_sam")

    # =========================================================
    # SAM3D 生成工具
    # =========================================================
    def _generate_sam3d_and_copy(
        self,
        image_file,
        mask_array,
        dst_ply,
        dst_pose_json=None,
        keep_loaded=False,
        tag="sam3d",
    ):
        """
        使用当前进程的 self.sam3d 生成 3D，然后把 output_3d/reconstructed_mesh.ply
        复制到指定输出位置。

        关键：
          keep_loaded=True  时，要求 modules/sam3d_engine.py 支持 generate_3d(..., keep_loaded=True)
          keep_loaded=False 时，生成完释放 SAM3D 模型
        """
        image_file = Path(image_file)
        dst_ply = Path(dst_ply)

        default_sam3d_dir = resolve_path("./output_3d")
        default_sam3d_dir.mkdir(parents=True, exist_ok=True)

        default_mesh_ply = default_sam3d_dir / "reconstructed_mesh.ply"
        default_mesh_obj = default_sam3d_dir / "reconstructed_mesh.obj"
        default_pose_json = default_sam3d_dir / "sam3d_pose.json"

        self._safe_unlink(default_mesh_ply)
        self._safe_unlink(default_mesh_obj)
        self._safe_unlink(default_pose_json)

        print(f"[SAM3D {tag}] 开始生成，keep_loaded={keep_loaded} ...")

        try:
            success = self.sam3d.generate_3d(
                str(image_file),
                mask_array,
                keep_loaded=keep_loaded,
            )
        except TypeError:
            print("\n❌ 你的 modules/sam3d_engine.py 还不支持 keep_loaded 参数。")
            print("请先修改 SAM3DEngine.generate_3d：")
            print("  def generate_3d(self, image_path, mask, keep_loaded=False):")
            print("并把结尾的 self._unload_model() 改成：")
            print("  if not keep_loaded:")
            print("      self._unload_model()")
            raise

        if not success:
            raise RuntimeError(f"SAM3D {tag} 生成失败。")

        if not default_mesh_ply.exists():
            raise FileNotFoundError(f"SAM3D {tag} 未生成点云: {default_mesh_ply}")

        dst_ply.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(default_mesh_ply), str(dst_ply))
        print(f"[SAM3D {tag}] 点云已保存: {dst_ply}")

        if dst_pose_json is not None:
            dst_pose_json = Path(dst_pose_json)
            if default_pose_json.exists():
                dst_pose_json.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(default_pose_json), str(dst_pose_json))
                print(f"[SAM3D {tag}] pose json 已保存: {dst_pose_json}")
            else:
                print(f"⚠️ SAM3D {tag} 没有生成 sam3d_pose.json，继续执行。")

        # 备份 obj
        try:
            if default_mesh_obj.exists():
                backup_obj = dst_ply.parent / f"{dst_ply.stem}.obj"
                shutil.copyfile(str(default_mesh_obj), str(backup_obj))
                print(f"[SAM3D {tag}] obj 已备份: {backup_obj}")
        except Exception as e:
            print(f"⚠️ SAM3D {tag} obj 备份失败: {e}")

    # =========================================================
    # 常规工具函数
    # =========================================================
    def _normalize_mask(self, mask):
        mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = np.squeeze(mask)

        if mask.dtype != np.bool_:
            mask = mask > 0.5

        return mask.astype(np.uint8)

    def _save_mask(self, mask, png_path, npy_path):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(str(png_path), mask_uint8)
        np.save(str(npy_path), mask.astype(np.uint8))

        print(f"[Mask] PNG 已保存: {png_path}")
        print(f"[Mask] NPY 已保存: {npy_path}")

    def _visualize_pro(self, img_path, mask, label, save_name):
        img = cv2.imread(str(img_path))

        if img is None:
            print("❌ 无法读取图像，2D 可视化失败。")
            return

        mask_bool = mask > 0
        mask_uint8 = mask_bool.astype(np.uint8) * 255

        overlay = img.copy()
        overlay[mask_bool] = [0, 255, 0]
        result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 500:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
                y_text_top = max(0, y - 30)
                cv2.rectangle(result, (x, y_text_top), (x + w, y), (255, 255, 0), -1)
                cv2.putText(
                    result,
                    label.upper(),
                    (x + 5, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

        cv2.imwrite(str(save_name), result)
        print(f"[Pipeline4] 2D 定位结果图已保存: {save_name}")

    def _safe_unlink(self, path):
        try:
            path = Path(path)
            if path.exists():
                path.unlink()
                print(f"[Clean] 删除旧文件: {path}")
        except Exception as e:
            print(f"⚠️ 删除旧文件失败 {path}: {e}")

    def _export_final_pose_json(self, transform_json, aligned_ply, out_json):
        transform_json = Path(transform_json)
        aligned_ply = Path(aligned_ply)
        out_json = Path(out_json)

        with open(transform_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        scale = float(data["scale"])

        if "R_single_to_real" in data:
            R = np.asarray(data["R_single_to_real"], dtype=np.float64)
        elif "R_total_single_to_real" in data:
            R = np.asarray(data["R_total_single_to_real"], dtype=np.float64)
        else:
            raise RuntimeError("transform JSON 中找不到 R_single_to_real / R_total_single_to_real")

        if "t_single_to_real" in data:
            t = np.asarray(data["t_single_to_real"], dtype=np.float64).reshape(3)
        elif "t_total_single_to_real" in data:
            t = np.asarray(data["t_total_single_to_real"], dtype=np.float64).reshape(3)
        else:
            raise RuntimeError("transform JSON 中找不到 t_single_to_real / t_total_single_to_real")

        T_se3 = np.eye(4, dtype=np.float64)
        T_se3[:3, :3] = R
        T_se3[:3, 3] = t

        T_scaled = np.eye(4, dtype=np.float64)
        T_scaled[:3, :3] = scale * R
        T_scaled[:3, 3] = t

        out = {
            "definition": "p_rgbd = scale * R_sam_to_rgbd @ p_sam + t_sam_to_rgbd",
            "transform_type": "similarity_transform",

            "scale": scale,
            "R_sam_to_rgbd": R.tolist(),
            "t_sam_to_rgbd": t.tolist(),

            "T_rotation_translation_only": T_se3.tolist(),
            "T_scaled_for_points": T_scaled.tolist(),

            "pure_sam3d_pointcloud_rgbd": str(aligned_ply.resolve()),
            "source_transform_json": str(transform_json.resolve()),

            "for_grasp": {
                "center_rgbd": "scale * R_sam_to_rgbd @ center_sam + t_sam_to_rgbd",
                "rotation_rgbd": "R_sam_to_rgbd @ R_grasp_sam",
                "width_rgbd": "scale * width_sam",
                "warning": "机器人末端姿态旋转只用 R_sam_to_rgbd，不要把 scale 塞进旋转矩阵。scale 只用于点、距离、夹爪宽度。"
            },
        }

        out_json.parent.mkdir(parents=True, exist_ok=True)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"[Final Pose] 已保存: {out_json}")


# =========================================================
# 主程序
# =========================================================

if __name__ == "__main__":
    # =========================================================
    # 用户配置区
    # =========================================================

    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "<YOUR_DEEPSEEK_KEY>")

    TEST_AUDIO = "SAM-3D-Grasp/data/test_audio3.m4a"
    TEST_IMAGE = "SAM-3D-Grasp/data/test/000002/rgb/000000.png"
    TEST_DEPTH = "SAM-3D-Grasp/data/test/000002/depth/000000.png"

    # 如果语义解析容易错，比如识别成 bleach bottle，可以强制：
    # FORCE_TARGET_EN = "yellow duck"
    FORCE_TARGET_EN = "white watering can"

    FX = 607.0
    FY = 607.0
    CX = 320.0
    CY = 240.0
    WIDTH = 640
    HEIGHT = 480

    pipeline = Pipeline4TableAxisVisibleShell(
        ds_key=DEEPSEEK_KEY,
        force_target_en=FORCE_TARGET_EN,
    )

    pipeline.run(
        audio_file=TEST_AUDIO,
        image_file=TEST_IMAGE,
        depth_file=TEST_DEPTH,

        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        width=WIDTH,
        height=HEIGHT,

        out_root="./output_pipeline4_kettle",

        # 第一次完整运行：None
        # 如果你已经有 single SAM3D，可以填：
        # reuse_single_sam3d_ply=None,
        reuse_single_sam3d_ply=None,

        # 第一次完整运行：None
        # 如果你已经有 joint SAM3D，可以填：
        # reuse_joint_ply=None,
        reuse_joint_ply=None,

        fixed_scale=-1.0,
        height_percentile=98.0,

        yaw_step_deg=5.0,
        fine_yaw_step_deg=1.0,

        visible_scale_min=-1.0,
        visible_scale_max=-1.0,
        visible_scale_steps=5,
        visible_yaw_range_deg=25.0,
        visible_depth_margin=0.006,
        camera_z_mode="min",
    )