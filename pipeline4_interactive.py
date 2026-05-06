#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline4_interactive_clean.py

基于 pipeline4.py 修改的干净交互版。

改动：
  1. 去掉固定 TEST_AUDIO / TEST_IMAGE / TEST_DEPTH
  2. 支持交互输入 RGB 图、Depth 图、语音文件
  3. 支持交互输入 RGB 图、Depth 图、文字指令
  4. 去掉“强制英文目标”输入
  5. 去掉手动输入 fx / fy / cx / cy / width / height
  6. 自动读取相机内参：
       - 优先读取 intrinsics.json / camera_intrinsics.json
       - 找不到则使用默认焦距 + 图像中心点
  7. 默认 fixed_scale = -1.0，自动估计尺度
  8. 默认不复用 single / joint SAM3D 点云，每次完整运行

最终输出：
  output_pipeline4_interactive/run_xxx/final/sam3d_pure_in_rgbd.ply
  output_pipeline4_interactive/run_xxx/final/sam3d_pose_in_rgbd.json

最终变换定义：
  p_rgbd = scale * R_sam_to_rgbd @ p_sam + t_sam_to_rgbd

抓取使用：
  center_rgbd = scale * R_sam_to_rgbd @ center_sam + t_sam_to_rgbd
  R_grasp_rgbd = R_sam_to_rgbd @ R_grasp_sam
  width_rgbd = scale * width_sam
"""

import os
import sys
import re
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

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
    if path_like is None:
        raise ValueError("resolve_path 收到 None 路径")

    path_like = str(path_like).strip().strip('"').strip("'")

    if not path_like:
        raise ValueError("resolve_path 收到空路径")

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
# 自动相机内参
# =========================================================

def auto_get_intrinsics(
    image_file,
    depth_file,
    default_fx=607.0,
    default_fy=607.0,
):
    """
    自动获取相机内参。

    优先级：
      1. RGB 图同目录下的 intrinsics.json
      2. RGB 图同目录下的 camera_intrinsics.json
      3. Depth 图同目录下的 intrinsics.json
      4. Depth 图同目录下的 camera_intrinsics.json
      5. 仓库根目录 intrinsics.json
      6. 仓库根目录 camera_intrinsics.json
      7. 使用默认焦距 + 图像中心点自动补齐

    支持格式 1：
    {
      "fx": 607.0,
      "fy": 607.0,
      "cx": 320.0,
      "cy": 240.0,
      "width": 640,
      "height": 480
    }

    支持格式 2：
    {
      "K": [
        [607.0, 0.0, 320.0],
        [0.0, 607.0, 240.0],
        [0.0, 0.0, 1.0]
      ],
      "width": 640,
      "height": 480
    }
    """
    image_path = resolve_path(image_file, must_exist=True)
    depth_path = resolve_path(depth_file, must_exist=True)

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取 RGB 图像: {image_path}")

    height, width = img.shape[:2]

    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_img is not None:
        dh, dw = depth_img.shape[:2]
        if (dw, dh) != (width, height):
            print(
                f"⚠️ RGB 和 Depth 尺寸不一致："
                f"RGB=({width}, {height}), Depth=({dw}, {dh})"
            )
            print("⚠️ 后续点云反投影可能需要确认 RGB-D 是否已经对齐。")

    candidate_jsons = [
        image_path.parent / "intrinsics.json",
        image_path.parent / "camera_intrinsics.json",
        depth_path.parent / "intrinsics.json",
        depth_path.parent / "camera_intrinsics.json",
        REPO_ROOT / "intrinsics.json",
        REPO_ROOT / "camera_intrinsics.json",
    ]

    for jp in candidate_jsons:
        if jp.exists():
            print(f"[Auto K] 找到相机内参文件: {jp}")

            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "K" in data:
                K = np.asarray(data["K"], dtype=np.float64)
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                cx = float(K[0, 2])
                cy = float(K[1, 2])
            else:
                fx = float(data.get("fx", default_fx))
                fy = float(data.get("fy", default_fy))
                cx = float(data.get("cx", width / 2.0))
                cy = float(data.get("cy", height / 2.0))

            width = int(data.get("width", width))
            height = int(data.get("height", height))

            return fx, fy, cx, cy, width, height

    print("[Auto K] 未找到 intrinsics.json / camera_intrinsics.json")
    print("[Auto K] 使用默认焦距 + 图像尺寸自动补齐。")

    fx = float(default_fx)
    fy = float(default_fy)
    cx = float(width / 2.0)
    cy = float(height / 2.0)

    return fx, fy, cx, cy, int(width), int(height)


# =========================================================
# 交互式输入工具
# =========================================================

def ask_str(prompt, default=None, allow_empty=False):
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        return s.strip().strip('"').strip("'")

    while True:
        s = input(f"{prompt}: ").strip().strip('"').strip("'")
        if s or allow_empty:
            return s
        print("❌ 输入不能为空，请重新输入。")


# =========================================================
# Pipeline4
# =========================================================

class Pipeline4TableAxisVisibleShell:
    def __init__(self, ds_key):
        self.asr = None
        self.slm = SemanticParser(api_key=ds_key)
        self.sam = SAM3Segmenter()
        self.sam3d = SAM3DEngine()

    def _get_asr(self):
        if self.asr is None:
            print("[ASR] 正在初始化 SenseVoiceASR...")
            self.asr = SenseVoiceASR()
        return self.asr

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

        text_command=None,

        out_root="./output_pipeline4",

        reuse_single_sam3d_ply=None,
        reuse_joint_ply=None,

        table_bbox_pad=120,
        table_ransac_dist=0.015,

        joint_bbox_pad=120,
        joint_plane_thresh=0.018,
        joint_max_table_ratio=3.0,

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
    ):
        print("\n" + "█" * 80)
        print("🚀 Pipeline4：交互式 RGB-D + 自动相机内参 + visible-shell 配准")
        print("█" * 80)

        # ---------------------------------------------------------
        # 路径解析
        # ---------------------------------------------------------

        image_file = resolve_path(image_file, must_exist=True)
        depth_file = resolve_path(depth_file, must_exist=True)

        text_command = "" if text_command is None else str(text_command).strip()

        if text_command:
            audio_file = None
        else:
            if audio_file is None or str(audio_file).strip() == "":
                raise ValueError("未提供语音文件，也未提供文字指令 text_command。")
            audio_file = resolve_path(audio_file, must_exist=True)

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

        if audio_file is not None:
            print(f"[Input] audio : {audio_file}")
        else:
            print(f"[Input] text  : {text_command}")

        print(f"[Input] rgb   : {image_file}")
        print(f"[Input] depth : {depth_file}")
        print(f"[Input] K     : fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}")
        print(f"[Output root] {out_root}")

        # ---------------------------------------------------------
        # Step 1: ASR 或文字输入
        # ---------------------------------------------------------

        print("\n[Step 1/10] 🎙️ 语音 / 文字指令输入...")

        if text_command:
            raw_speech = text_command
            print(f"[TEXT] {raw_speech}")
        else:
            asr = self._get_asr()
            raw_speech = asr.transcribe(str(audio_file))

            if not raw_speech:
                print("❌ 语音识别失败，pipeline4 终止。")
                return

            print(f"[ASR] {raw_speech}")

        # ---------------------------------------------------------
        # Step 2: 语义解析
        # ---------------------------------------------------------

        print("\n[Step 2/10] 🧠 语义解析目标物体...")

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
            "--above-mult", "8",
            "--joint-bottom-filter-percentile", "0",
            "--voxel-size", "0.035",
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
            print("\n❌ 你的 modules/sam3d_engine_v2.py 还不支持 keep_loaded 参数。")
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
    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "<YOUR_DEEPSEEK_KEY>")

    DEFAULT_AUDIO = "SAM-3D-Grasp/data/test_audio1.m4a"
    DEFAULT_IMAGE = "SAM-3D-Grasp/data/test/000002/rgb/000000.png"
    DEFAULT_DEPTH = "SAM-3D-Grasp/data/test/000002/depth/000000.png"

    DEFAULT_FX = 607.0
    DEFAULT_FY = 607.0

    print("\n" + "█" * 80)
    print("交互式 Pipeline4：RGB-D + 语音/文字指令 + 自动相机内参")
    print("█" * 80)
    print(f"[Repo] {REPO_ROOT}")
    print("\n说明：")
    print("  1. 模式 1：输入 RGB 图 + Depth 图 + 语音文件")
    print("  2. 模式 2：输入 RGB 图 + Depth 图 + 文字指令")
    print("  3. 系统会自动读取图像尺寸和相机内参")
    print("  4. 优先读取 intrinsics.json / camera_intrinsics.json")
    print("  5. 没有内参文件时，使用默认焦距 + 图像中心点")
    print("  6. 不再询问强制英文目标，目标完全由语音/文字指令解析得到")
    print("█" * 80)

    while True:
        print("\n请选择运行模式：")
        print("1. RGB 图 + Depth 图 + 语音文件")
        print("2. RGB 图 + Depth 图 + 文字指令")
        print("q. 退出")

        mode = input("\n请输入模式: ").strip().lower()

        if mode in ["q", "quit", "exit"]:
            print("已退出交互式 pipeline。")
            break

        if mode not in ["1", "2"]:
            print("❌ 模式无效，请重新输入。")
            continue

        try:
            print("\n" + "-" * 80)
            print("请输入本次运行的 RGB-D 数据")
            print("-" * 80)

            image_file = ask_str("RGB 图片路径", DEFAULT_IMAGE)
            depth_file = ask_str("Depth 深度图路径", DEFAULT_DEPTH)

            audio_file = None
            text_command = None

            if mode == "1":
                audio_file = ask_str("语音文件路径", DEFAULT_AUDIO)
            else:
                text_command = ask_str("请输入文字抓取指令，例如：抓取黄色鸭子")

            print("\n" + "-" * 80)
            print("自动读取相机内参与图像尺寸")
            print("-" * 80)

            fx, fy, cx, cy, width, height = auto_get_intrinsics(
                image_file=image_file,
                depth_file=depth_file,
                default_fx=DEFAULT_FX,
                default_fy=DEFAULT_FY,
            )

            print(
                f"[Auto K] fx={fx}, fy={fy}, "
                f"cx={cx}, cy={cy}, width={width}, height={height}"
            )

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_out_root = f"./output_pipeline4_interactive/run_{run_id}"

            print("\n" + "-" * 80)
            print("输出设置")
            print("-" * 80)

            out_root = ask_str("输出目录 out_root", default_out_root)

            print("\n" + "█" * 80)
            print("本次交互式运行配置")
            print("█" * 80)
            print(f"RGB     : {image_file}")
            print(f"Depth   : {depth_file}")
            print(f"Audio   : {audio_file}")
            print(f"Text    : {text_command}")
            print(f"Auto K  : fx={fx}, fy={fy}, cx={cx}, cy={cy}, width={width}, height={height}")
            print(f"out_root: {out_root}")
            print("█" * 80)

            confirm = input("确认开始运行？输入 y 开始，其他键返回主菜单: ").strip().lower()
            if confirm != "y":
                print("已取消本次运行，返回主菜单。")
                continue

            pipeline = Pipeline4TableAxisVisibleShell(
                ds_key=DEEPSEEK_KEY,
            )

            pipeline.run(
                audio_file=audio_file,
                image_file=image_file,
                depth_file=depth_file,

                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=width,
                height=height,

                text_command=text_command,

                out_root=out_root,

                reuse_single_sam3d_ply=None,
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

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断，返回主菜单。")
            continue

        except Exception as e:
            print("\n❌ 本次 pipeline 运行失败:")
            print(e)
            print("返回主菜单，可修改输入后重新运行。")
            continue