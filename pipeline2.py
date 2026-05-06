#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# =========================================================
# 🔧 底层环境护盾
# =========================================================
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np

from modules.asr_sensevoice import SenseVoiceASR
from modules.slm_parser import SemanticParser
from modules.sam3_segment import SAM3Segmenter
from modules.sam3d_engine import SAM3DEngine


class GraspingPipeline:
    def __init__(self, ds_key):
        """
        端到端具身智能流水线。
        注意：这里只实例化，不预先常驻加载大模型权重。
        """
        self.asr = SenseVoiceASR()
        self.slm = SemanticParser(api_key=ds_key)
        self.sam = SAM3Segmenter()
        self.sam3d = SAM3DEngine()

    def run(
        self,
        audio_file,
        image_file,
        model_dir,
        use_duck_registration_target=False,
        duck_sam3d_path="./sam3d_duck_clean.ply",
        duck_partial_path="./duck_partial_real_clean_interp_clean.ply",
        duck_pose_json="./output_3d/sam3d_pose.json",
    ):
        print("\n" + "█" * 60)
        print("🚀 具身智能 3D 抓取系统：全链路启动（新版 RGB 姿态恢复配准）")
        print("█" * 60)

        # ---------------------------------------------------------
        # 路径解析
        # image_file 示例: ./ho3d_test/train/ABF13/rgb/0000.jpg
        # ---------------------------------------------------------
        base_dir = os.path.dirname(os.path.dirname(image_file))   # -> ./ho3d_test/train/ABF13
        file_name = os.path.basename(image_file).split('.')[0]    # -> 0000

        meta_file = os.path.join(base_dir, "meta", f"{file_name}.pkl")
        partial_output_dir = f"./output_partial_{os.path.basename(base_dir)}"

        # 2D 可视化
        mask_output_path = "pipeline_output_2D_result.jpg"

        # 3D 输出
        sam3d_output_dir = "./output_3d"
        sam3d_output_path = os.path.join(sam3d_output_dir, "reconstructed_mesh.ply")
        pose_json_path = os.path.join(sam3d_output_dir, "sam3d_pose.json")

        # partial 输出
        partial_ply_path = os.path.join(partial_output_dir, "visible_all_cv_strict.ply")

        # 新版配准输出
        rgbpose_output_dir = "./output_restore_rgb_pose_refined"

        # ---------------------------------------------------------
        # 阶段 1: 语音转文字
        # ---------------------------------------------------------
        print("\n[Step 1/6] 🎙️ 正在处理语音指令...")
        raw_speech = self.asr.transcribe(audio_file)
        if not raw_speech:
            print("❌ 语音识别失败，流水线终止。")
            return

        # ---------------------------------------------------------
        # 阶段 2: 语义解构
        # ---------------------------------------------------------
        print("\n[Step 2/6] 🧠 正在通过 DeepSeek 解析语义...")
        target_en = self.slm.extract_target(raw_speech)
        if not target_en:
            print("❌ 语义解析失败，流水线终止。")
            return

        # ---------------------------------------------------------
        # 阶段 3: 2D 视觉分割
        # ---------------------------------------------------------
        print(f"\n[Step 3/6] 👁️ 正在图像中定位目标: '{target_en}'...")
        mask = self.sam.segment_by_text(image_file, target_en)

        if mask is None:
            print(f"❌ 视觉识别失败：图中未找到 '{target_en}'，流水线终止。")
            return

        self._visualize_pro(image_file, mask, target_en, save_name=mask_output_path)

        # =========================================================
        # ⚠️ 显存隔离墙
        # =========================================================
        print("\n[Memory System] 🧱 触发显存隔离墙：强制清空前置感知模型...")
        self.sam._unload_model()

        # ---------------------------------------------------------
        # 阶段 4: 3D 重建
        # ---------------------------------------------------------
        # 如果开启鸭子调试模式，这一步只作为“前面链路演示”，
        # 但最终配准会优先使用你给定的 duck 文件。
        # ---------------------------------------------------------
        print(f"\n[Step 4/6] 🧊 开始将 '{target_en}' 重建为完整 3D 模型...")
        success = self.sam3d.generate_3d(image_file, mask)
        if not success:
            print("\n❌ 3D 重建失败，请检查报错日志。")
            return

        # pose json 必须存在，新版配准会用到
        if not os.path.exists(pose_json_path):
            print(f"❌ 未找到姿态文件: {pose_json_path}")
            print("   请先确认 modules/sam3d_engine.py 已经按新版保存了 sam3d_pose.json")
            return

        # ---------------------------------------------------------
        # 阶段 5: 提取真实 partial 点云
        # ---------------------------------------------------------
        # 如果是鸭子调试模式，并且你已经有 duck partial，
        # 这里就不再强制跑 step1_auto.py。
        # ---------------------------------------------------------
        if use_duck_registration_target:
            print("\n[Step 5/6] 📸 鸭子调试模式：跳过 step1_auto，直接使用预先准备好的 duck partial")
            if not os.path.exists(duck_partial_path):
                print(f"❌ 找不到 duck partial: {duck_partial_path}")
                return
            if not os.path.exists(duck_sam3d_path):
                print(f"❌ 找不到 duck sam3d 点云: {duck_sam3d_path}")
                return
            if not os.path.exists(duck_pose_json):
                print(f"❌ 找不到 duck pose json: {duck_pose_json}")
                return

            reg_sam3d_path = duck_sam3d_path
            reg_partial_path = duck_partial_path
            reg_pose_json = duck_pose_json
            reg_out_dir = "./output_duck_rgb_pose_refined"
        else:
            print("\n[Step 5/6] 📸 正在提取真实世界残缺视觉点云...")
            step1_cmd = [
                "python", "step1_auto.py",
                "--meta", meta_file,
                "--model-dir", model_dir,
                "--out-dir", partial_output_dir
            ]
            print(f"    执行命令: {' '.join(step1_cmd)}")
            result_step1 = subprocess.run(step1_cmd)
            if result_step1.returncode != 0:
                print("❌ 残缺点云提取失败！")
                return

            if not os.path.exists(partial_ply_path):
                print(f"❌ 没有找到 partial 输出: {partial_ply_path}")
                return
            if not os.path.exists(sam3d_output_path):
                print(f"❌ 没有找到 sam3d 输出: {sam3d_output_path}")
                return

            reg_sam3d_path = sam3d_output_path
            reg_partial_path = partial_ply_path
            reg_pose_json = pose_json_path
            reg_out_dir = rgbpose_output_dir

        # ---------------------------------------------------------
        # 阶段 6: 新版 RGB 位姿恢复 + 防穿模精修配准
        # ---------------------------------------------------------
        print("\n[Step 6/6] 🎯 启动新版 RGB 位姿恢复 + 防穿模精修配准...")

        step6_cmd = [
            "python", "step_restore_rgb_pose.py",
            "--model-dir", model_dir,      # 兼容参数，新版内部不拿它做尺度锁定
            "--sam3d", reg_sam3d_path,
            "--partial", reg_partial_path,
            "--pose-json", reg_pose_json,
            "--out-dir", reg_out_dir,
        ]
        print(f"    执行命令: {' '.join(step6_cmd)}")
        result_step6 = subprocess.run(step6_cmd)

        if result_step6.returncode != 0:
            print("\n❌ 新版 RGB 位姿恢复配准失败，请检查日志。")
            return

        # ---------------------------------------------------------
        # 输出路径
        # ---------------------------------------------------------
        full_refined_ply = os.path.join(reg_out_dir, "full_rgb_pose_refined.ply")
        shell_refined_ply = os.path.join(reg_out_dir, "visible_rgb_shell_refined.ply")
        merged_refined_ply = os.path.join(reg_out_dir, "merged_rgb_pose_refined.ply")
        result_json = os.path.join(reg_out_dir, "pose_decode_result.json")

        print("\n" + "█" * 60)
        print("🎉 新版端到端配准流程已跑通！")
        print(f"🖼️ 【输出项 1】2D 定位结果图: {mask_output_path}")
        print(f"💎 【输出项 2】恢复到 RGB 视角的完整点云: {full_refined_ply}")
        print(f"🟢 【输出项 3】RGB 视角可见壳: {shell_refined_ply}")
        print(f"🔴🔵【输出项 4】最终蓝红融合效果图: {merged_refined_ply}")
        print(f"🧾 【输出项 5】配准参数与指标: {result_json}")
        print("█" * 60)

    def _visualize_pro(self, img_path, mask, label, save_name="pipeline_output_2D_result.jpg"):
        """
        专业级 2D 结果可视化：高亮掩码 + 红色轮廓 + 青色 BBox
        """
        img = cv2.imread(img_path)
        if img is None:
            print("❌ 无法读取图像，2D 可视化失败。")
            return

        mask_bool = mask > 0.5
        mask_uint8 = (mask_bool.astype(np.uint8)) * 255

        overlay = img.copy()
        overlay[mask_bool] = [0, 255, 0]  # 亮绿色
        result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 500:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(result, (x, y - 30), (x + w, y), (255, 255, 0), -1)
                cv2.putText(
                    result, label.upper(), (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
                )

        cv2.imwrite(save_name, result)
        print(f"[Pipeline] 🖼️ 2D 定位结果图已存至: {save_name}")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # ⚙️ 用户配置区
    # ---------------------------------------------------------
    DEEPSEEK_KEY = "<YOUR_DEEPSEEK_KEY>"

    TEST_AUDIO = "./ho3d_test/white.m4a"
    TEST_IMAGE = "./ho3d_test/train/ABF13/rgb/0000.jpg"
    MODEL_DIR = "./ho3d_test/models/021_bleach_cleanser"

    # 是否直接使用你已经准备好的鸭子配准目标
    USE_DUCK_REGISTRATION_TARGET = True

    # 鸭子配准目标文件
    DUCK_SAM3D_PLY = "./sam3d_duck_clean.ply"
    DUCK_PARTIAL_PLY = "./duck_partial_real_clean_interp_clean.ply"
    DUCK_POSE_JSON = "./output_3d/sam3d_pose.json"
    # ---------------------------------------------------------

    pipeline = GraspingPipeline(ds_key=DEEPSEEK_KEY)
    pipeline.run(
        audio_file=TEST_AUDIO,
        image_file=TEST_IMAGE,
        model_dir=MODEL_DIR,
        use_duck_registration_target=USE_DUCK_REGISTRATION_TARGET,
        duck_sam3d_path=DUCK_SAM3D_PLY,
        duck_partial_path=DUCK_PARTIAL_PLY,
        duck_pose_json=DUCK_POSE_JSON,
    )