#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

# =========================================================
# 🔧 底层环境护盾：强制环境变量与路径注入
# =========================================================
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.abspath("."))
# =========================================================

import cv2
import numpy as np

from modules.asr_sensevoice import SenseVoiceASR
from modules.slm_parser import SemanticParser
from modules.sam3_segment import SAM3Segmenter
from modules.sam3d_engine import SAM3DEngine


class GraspingPipeline:
    def __init__(self, ds_key):
        """
        初始化端到端具身智能流水线。
        注意：这里只实例化，不在 __init__ 里硬加载大模型。
        """
        self.asr = SenseVoiceASR()
        self.slm = SemanticParser(api_key=ds_key)
        self.sam = SAM3Segmenter()
        self.sam3d = SAM3DEngine()

    def run(self, audio_file, image_file, model_dir):
        print("\n" + "█" * 60)
        print("🚀 具身智能 3D 抓取系统：全链路启动")
        print("█" * 60)

        # ---------------------------------------------------------
        # 路径解析
        # image_file 示例: ./ho3d_test/train/ABF13/rgb/0000.jpg
        # ---------------------------------------------------------
        base_dir = os.path.dirname(os.path.dirname(image_file))   # -> ./ho3d_test/train/ABF13
        file_name = os.path.basename(image_file).split('.')[0]    # -> 0000

        meta_file = os.path.join(base_dir, "meta", f"{file_name}.pkl")

        # 2D 可视化
        vis2d_path = "pipeline_output_2D_result.jpg"

        # SAM-3D 输出（这里保持你原来成功链路的原始输出，不做额外改动）
        sam3d_output_dir = "./output_3d"
        sam3d_output_path = os.path.join(sam3d_output_dir, "reconstructed_mesh.ply")
        pose_json_path = os.path.join(sam3d_output_dir, "sam3d_pose.json")

        # 真实 partial 输出目录
        partial_output_dir = f"./output_partial_{os.path.basename(base_dir)}"
        partial_ply_path = os.path.join(partial_output_dir, "visible_all_cv_strict.ply")

        # 最终 RGB pose 配准输出目录
        rgb_pose_output_dir = "./output_partial_locked"

        # 你那份成功脚本的文件名
        # 这里假设你已经把“成功版 partial 锁尺度脚本”保存成这个名字
        register_script = "./step_restore_rgb_pose_partial_locked.py"

        # ---------------------------------------------------------
        # 阶段 1: 语音转文字
        # ---------------------------------------------------------
        print("\n[Step 1/6] 🎙️ 正在处理语音指令...")
        raw_speech = self.asr.transcribe(audio_file)
        if not raw_speech:
            print("❌ 语音识别失败，流水线终止。")
            return

        # ---------------------------------------------------------
        # 阶段 2: 语义解析
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

        self._visualize_pro(image_file, mask, target_en, save_name=vis2d_path)

        # ---------------------------------------------------------
        # 显存隔离墙
        # ---------------------------------------------------------
        print("\n[Memory System] 🧱 触发显存隔离墙：强制清空前置感知模型...")
        self.sam._unload_model()

        # ---------------------------------------------------------
        # 阶段 4: SAM-3D 重建
        # ---------------------------------------------------------
        print(f"\n[Step 4/6] 🧊 开始将 '{target_en}' 重建为完整 3D 模型...")
        success = self.sam3d.generate_3d(image_file, mask)
        if not success:
            print("\n❌ 3D 重建失败，请检查报错日志。")
            return

        # 关键检查：你当前的 SAM-3D 导出链必须已经能导出这两个文件
        if not os.path.exists(sam3d_output_path):
            print(f"❌ 找不到 SAM-3D 输出完整点云: {sam3d_output_path}")
            return

        if not os.path.exists(pose_json_path):
            print(f"❌ 找不到姿态文件: {pose_json_path}")
            print("   请先确认你当前的 sam3d_engine 已经导出 sam3d_pose.json。")
            return

        # ---------------------------------------------------------
        # 阶段 5: 提取真实世界 partial
        # ---------------------------------------------------------
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
            print(f"❌ 找不到真实 partial 点云: {partial_ply_path}")
            return

        # ---------------------------------------------------------
        # 阶段 6: 直接调用成功版 partial 锁尺度配准脚本
        # 重点：这里不再插入 bridge / PointDSC / pose 反解 / residual 二次处理
        #      而是把原始 full + 原始 partial + pose_json 直接喂给成功脚本
        # ---------------------------------------------------------
        print("\n[Step 6/6] 🎯 调用成功版 partial 锁尺度 RGB 位姿恢复脚本...")

        if not os.path.exists(register_script):
            print(f"❌ 找不到配准脚本: {register_script}")
            print("   请先把你那份成功版脚本保存成这个文件名。")
            return

        step6_cmd = [
            "python", register_script,
            "--sam3d", sam3d_output_path,
            "--partial", partial_ply_path,
            "--pose-json", pose_json_path,
            "--out-dir", rgb_pose_output_dir,
            "--front-mode", "min"
        ]

        print(f"    执行命令: {' '.join(step6_cmd)}")
        result_step6 = subprocess.run(step6_cmd)

        if result_step6.returncode != 0:
            print("\n❌ partial 锁尺度版 RGB 位姿恢复失败，请检查日志。")
            return

        # ---------------------------------------------------------
        # 输出整理
        # ---------------------------------------------------------
        full_rgb_pose_path = os.path.join(rgb_pose_output_dir, "full_rgb_pose.ply")
        visible_shell_path = os.path.join(rgb_pose_output_dir, "visible_rgb_shell.ply")
        merged_rgb_pose_path = os.path.join(rgb_pose_output_dir, "merged_rgb_pose.ply")
        result_json_path = os.path.join(rgb_pose_output_dir, "pose_decode_result.json")

        print("\n" + "█" * 60)
        print("🎉 端到端具身智能抓取系统跑通全流程（已接入成功版配准核心）")
        print(f"🖼️  2D 定位结果: {vis2d_path}")
        print(f"🧊 SAM-3D 完整点云: {sam3d_output_path}")
        print(f"📸 真实 partial 点云: {partial_ply_path}")
        print(f"📦 【输出项 1】完整 RGB 位姿点云: {full_rgb_pose_path}")
        print(f"📦 【输出项 2】可见壳层点云: {visible_shell_path}")
        print(f"📦 【输出项 3】红蓝融合可视化: {merged_rgb_pose_path}")
        print(f"📄 【输出项 4】配准结果 JSON: {result_json_path}")
        print("█" * 60)

    def _visualize_pro(self, img_path, mask, label, save_name="pipeline_output_2D_result.jpg"):
        """
        专业级 2D 结果可视化：高亮掩码 + 红色轮廓 + 青色 BBox
        """
        img = cv2.imread(img_path)
        if img is None:
            return

        mask_bool = mask > 0.5
        mask_uint8 = (mask_bool).astype(np.uint8) * 255

        overlay = img.copy()
        overlay[mask_bool] = [0, 255, 0]
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
    # ---------------------------------------------------------

    pipeline = GraspingPipeline(ds_key=DEEPSEEK_KEY)
    pipeline.run(TEST_AUDIO, TEST_IMAGE, MODEL_DIR)