#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import shutil
import subprocess
from pathlib import Path

# =========================================================
# 🔧 底层环境护盾
# =========================================================
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 当前脚本应放在 /root/SAM-3D-Grasp/pipeline_real_depth.py
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from modules.asr_sensevoice import SenseVoiceASR
from modules.slm_parser import SemanticParser
from modules.sam3_segment import SAM3Segmenter
from modules.sam3d_engine import SAM3DEngine


# =========================================================
# 路径工具：兼容 "SAM-3D-Grasp/xxx" 和 "./xxx"
# =========================================================
def resolve_path(path_like, must_exist=False):
    """
    兼容两种写法：
    1. SAM-3D-Grasp/data/test/000002/rgb/000000.png
    2. data/test/000002/rgb/000000.png
    3. /root/SAM-3D-Grasp/data/test/000002/rgb/000000.png
    """
    p = Path(path_like)

    if p.is_absolute():
        q = p
    else:
        raw = str(p).replace("\\", "/")
        repo_prefix = REPO_ROOT.name + "/"

        if raw.startswith(repo_prefix):
            # SAM-3D-Grasp/data/... -> /root/SAM-3D-Grasp/data/...
            q = REPO_ROOT / raw[len(repo_prefix):]
        else:
            q = REPO_ROOT / p

    if must_exist and not q.exists():
        raise FileNotFoundError(f"找不到文件: {q}")

    return q


class GraspingPipelineRealDepth:
    def __init__(self, ds_key, force_target_en=None):
        """
        真正真实深度图版本 pipeline。

        force_target_en:
            None: 使用语音 + DeepSeek 解析目标。
            "yellow duck": 强制目标为黄色鸭子，绕过语义误解析。
        """
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
        extract_real_pcd_script,
        register_script,
        front_mode="min",
    ):
        print("\n" + "█" * 70)
        print("🚀 具身智能 3D 抓取系统：真实深度图 partial 版本")
        print("█" * 70)

        # ---------------------------------------------------------
        # 路径解析
        # ---------------------------------------------------------
        audio_file = resolve_path(audio_file, must_exist=True)
        image_file = resolve_path(image_file, must_exist=True)
        depth_file = resolve_path(depth_file, must_exist=True)
        extract_real_pcd_script = resolve_path(extract_real_pcd_script, must_exist=True)
        register_script = resolve_path(register_script, must_exist=True)

        work_dir = resolve_path("./output_pipeline_real_depth")
        work_dir.mkdir(parents=True, exist_ok=True)

        mask_png_path = work_dir / "target_mask.png"
        mask_npy_path = work_dir / "target_mask.npy"
        mask_vis_path = work_dir / "target_2d_result.jpg"

        sam3d_output_dir = resolve_path("./output_3d")
        sam3d_output_dir.mkdir(parents=True, exist_ok=True)

        sam3d_output_path = sam3d_output_dir / "reconstructed_mesh.ply"
        pose_json_path = sam3d_output_dir / "sam3d_pose.json"

        partial_output_dir = resolve_path("./output_real_partial_from_depth")
        partial_output_dir.mkdir(parents=True, exist_ok=True)

        preferred_partial_ply = partial_output_dir / "real_partial_from_depth.ply"

        reg_out_dir = resolve_path("./output_rgb_pose_real_depth")
        reg_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Input] audio : {audio_file}")
        print(f"[Input] rgb   : {image_file}")
        print(f"[Input] depth : {depth_file}")
        print(f"[Script] extract_real_pcd : {extract_real_pcd_script}")
        print(f"[Script] register         : {register_script}")

        # ---------------------------------------------------------
        # 清理旧输出，避免误用上一次 bottle / duck 的结果
        # ---------------------------------------------------------
        print("\n[Clean] 清理关键旧输出，避免复用上一次结果...")
        self._safe_unlink(sam3d_output_path)
        self._safe_unlink(pose_json_path)
        self._safe_unlink(preferred_partial_ply)

        # 注意：
        # 不删除 /root/SAM-3D-Grasp/duck_partial_real_clean_interp_clean.ply，
        # 因为 extract_real_pcd.py 当前可能固定写这个文件。
        # 后面会根据时间戳判断是否是本次新生成。

        # ---------------------------------------------------------
        # Step 1: 语音转文字
        # ---------------------------------------------------------
        print("\n[Step 1/6] 🎙️ 正在处理语音指令...")
        raw_speech = self.asr.transcribe(str(audio_file))

        if not raw_speech:
            print("❌ 语音识别失败，pipeline 终止。")
            return

        print(f"[ASR] 原始语音识别结果: {raw_speech}")

        # ---------------------------------------------------------
        # Step 2: 语义解析
        # ---------------------------------------------------------
        print("\n[Step 2/6] 🧠 正在解析目标物体...")

        if self.force_target_en is not None:
            target_en = self.force_target_en
            print(f"[SLM] 已启用强制目标: {target_en}")
        else:
            target_en = self.slm.extract_target(raw_speech)

        if not target_en:
            print("❌ 语义解析失败，pipeline 终止。")
            return

        print(f"[Target] 当前抓取目标: {target_en}")

        # ---------------------------------------------------------
        # Step 3: RGB 图像中分割目标
        # ---------------------------------------------------------
        print(f"\n[Step 3/6] 👁️ 正在 RGB 图中定位目标: {target_en}")
        mask = self.sam.segment_by_text(str(image_file), target_en)

        if mask is None:
            print(f"❌ 视觉识别失败：图中未找到目标 '{target_en}'。")
            return

        mask = self._normalize_mask(mask)
        self._save_mask(mask, mask_png_path, mask_npy_path)
        self._visualize_pro(str(image_file), mask, target_en, str(mask_vis_path))

        print("\n[Memory System] 🧱 清空前置分割模型显存...")
        try:
            self.sam._unload_model()
        except Exception as e:
            print(f"⚠️ SAM 模型卸载失败，但不影响继续执行: {e}")

        # ---------------------------------------------------------
        # Step 4: SAM-3D 从 RGB + mask 重建完整点云
        # ---------------------------------------------------------
        print(f"\n[Step 4/6] 🧊 正在生成目标 '{target_en}' 的 SAM-3D 完整点云...")
        success = self.sam3d.generate_3d(str(image_file), mask)

        if not success:
            print("❌ SAM-3D 重建失败，请检查上方日志。")
            return

        if not sam3d_output_path.exists():
            print(f"❌ SAM-3D 未生成完整点云: {sam3d_output_path}")
            return

        if not pose_json_path.exists():
            print(f"❌ SAM-3D 未生成位姿文件: {pose_json_path}")
            print("请确认 modules/sam3d_engine.py 已经正确导出 sam3d_pose.json。")
            return

        print(f"[SAM-3D] 完整点云: {sam3d_output_path}")
        print(f"[SAM-3D] 位姿 JSON: {pose_json_path}")

        # ---------------------------------------------------------
        # Step 5: 用真实 depth + mask 提取真实残缺点云
        # ---------------------------------------------------------
        print("\n[Step 5/6] 📸 正在用真实深度图提取残缺点云...")
        partial_ply_path = self._run_extract_real_pcd(
            script_path=extract_real_pcd_script,
            rgb_path=image_file,
            depth_path=depth_file,
            mask_path=mask_png_path,
            out_dir=partial_output_dir,
            preferred_out_ply=preferred_partial_ply,
        )

        if partial_ply_path is None or not Path(partial_ply_path).exists():
            print("❌ 真实残缺点云提取失败，没有找到输出 .ply。")
            print("你可以手动检查这些位置：")
            print(f"  1. {preferred_partial_ply}")
            print(f"  2. {REPO_ROOT / 'duck_partial_real_clean_interp_clean.ply'}")
            print(f"  3. {REPO_ROOT / 'duck_partial_real_clean_raw_interp.ply'}")
            return

        partial_ply_path = Path(partial_ply_path)
        print(f"[Partial] 真实残缺点云: {partial_ply_path}")

        # ---------------------------------------------------------
        # Step 6: 调用你跑通的配准脚本
        # ---------------------------------------------------------
        print("\n[Step 6/6] 🎯 启动 RGB pose partial-locked 配准...")
        ok = self._run_registration(
            register_script=register_script,
            sam3d_ply=sam3d_output_path,
            partial_ply=partial_ply_path,
            pose_json=pose_json_path,
            out_dir=reg_out_dir,
            front_mode=front_mode,
        )

        if not ok:
            print("❌ 配准失败，请检查 step_restore_rgb_pose_partial_locked.py 的日志。")
            return

        # ---------------------------------------------------------
        # 输出汇总
        # ---------------------------------------------------------
        full_rgb_pose_path = reg_out_dir / "full_rgb_pose.ply"
        visible_shell_path = reg_out_dir / "visible_rgb_shell.ply"
        merged_rgb_pose_path = reg_out_dir / "merged_rgb_pose.ply"
        result_json_path = reg_out_dir / "pose_decode_result.json"

        print("\n" + "█" * 70)
        print("🎉 真实 depth partial pipeline 已完成！")
        print(f"🖼️  2D 目标定位图: {mask_vis_path}")
        print(f"🎭 目标 mask PNG : {mask_png_path}")
        print(f"🎭 目标 mask NPY : {mask_npy_path}")
        print(f"🧊 SAM-3D 完整点云: {sam3d_output_path}")
        print(f"🧾 SAM-3D pose json: {pose_json_path}")
        print(f"📸 真实 depth 残缺点云: {partial_ply_path}")
        print(f"📦 恢复到 RGB 视角的完整点云: {full_rgb_pose_path}")
        print(f"🟢 RGB 可见壳层: {visible_shell_path}")
        print(f"🔴🔵 最终融合点云: {merged_rgb_pose_path}")
        print(f"🧾 配准参数与指标: {result_json_path}")
        print("█" * 70)

    # =========================================================
    # 工具函数
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
        img = cv2.imread(img_path)

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

        cv2.imwrite(save_name, result)
        print(f"[Pipeline] 🖼️ 2D 定位结果图已保存: {save_name}")

    def _safe_unlink(self, path):
        try:
            path = Path(path)
            if path.exists():
                path.unlink()
                print(f"[Clean] 删除旧文件: {path}")
        except Exception as e:
            print(f"⚠️ 删除旧文件失败 {path}: {e}")

    def _get_help_text(self, script_path):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return (result.stdout or "") + "\n" + (result.stderr or "")
        except Exception:
            return ""

    def _pick_flag(self, help_text, candidates):
        """
        从脚本 help 文本里自动找参数名。
        例如 extract_real_pcd.py 可能叫 --rgb，也可能叫 --image。
        """
        if not help_text.strip():
            return None

        for flag in candidates:
            pattern = r"(^|\\s)" + re.escape(flag) + r"([,\\s=]|$)"
            if re.search(pattern, help_text):
                return flag

        return None

    def _find_actual_partial_output(self, out_dir, preferred_out_ply, start_time):
        """
        兼容 extract_real_pcd.py 没有按 --out 保存的情况。

        你的当前日志显示它实际保存为：
            duck_partial_real_clean_raw_interp.ply
            duck_partial_real_clean_interp_clean.ply

        而且保存位置是当前工作目录：
            /root/SAM-3D-Grasp/
        """
        out_dir = Path(out_dir)
        preferred_out_ply = Path(preferred_out_ply)

        # 1. 优先标准输出路径
        if preferred_out_ply.exists():
            print(f"[extract_real_pcd] ✅ 找到标准输出 partial: {preferred_out_ply}")
            return preferred_out_ply

        # 2. 常见命名，优先 clean 版本
        common_names = [
            "real_partial_from_depth.ply",
            "duck_partial_real_clean_interp_clean.ply",
            "duck_partial_real_clean_raw_interp.ply",
            "visible_all_cv_strict.ply",
            "visible_partial.ply",
            "target_partial.ply",
            "partial_real.ply",
            "partial.ply",
        ]

        # 3. 搜索目录
        search_dirs = [
            out_dir,
            REPO_ROOT,
            REPO_ROOT / "output_real_partial_from_depth",
            REPO_ROOT / "output_3d",
        ]

        # 4. 优先按固定文件名找
        for d in search_dirs:
            for name in common_names:
                candidate = d / name
                if candidate.exists():
                    # 尽量判断是不是本次新生成的
                    try:
                        mtime = candidate.stat().st_mtime
                        if mtime < start_time - 5.0:
                            print(f"[extract_real_pcd] ⚠️ 找到旧 partial，但时间早于本次运行，暂不优先使用: {candidate}")
                            continue
                    except Exception:
                        pass

                    print(f"[extract_real_pcd] ✅ 找到实际输出 partial: {candidate}")

                    # 复制到标准路径，后续统一使用
                    try:
                        preferred_out_ply.parent.mkdir(parents=True, exist_ok=True)
                        if candidate.resolve() != preferred_out_ply.resolve():
                            shutil.copyfile(str(candidate), str(preferred_out_ply))
                            print(f"[extract_real_pcd] 📦 已同步复制到标准路径: {preferred_out_ply}")
                            return preferred_out_ply
                    except Exception as e:
                        print(f"⚠️ 复制 partial 到标准路径失败，直接使用原始路径: {e}")

                    return candidate

        # 5. 找本次新生成的所有 ply，优先 clean，取最新
        new_plys = []
        for d in search_dirs:
            if not d.exists():
                continue

            for p in d.glob("*.ply"):
                try:
                    if p.stat().st_mtime >= start_time - 5.0:
                        new_plys.append(p)
                except Exception:
                    pass

        if new_plys:
            # 优先包含 clean 的点云
            clean_plys = [p for p in new_plys if "clean" in p.name.lower()]
            candidates = clean_plys if clean_plys else new_plys
            candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
            candidate = candidates[0]

            print(f"[extract_real_pcd] ✅ 找到本次新生成的最新 partial: {candidate}")

            try:
                preferred_out_ply.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != preferred_out_ply.resolve():
                    shutil.copyfile(str(candidate), str(preferred_out_ply))
                    print(f"[extract_real_pcd] 📦 已同步复制到标准路径: {preferred_out_ply}")
                    return preferred_out_ply
            except Exception as e:
                print(f"⚠️ 复制 partial 到标准路径失败，直接使用原始路径: {e}")

            return candidate

        # 6. 最后兜底：找所有 ply，优先 clean，取最新
        all_plys = []
        for d in search_dirs:
            if not d.exists():
                continue

            for p in d.glob("*.ply"):
                try:
                    all_plys.append(p)
                except Exception:
                    pass

        if all_plys:
            clean_plys = [p for p in all_plys if "clean" in p.name.lower()]
            candidates = clean_plys if clean_plys else all_plys
            candidates = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)
            candidate = candidates[0]

            print(f"[extract_real_pcd] ⚠️ 没找到明确的新 partial，兜底使用最新 partial: {candidate}")

            try:
                preferred_out_ply.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != preferred_out_ply.resolve():
                    shutil.copyfile(str(candidate), str(preferred_out_ply))
                    print(f"[extract_real_pcd] 📦 已同步复制到标准路径: {preferred_out_ply}")
                    return preferred_out_ply
            except Exception as e:
                print(f"⚠️ 复制 partial 到标准路径失败，直接使用原始路径: {e}")

            return candidate

        return None

    def _run_extract_real_pcd(
        self,
        script_path,
        rgb_path,
        depth_path,
        mask_path,
        out_dir,
        preferred_out_ply,
    ):
        """
        调用 /root/SAM-3D-Grasp/extract_real_pcd.py。

        重点：
        - 不用 HO3D GT model。
        - 不用 step1_auto.py。
        - 只使用真实 RGB、真实 depth、当前目标 mask。
        - 如果 extract_real_pcd.py 不按 --out 保存，也能自动找到它实际写出的 ply。
        """
        out_dir = Path(out_dir)
        preferred_out_ply = Path(preferred_out_ply)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 记录开始时间，用于判断哪些 .ply 是本次生成
        start_time = time.time()

        help_text = self._get_help_text(script_path)

        commands = []

        # -----------------------------------------------------
        # 优先：根据 -h 自动组装命令
        # -----------------------------------------------------
        if help_text.strip():
            rgb_flag = self._pick_flag(
                help_text,
                ["--rgb", "--rgb-path", "--image", "--image-path", "--image-file", "--color"],
            )
            depth_flag = self._pick_flag(
                help_text,
                ["--depth", "--depth-path", "--depth-file", "--depth-image"],
            )
            mask_flag = self._pick_flag(
                help_text,
                ["--mask", "--mask-path", "--mask-file", "--seg", "--seg-mask"],
            )
            out_flag = self._pick_flag(
                help_text,
                ["--out", "--output", "--out-ply", "--output-ply", "--save-path", "--ply"],
            )
            out_dir_flag = self._pick_flag(
                help_text,
                ["--out-dir", "--output-dir", "--save-dir"],
            )

            cmd = [sys.executable, str(script_path)]

            if rgb_flag:
                cmd += [rgb_flag, str(rgb_path)]
            if depth_flag:
                cmd += [depth_flag, str(depth_path)]
            if mask_flag:
                cmd += [mask_flag, str(mask_path)]

            if out_flag:
                cmd += [out_flag, str(preferred_out_ply)]
            elif out_dir_flag:
                cmd += [out_dir_flag, str(out_dir)]

            if len(cmd) > 2:
                commands.append(cmd)

        # -----------------------------------------------------
        # fallback 1：你的日志里当前可运行的形式
        # -----------------------------------------------------
        commands.append([
            sys.executable, str(script_path),
            "--rgb", str(rgb_path),
            "--depth", str(depth_path),
            "--mask", str(mask_path),
            "--out", str(preferred_out_ply),
        ])

        # fallback 2：image/output 写法
        commands.append([
            sys.executable, str(script_path),
            "--image", str(rgb_path),
            "--depth", str(depth_path),
            "--mask", str(mask_path),
            "--output", str(preferred_out_ply),
        ])

        # fallback 3：out-dir 写法
        commands.append([
            sys.executable, str(script_path),
            "--rgb", str(rgb_path),
            "--depth", str(depth_path),
            "--mask", str(mask_path),
            "--out-dir", str(out_dir),
        ])

        # 去重
        unique_commands = []
        seen = set()
        for cmd in commands:
            key = tuple(cmd)
            if key not in seen:
                unique_commands.append(cmd)
                seen.add(key)

        last_returncode = None

        for idx, cmd in enumerate(unique_commands, start=1):
            print(f"\n[extract_real_pcd] 尝试命令 #{idx}:")
            print(" ".join(cmd))

            result = subprocess.run(cmd)
            last_returncode = result.returncode

            if result.returncode == 0:
                print("[extract_real_pcd] ✅ 命令执行成功。")
                break

            # argparse 参数错误一般是 2，可以继续尝试其他参数形式
            if result.returncode == 2:
                print("[extract_real_pcd] ⚠️ 参数形式可能不匹配，尝试下一种写法...")
                continue

            # 非参数错误，通常是真实运行错误，不盲目继续
            print(f"[extract_real_pcd] ❌ 运行失败，returncode={result.returncode}")
            break

        if last_returncode != 0:
            print("\n❌ extract_real_pcd.py 执行失败。")
            print("请检查它实际支持的参数：")
            print(f"python {script_path} -h")
            return None

        # -----------------------------------------------------
        # 关键修复点：
        # extract_real_pcd.py 可能忽略 --out，写死输出文件名。
        # 所以这里专门找实际生成的 partial。
        # -----------------------------------------------------
        partial_ply = self._find_actual_partial_output(
            out_dir=out_dir,
            preferred_out_ply=preferred_out_ply,
            start_time=start_time,
        )

        return partial_ply

    def _run_registration(
        self,
        register_script,
        sam3d_ply,
        partial_ply,
        pose_json,
        out_dir,
        front_mode="min",
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(register_script),
            "--sam3d", str(sam3d_ply),
            "--partial", str(partial_ply),
            "--pose-json", str(pose_json),
            "--out-dir", str(out_dir),
            "--front-mode", str(front_mode),
        ]

        print("[Register] 执行命令:")
        print(" ".join(cmd))

        result = subprocess.run(cmd)

        if result.returncode == 0:
            return True

        # 如果配准脚本不支持 --front-mode，则自动去掉重试一次
        if result.returncode == 2:
            print("\n[Register] ⚠️ 检测到参数错误，尝试去掉 --front-mode 重试一次。")

            cmd_retry = [
                sys.executable,
                str(register_script),
                "--sam3d", str(sam3d_ply),
                "--partial", str(partial_ply),
                "--pose-json", str(pose_json),
                "--out-dir", str(out_dir),
            ]

            print("[Register] 重试命令:")
            print(" ".join(cmd_retry))

            retry = subprocess.run(cmd_retry)
            return retry.returncode == 0

        return False


if __name__ == "__main__":
    # =========================================================
    # ⚙️ 用户配置区
    # =========================================================

    # 推荐使用环境变量：
    # export DEEPSEEK_KEY="你的key"
    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "<YOUR_DEEPSEEK_KEY>")

    # 你这次给的真实输入
    # 注意：脚本会自动把 SAM-3D-Grasp/data/... 解析成 /root/SAM-3D-Grasp/data/...
    TEST_AUDIO = "SAM-3D-Grasp/data/test_audio1.m4a"
    TEST_IMAGE = "SAM-3D-Grasp/data/test/000002/rgb/000000.png"
    TEST_DEPTH = "SAM-3D-Grasp/data/test/000002/depth/000000.png"

    # 真实残缺点云提取算法
    EXTRACT_REAL_PCD_SCRIPT = "SAM-3D-Grasp/extract_real_pcd.py"

    # 你已经跑通的配准代码
    REGISTER_SCRIPT = "SAM-3D-Grasp/step_restore_rgb_pose_partial_locked.py"

    # 默认使用语音 + DeepSeek 解析。
    # 如果它又把鸭子误解析成 bleach bottle，就改成：
    # FORCE_TARGET_EN = "yellow duck"
    FORCE_TARGET_EN = None

    # =========================================================

    pipeline = GraspingPipelineRealDepth(
        ds_key=DEEPSEEK_KEY,
        force_target_en=FORCE_TARGET_EN,
    )

    pipeline.run(
        audio_file=TEST_AUDIO,
        image_file=TEST_IMAGE,
        depth_file=TEST_DEPTH,
        extract_real_pcd_script=EXTRACT_REAL_PCD_SCRIPT,
        register_script=REGISTER_SCRIPT,
        front_mode="min",
    )