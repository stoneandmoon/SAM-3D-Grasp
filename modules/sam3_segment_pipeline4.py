#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modules/sam3_segment_pipeline4.py

这是给 pipeline4 使用的 SAM3 2D 分割模块。

和原来的 sam3_segment.py 相比：
1. 保留原始接口：SAM3Segmenter.segment_by_text(image_path, text_query)
2. 新增 keep_loaded 参数：
   - keep_loaded=False：默认行为，推理后自动卸载 SAM3 分割模型，释放显存
   - keep_loaded=True ：推理后保留模型，适合连续分割多张图
3. 新增 close() / unload() 方法
4. 更稳的显存清理
5. 不覆盖你原来的 modules/sam3_segment.py

注意：
  这个文件负责的是 SAM3 2D 分割，不是 SAM3D 3D 重建。
  之前 pipeline4 被 kill 的核心原因还是 SAM3D joint 生成时重新开进程加载权重；
  那个还需要改 modules/sam3d_engine.py 和 pipeline4.py。
"""

import os
import sys
import gc
from pathlib import Path

import torch
import numpy as np
from PIL import Image


# ============================================================
# 仓库路径
# ============================================================

SAM3_REPO_BASE = "/root/SAM-3D-Grasp"

if SAM3_REPO_BASE not in sys.path:
    sys.path.append(SAM3_REPO_BASE)


# ============================================================
# SAM3 官方组件导入
# ============================================================

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"[SAM3] 导入核心组件失败: {e}")
    build_sam3_image_model = None
    Sam3Processor = None


# ============================================================
# SAM3 Segmenter
# ============================================================

class SAM3Segmenter:
    def __init__(
        self,
        device=None,
        auto_unload=True,
        load_from_HF=True,
        enable_segmentation=True,
    ):
        """
        参数：
          device:
            None 时自动选择 cuda/cpu。

          auto_unload:
            True  表示 segment_by_text 默认推理完自动释放显存。
            False 表示默认保留模型。
            也可以在 segment_by_text(..., keep_loaded=True/False) 里单次覆盖。

          load_from_HF:
            传给 build_sam3_image_model。

          enable_segmentation:
            传给 build_sam3_image_model。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_unload = bool(auto_unload)
        self.load_from_HF = bool(load_from_HF)
        self.enable_segmentation = bool(enable_segmentation)

        self.model = None
        self.processor = None

        self.repo_root = Path(SAM3_REPO_BASE)
        self.vocab_path = self.repo_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

    # ------------------------------------------------------------
    # 显存工具
    # ------------------------------------------------------------
    def _cuda_cleanup(self):
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    # ------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------
    def _load_model(self):
        if self.model is not None and self.processor is not None:
            return

        if build_sam3_image_model is None or Sam3Processor is None:
            raise ImportError(
                "SAM3 核心组件未成功导入，请检查 sam3 包和路径。"
            )

        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"找不到 SAM3 vocab 文件: {self.vocab_path}"
            )

        print(f"[SAM3] 正在载入视觉权重至 {self.device}...")

        self.model = build_sam3_image_model(
            bpe_path=str(self.vocab_path),
            device=self.device,
            eval_mode=True,
            load_from_HF=self.load_from_HF,
            enable_segmentation=self.enable_segmentation,
        )

        self.processor = Sam3Processor(self.model)

        print("[SAM3] 模型加载完成。")

    # ------------------------------------------------------------
    # 卸载模型
    # ------------------------------------------------------------
    def _unload_model(self):
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass

        if self.processor is not None:
            try:
                del self.processor
            except Exception:
                pass

        self.model = None
        self.processor = None

        self._cuda_cleanup()

        print("[SAM3] 显存深度清理完成。")

    def unload(self):
        self._unload_model()

    def close(self):
        self._unload_model()

    # ------------------------------------------------------------
    # 分割
    # ------------------------------------------------------------
    def segment_by_text(
        self,
        image_path: str,
        text_query: str,
        keep_loaded=None,
        return_score=False,
    ):
        """
        输入：
          image_path:
            RGB 图片路径。

          text_query:
            文本提示词，例如 "yellow duck"。

          keep_loaded:
            None:
              使用 self.auto_unload 决定是否卸载。
            True:
              本次推理后保留模型。
            False:
              本次推理后卸载模型。

          return_score:
            False:
              只返回 best_mask。
            True:
              返回 (best_mask, best_score)。

        返回：
          best_mask:
            numpy array，通常是 H x W，数值为 bool/0-1。
        """
        image_path = str(image_path)

        if not os.path.exists(image_path):
            print(f"[SAM3] 找不到图像: {image_path}")
            return (None, None) if return_score else None

        if not text_query or not str(text_query).strip():
            print("[SAM3] text_query 为空。")
            return (None, None) if return_score else None

        # keep_loaded 的默认逻辑：
        # auto_unload=True  -> 默认 keep_loaded=False
        # auto_unload=False -> 默认 keep_loaded=True
        if keep_loaded is None:
            keep_loaded = not self.auto_unload

        try:
            self._load_model()

            image = Image.open(image_path).convert("RGB")

            print(f"[SAM3] 正在执行零样本分割，提示词: '{text_query}'...")

            with torch.no_grad():
                state = self.processor.set_image(image)
                output = self.processor.set_text_prompt(
                    state=state,
                    prompt=text_query,
                )

            if output is None:
                print("[SAM3] 输出为空。")
                return (None, None) if return_score else None

            masks = output.get("masks", None)
            scores = output.get("scores", None)

            if masks is None or scores is None:
                print(f"[SAM3] 输出缺少 masks/scores，keys={list(output.keys())}")
                return (None, None) if return_score else None

            if masks.numel() == 0:
                print(f"[SAM3] 未能识别到物体: {text_query}")
                return (None, None) if return_score else None

            best_idx = torch.argmax(scores).item()
            best_score = float(scores[best_idx].detach().cpu().item())

            best_mask = masks[best_idx].detach().cpu().numpy()

            if best_mask.ndim == 3:
                best_mask = best_mask[0]

            # 保持和原 pipeline 兼容：
            # 后续 _normalize_mask 会再做 >0.5。
            best_mask = np.asarray(best_mask)

            print(f"[SAM3] 分割成功 (Score: {best_score:.2f})")

            if return_score:
                return best_mask, best_score

            return best_mask

        except Exception as e:
            print(f"[SAM3] 推理异常: {e}")
            return (None, None) if return_score else None

        finally:
            if keep_loaded:
                print("[SAM3] keep_loaded=True，本次分割后保留 SAM3 模型。")
            else:
                self._unload_model()

    # ------------------------------------------------------------
    # 兼容辅助
    # ------------------------------------------------------------
    def __del__(self):
        try:
            self._unload_model()
        except Exception:
            pass
