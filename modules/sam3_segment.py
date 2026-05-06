import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import gc

# 确保指向你下载的官方仓库
SAM3_REPO_BASE = "/root/SAM-3D-Grasp"
if SAM3_REPO_BASE not in sys.path:
    sys.path.append(SAM3_REPO_BASE)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError as e:
    print(f"[SAM3] 导入核心组件失败: {e}")

class SAM3Segmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is None:
            print("[SAM3] 正在载入视觉权重至 2080 Ti...")
            vocab_path = os.path.join(SAM3_REPO_BASE, "assets", "bpe_simple_vocab_16e6.txt.gz")
            self.model = build_sam3_image_model(
                bpe_path=vocab_path,
                device=self.device,
                eval_mode=True,
                load_from_HF=True,
                enable_segmentation=True
            )
            self.processor = Sam3Processor(self.model)
            print("[SAM3] 模型加载完成。")

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[SAM3] 显存深度清理完成。")

    def segment_by_text(self, image_path: str, text_query: str):
        if not os.path.exists(image_path):
            return None

        try:
            self._load_model()
            # 使用 PIL 读取以匹配官方预处理逻辑
            image = Image.open(image_path).convert("RGB")
            
            print(f"[SAM3] 正在执行零样本分割，提示词: '{text_query}'...")
            with torch.no_grad():
                state = self.processor.set_image(image)
                output = self.processor.set_text_prompt(state=state, prompt=text_query)
            
            masks = output["masks"]
            scores = output["scores"]
            
            if masks.numel() == 0:
                print(f"[SAM3] ⚠ 未能识别到物体: {text_query}")
                return None
                
            # 提取置信度最高的掩码
            best_idx = torch.argmax(scores).item()
            best_mask = masks[best_idx].detach().cpu().numpy()
            
            if best_mask.ndim == 3:
                best_mask = best_mask[0]
                
            print(f"[SAM3] 分割成功 (Score: {scores[best_idx].item():.2f})")
            return best_mask

        except Exception as e:
            print(f"[SAM3] 推理异常: {e}")
            return None
        finally:
            self._unload_model()