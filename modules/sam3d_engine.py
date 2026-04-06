import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import gc
from omegaconf import OmegaConf
from hydra.utils import instantiate

# 确保 SAM_3D 的代码库在环境变量中
SAM3D_REPO_BASE = "/home/zhn/SAM_3D"
if SAM3D_REPO_BASE not in sys.path:
    sys.path.append(SAM3D_REPO_BASE)

class SAM3DEngine:
    def __init__(self, ckpt_dir="/home/zhn/SAM_3D/sam3d_checkpoints"):
        self.ckpt_dir = ckpt_dir
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """利用 Hydra 和 OmegaConf 动态实例化 3D 流水线"""
        if self.pipeline is None:
            print("[SAM-3D] ⏳ 正在加载 10GB 级 3D 重建大模型 (这可能需要一分钟)...")
            
            # 1. 加载 YAML 配置
            config_path = os.path.join(self.ckpt_dir, "pipeline.yaml")
            cfg = OmegaConf.load(config_path)
            
            # 2. 动态修改配置中的路径，使其指向绝对路径
            # 防止因为运行目录不同导致找不到 .ckpt 文件
            cfg.ss_generator_ckpt_path = os.path.join(self.ckpt_dir, cfg.ss_generator_ckpt_path)
            cfg.slat_generator_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_generator_ckpt_path)
            cfg.ss_decoder_ckpt_path = os.path.join(self.ckpt_dir, cfg.ss_decoder_ckpt_path)
            cfg.slat_decoder_gs_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_decoder_gs_ckpt_path)
            cfg.slat_decoder_mesh_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_decoder_mesh_ckpt_path)
            
            # 3. 实例化 Pipeline
            # 这一步会把 MoGe 深度模型和所有 Generator/Decoder 加载进显存
            self.pipeline = instantiate(cfg).to(self.device)
            self.pipeline.eval()
            print("[SAM-3D] ✅ 3D 流水线加载完毕！")

    def _unload_model(self):
        """严格的显存清理，防止 OOM"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[SAM-3D] �� 3D 模型已卸载，显存已释放。")

    def generate_3d(self, image_path: str, mask: np.ndarray, output_dir: str = "./output_3d"):
        """
        核心方法：输入原图和 2D 掩码，输出 3D 模型
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self._load_model()
            
            print("[SAM-3D] ⚙️ 正在执行单目深度估计与 3D 反投影...")
            
            # 将 numpy mask (H, W) 转换为 PIL Image 以适配 SAM-3D 的预处理
            # 确保掩码是 0 和 255 的格式
            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_uint8)
            
            image_pil = Image.open(image_path).convert("RGB")
            
            # 执行推理 (具体调用方法取决于 InferencePipelinePointMap 的实现)
            # 大多数此类 Pipeline 接收 image 和 mask，并自动保存到指定目录
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    # 注意：如果此处报错，我们需要查看 InferencePipelinePointMap 的源码
                    # 通常是 __call__ 或 forward 方法
                    results = self.pipeline(image=image_pil, mask=mask_pil)
                    
                    # 假设 pipeline 提供了一个保存方法
                    if hasattr(self.pipeline, 'save'):
                        self.pipeline.save(results, output_dir)
                    else:
                        print(f"[SAM-3D] ⚠️ 推理完成，请手动检查返回的 results: {type(results)}")

            print(f"[SAM-3D] �� 3D 重建成功！结果保存在: {output_dir}")
            return True

        except Exception as e:
            print(f"[SAM-3D] ❌ 3D 重建过程崩溃: {e}")
            return False
        finally:
            self._unload_model()