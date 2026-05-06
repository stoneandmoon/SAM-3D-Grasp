import os
import sys
import torch
import json

# ==========================================
# 🛡️ Dynamo 隔离墙 (Torch Compile 修复)
# 彻底禁用 Dynamo 编译加速，强制退回安全稳定的 Eager 模式
# ==========================================
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

import numpy as np
import cv2
from PIL import Image
import gc
from omegaconf import OmegaConf
from hydra.utils import instantiate
import trimesh
import open3d as o3d

# 确保 SAM_3D 的代码库在环境变量中
SAM3D_REPO_BASE = "/root/SAM-3D-Grasp"
if SAM3D_REPO_BASE not in sys.path:
    sys.path.append(SAM3D_REPO_BASE)


class SAM3DEngine:
    def __init__(self, ckpt_dir="/root/SAM-3D-Grasp/sam3d_checkpoints"):
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
            cfg.ss_generator_ckpt_path = os.path.join(self.ckpt_dir, cfg.ss_generator_ckpt_path)
            cfg.slat_generator_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_generator_ckpt_path)
            cfg.ss_decoder_ckpt_path = os.path.join(self.ckpt_dir, cfg.ss_decoder_ckpt_path)
            cfg.slat_decoder_gs_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_decoder_gs_ckpt_path)
            cfg.slat_decoder_mesh_ckpt_path = os.path.join(self.ckpt_dir, cfg.slat_decoder_mesh_ckpt_path)

            # 3. 实例化 Pipeline
            self.pipeline = instantiate(cfg)

            print("[SAM-3D] ✅ 3D 流水线加载完毕！")

    def _unload_model(self):
        """严格的显存清理，防止 OOM"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[SAM-3D] ♻️ 3D 模型已卸载，显存已释放。")

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _export_clean_mesh(raw_mesh, output_dir):
        """
        导出清洗后的 mesh:
        - reconstructed_mesh.obj
        - reconstructed_mesh.ply
        """
        clean_mesh = None

        # 情况 A：原始对象带 vertices / faces
        if hasattr(raw_mesh, "vertices") and hasattr(raw_mesh, "faces"):
            verts = raw_mesh.vertices.detach().cpu().numpy() if torch.is_tensor(raw_mesh.vertices) else np.asarray(raw_mesh.vertices)
            faces = raw_mesh.faces.detach().cpu().numpy() if torch.is_tensor(raw_mesh.faces) else np.asarray(raw_mesh.faces)
            clean_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        # 情况 B：已经是 trimesh 对象或 scene
        elif hasattr(raw_mesh, "export"):
            if isinstance(raw_mesh, trimesh.Scene):
                dumped = raw_mesh.dump()
                if isinstance(dumped, list) and len(dumped) > 0:
                    clean_mesh = trimesh.util.concatenate(dumped)
                else:
                    clean_mesh = None
            else:
                clean_mesh = raw_mesh

            if clean_mesh is not None and hasattr(clean_mesh, "process"):
                clean_mesh.process()

        if clean_mesh is None:
            print(f"[SAM-3D] ⚠️ 发现 glb 数据，但格式无法解析: {type(raw_mesh)}")
            return None

        obj_path = os.path.join(output_dir, "reconstructed_mesh.obj")
        ply_path = os.path.join(output_dir, "reconstructed_mesh.ply")

        clean_mesh.export(obj_path, file_type="obj")
        clean_mesh.export(ply_path, file_type="ply")

        print(f"[SAM-3D] 💾 3D 几何模型已深度清洗并保存至: {obj_path}")
        print(f"[SAM-3D] 💾 3D 几何模型已同步保存至: {ply_path}")

        return clean_mesh

    @staticmethod
    def _export_pose_json(results, output_dir):
        pose_data = {}

        if "translation" in results and results["translation"] is not None:
            tr = SAM3DEngine._to_numpy(results["translation"])
            pose_data["translation"] = tr.reshape(-1).tolist()

        if "scale" in results and results["scale"] is not None:
            sc = SAM3DEngine._to_numpy(results["scale"])
            pose_data["scale"] = sc.reshape(-1).tolist()

        if "rotation" in results and results["rotation"] is not None:
            rq = SAM3DEngine._to_numpy(results["rotation"])
            pose_data["rotation_quat"] = rq.reshape(-1).tolist()

        pose_path = os.path.join(output_dir, "sam3d_pose.json")
        with open(pose_path, "w", encoding="utf-8") as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

        print(f"[SAM-3D] 💾 解析视位姿数据已保存至: {pose_path}")
        return pose_path

    @staticmethod
    def _export_pointmap(results, output_dir):
        """
        导出:
        - sam_pointmap.npy
        - sam_pointmap_colors.npy (如果有)
        - sam_partial_rgb.ply
        """
        if "pointmap" not in results or results["pointmap"] is None:
            print("[SAM-3D] ⚠️ 当前结果中没有 pointmap，跳过 pointmap 导出。")
            return None

        pm = SAM3DEngine._to_numpy(results["pointmap"])
        if pm.ndim == 4:
            pm = pm[0]

        if pm.ndim != 3 or pm.shape[-1] != 3:
            raise RuntimeError(f"pointmap 形状异常: {pm.shape}")

        pointmap_npy_path = os.path.join(output_dir, "sam_pointmap.npy")
        np.save(pointmap_npy_path, pm)
        print(f"[SAM-3D] 💾 pointmap numpy 已保存至: {pointmap_npy_path}")

        colors = None
        if "pointmap_colors" in results and results["pointmap_colors"] is not None:
            colors = SAM3DEngine._to_numpy(results["pointmap_colors"])
            if colors.ndim == 4:
                colors = colors[0]
            if colors.shape[:2] == pm.shape[:2]:
                color_npy_path = os.path.join(output_dir, "sam_pointmap_colors.npy")
                np.save(color_npy_path, colors)
                print(f"[SAM-3D] 💾 pointmap 颜色已保存至: {color_npy_path}")
            else:
                colors = None

        valid = np.isfinite(pm).all(axis=-1)

        pts = pm[valid].reshape(-1, 3)
        if len(pts) == 0:
            print("[SAM-3D] ⚠️ pointmap 有效点为空，跳过 sam_partial_rgb.ply 导出。")
            return None

        # 去掉全零点和极小噪声点
        norms = np.linalg.norm(pts, axis=1)
        keep = norms > 1e-8
        pts = pts[keep]

        if len(pts) == 0:
            print("[SAM-3D] ⚠️ pointmap 去噪后为空，跳过 sam_partial_rgb.ply 导出。")
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

        if colors is not None:
            cols = colors[valid].reshape(-1, 3)
            cols = cols[keep]
            cols = np.clip(cols, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

        partial_ply = os.path.join(output_dir, "sam_partial_rgb.ply")
        o3d.io.write_point_cloud(partial_ply, pcd)
        print(f"[SAM-3D] 💾 RGB 视角 partial 点云已保存至: {partial_ply}")

        return partial_ply

    def generate_3d(self, image_path: str, mask: np.ndarray, output_dir: str = "./output_3d"):
        """
        核心方法：输入原图和 2D 掩码，输出：
        - 完整 3D mesh
        - sam3d_pose.json
        - sam_pointmap.npy
        - sam_partial_rgb.ply
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            self._load_model()

            print("[SAM-3D] ⚙️ 正在执行单目深度估计与 3D 反投影...")

            mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_uint8)
            image_pil = Image.open(image_path).convert("RGB")

            with torch.no_grad():
                results = self.pipeline.run(
                    image=image_pil,
                    mask=mask_pil,
                    with_texture_baking=False,
                    with_layout_postprocess=False
                )

                # 1) 导出 mesh
                if "glb" in results and results["glb"] is not None:
                    try:
                        self._export_clean_mesh(results["glb"], output_dir)
                    except Exception as export_e:
                        print(f"[SAM-3D] ❌ 网格清洗与导出失败: {export_e}")
                else:
                    print("[SAM-3D] ⚠️ 当前结果中没有 glb，跳过完整 mesh 导出。")

                # 2) 导出 pose json
                self._export_pose_json(results, output_dir)

                # 3) 导出 pointmap / sam_partial_rgb
                self._export_pointmap(results, output_dir)

            print("[SAM-3D] 🎉 3D 重建流程全部闭环！")
            return True

        except Exception as e:
            import traceback
            print(f"[SAM-3D] ❌ 3D 重建过程崩溃: {e}")
            traceback.print_exc()
            return False
        finally:
            self._unload_model()