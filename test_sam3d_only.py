#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import open3d as o3d

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl" 
sys.path.insert(0, os.path.abspath("."))

from modules.sam3d_engine import SAM3DEngine

if __name__ == "__main__":
    print("🚀 启动 SAM-3D 单一模型提取模块...")

    image_file = "data/test/000002/rgb/000000.png"
    mask_file = "pure_duck_mask.png"

    mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        print("❌ 无法读取掩码文件")
        sys.exit(1)
        
    mask_bool = mask_img > 127 

    print("\n[Step 1] 正在生成 3D 模型...")
    sam3d = SAM3DEngine()
    success = sam3d.generate_3d(image_file, mask_bool)

    if success:
        raw_obj_path = "./output_3d/reconstructed_mesh.obj"
        
        print("\n[Step 2] 正在清洗繁杂文件，提取单一纯净点云/网格...")
        # 在服务器上用 Open3D 读取那个杂乱的 obj
        mesh = o3d.io.read_triangle_mesh(raw_obj_path)
        # 重新计算法线，确保模型有立体光影
        mesh.compute_vertex_normals()
        
        # 强制保存为单一的 ply 文件放在根目录
        clean_ply_path = "sam3d_duck_clean.ply"
        o3d.io.write_triangle_mesh(clean_ply_path, mesh)
        
        print("\n" + "█"*50)
        print(f"✅ 提取成功！现在你只需要拉回这一个文件: {clean_ply_path}")
        print("█"*50)
    else:
        print("\n❌ 3D 重建失败。")