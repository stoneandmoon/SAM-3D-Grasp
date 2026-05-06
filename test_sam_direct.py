#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os

# 导入你 Pipeline 里的核心视觉模块
from modules.sam3_segment import SAM3Segmenter
from modules.sam3d_engine import SAM3DEngine

def main():
    # 1. 直接指定目标图片和提示词（绕过语音和 DeepSeek）
    image_file = "./ho3d_test/train/ABF13/rgb/0000.jpg"
    target_text = "bleach cleanser"  # 直接硬编码要抓取的物体
    
    print(f"[1] 初始化 SAM3，准备在图像中寻找: '{target_text}'...")
    sam = SAM3Segmenter()
    sam3d = SAM3DEngine()
    
    # 2. 调用 SAM3 提取 2D 掩码
    print("[2] 正在提取 2D Mask...")
    mask = sam.segment_by_text(image_file, target_text)
    
    if mask is None:
        print("❌ 提取失败：图中未找到物体！")
        return
        
    print("✅ 掩码提取成功！正在保存 2D 可视化结果...")
    # 简单可视化一下掩码，确保扣对了
    img = cv2.imread(image_file)
    overlay = img.copy()
    overlay[mask > 0.5] = [0, 255, 0]
    res_img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite("./ho3d_test/direct_sam3_mask.jpg", res_img)
    
    # 释放显存，给 SAM3D 腾地方
    print("[3] 卸载 SAM3 释放显存...")
    sam._unload_model()
    
    # 3. 调用 SAM3D 引擎生成预测的残缺点云
    print("[4] 启动 SAM3D 引擎，生成 3D 残缺点云...")
    # 注意：这里需要根据你 SAM3DEngine 实际的保存路径来找生成的模型
    success = sam3d.generate_3d(image_file, mask)
    
    if success:
        print("🎉 单点测试完成！SAM3D 成功生成了残缺点云！")
    else:
        print("❌ SAM3D 生成失败！")

if __name__ == "__main__":
    main()
