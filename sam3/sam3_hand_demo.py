#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def run_sam3_hand(
    image_path,
    out_mask_path,
    out_overlay_path=None,
    text_prompt="hand",      # 尽量简单：hand
    score_thresh=0.15,       # 自己的置信度阈值，先设低一点
):
    """
    用 SAM3 + 文本提示分割“手”，把所有 hand 实例的 mask 合并为一张 0/255 掩码。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    # 1. 加载 SAM3 模型
    model = build_sam3_image_model().to(device)

    # 关闭 Sam3Processor 内部的置信度截断，让它把所有候选都给出来
    processor = Sam3Processor(model, confidence_threshold=0.0)
    model.eval()

    # 2. 读图
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"[INFO] 输入图像尺寸: {w}x{h}")
    print(f"[INFO] 使用文本提示: {text_prompt!r}")

    # 3. 设置图像并用文本 prompt
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output["masks"]     # [N, 1, H, W] 或 [N, H, W]
    scores = output["scores"]   # [N]

    num_inst = masks.shape[0]
    print(f"[INFO] SAM3 返回 {num_inst} 个实例")

    if num_inst == 0 or masks.numel() == 0:
        print("[WARN] 没有任何 mask，输出全黑图")
        empty = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(out_mask_path, empty)
        return

    # 4. 用我们自己的阈值过滤实例
    scores_np = scores.detach().cpu().numpy()
    print("[INFO] 所有实例 scores:", scores_np)

    keep_idx = np.where(scores_np >= score_thresh)[0]
    print(f"[INFO] 分数 >= {score_thresh} 的实例索引: {keep_idx}")

    if len(keep_idx) == 0:
        print(f"[WARN] 所有 mask 置信度 < {score_thresh}，直接用全部 {num_inst} 个 mask 合并")
        keep_idx = np.arange(num_inst)

    masks_np = masks.detach().cpu().numpy()
    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0, :, :]

    combined = np.zeros_like(masks_np[0], dtype=np.float32)
    for i in keep_idx:
        combined = np.maximum(combined, masks_np[i])

    # 5. 二值化成 0/255 手掩码
    hand_mask = (combined > 0.5).astype(np.uint8) * 255
    cv2.imwrite(out_mask_path, hand_mask)
    print(f"[INFO] 保存手掩码: {out_mask_path}")

    # 6. 可选：保存叠加可视化
    if out_overlay_path is not None:
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("[WARN] 打不开原图，跳过 overlay 保存")
            return
        color = np.array([0, 0, 255], dtype=np.uint8)[None, None, :]
        mask_bool = hand_mask.astype(bool)
        overlay = img_bgr.copy()
        overlay[mask_bool] = overlay[mask_bool] * 0.4 + color * 0.6
        cv2.imwrite(out_overlay_path, overlay)
        print(f"[INFO] 保存叠加图: {out_overlay_path}")


if __name__ == "__main__":
    img_path = "/home/zhn/下载/ho3dmini/HO3D_v3/train/SS1/rgb/0500.jpg"
    out_mask = "/home/zhn/下载/ho3d_sam3_hand_demo_mask_0500.png"
    out_overlay = "/home/zhn/下载/ho3d_sam3_hand_demo_overlay_0500.png"

    os.makedirs(os.path.dirname(out_mask), exist_ok=True)
    run_sam3_hand(img_path, out_mask, out_overlay)
