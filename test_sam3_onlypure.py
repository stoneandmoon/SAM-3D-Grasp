import cv2
import numpy as np

# 仅导入 SAM3 模块
from modules.sam3_segment import SAM3Segmenter

def visualize_pro(img_path, mask, label):
    """
    专业级 2D 可视化函数，同时保存供下游使用的“纯净二值化掩码”
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 无法读取图像: {img_path}")
        return
            
    # 生成二值化掩码矩阵
    mask_bool = mask > 0.5
    mask_uint8 = (mask_bool).astype(np.uint8) * 255
    
    # ==========================================
    # [核心修改] 保存纯黑白掩码，供后续提取残缺点云使用
    # ==========================================
    pure_mask_name = "pure_duck_mask.png"
    cv2.imwrite(pure_mask_name, mask_uint8)
    print(f"✅ 纯净掩码已保存至: {pure_mask_name} (请将此文件喂给点云提取算法)")
    
    # 生成带颜色的可视化图像
    overlay = img.copy()
    overlay[mask_bool] = [0, 255, 0] # 亮绿色高亮
    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2) 
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500: # 过滤极小噪点
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.rectangle(result, (x, y - 30), (x + w, y), (255, 255, 0), -1)
            cv2.putText(result, label.upper(), (x + 5, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    save_name = "sam3_yellow_duck_test.jpg"
    cv2.imwrite(save_name, result)
    print(f"✅ 可视化结果已存至: {save_name} (仅供人眼观察检查)")


if __name__ == "__main__":
    print("🚀 启动 SAM3 独立测试模块...")
    
    # 1. 设定测试路径与目标词
    test_image = "data/test/000002/rgb/000000.png"
    target_prompt = "yellow duck"
    
    # 2. 初始化 SAM3 (此时不加载其他模型)
    print(f"\n[Step 1] 正在加载 SAM3 模型...")
    sam = SAM3Segmenter()
    
    # 3. 执行文本引导的掩码提取
    print(f"\n[Step 2] 正在图像中提取 '{target_prompt}' 的掩码...")
    mask = sam.segment_by_text(test_image, target_prompt)
    
    # 4. 可视化并保存结果
    if mask is not None:
        print("\n[Step 3] 提取成功，正在生成纯净掩码与可视化图像...")
        visualize_pro(test_image, mask, target_prompt)
    else:
        print(f"\n❌ 分割失败：图中未能找到目标 '{target_prompt}'。")