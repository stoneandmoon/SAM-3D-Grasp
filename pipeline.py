import os
import cv2
import numpy as np

# 导入你的核心感知与 3D 模块
from modules.asr_sensevoice import SenseVoiceASR
from modules.slm_parser import SemanticParser
from modules.sam3_segment import SAM3Segmenter
from modules.sam3d_engine import SAM3DEngine

class GraspingPipeline:
    def __init__(self, ds_key):
        """
        初始化端到端具身智能流水线。
        注意：此处仅实例化类，不加载大模型权重，确保 11GB 显存绝对安全。
        """
        self.asr = SenseVoiceASR()
        self.slm = SemanticParser(api_key=ds_key)
        self.sam = SAM3Segmenter()
        self.sam3d = SAM3DEngine()

    def run(self, audio_file, image_file):
        print("\n" + "█"*60)
        print("�� 具身智能 3D 抓取系统：全链路启动")
        print("█"*60)

        # ---------------------------------------------------------
        # 阶段 1: 语音转文字
        # ---------------------------------------------------------
        print("\n[Step 1/4] �� 正在处理语音指令...")
        raw_speech = self.asr.transcribe(audio_file)
        if not raw_speech:
            print("❌ 语音识别失败，流水线终止。")
            return

        # ---------------------------------------------------------
        # 阶段 2: 语义解构 (DeepSeek 中英转换)
        # ---------------------------------------------------------
        print("\n[Step 2/4] �� 正在通过 DeepSeek 解析语义...")
        target_en = self.slm.extract_target(raw_speech)
        if not target_en:
            print("❌ 语义解析失败，流水线终止。")
            return

        # ---------------------------------------------------------
        # 阶段 3: 2D 视觉分割 (SAM 3)
        # ---------------------------------------------------------
        print(f"\n[Step 3/4] ��️ 正在图像中定位目标: '{target_en}'...")
        mask = self.sam.segment_by_text(image_file, target_en)
        
        if mask is None:
            print(f"❌ 视觉识别失败：图中未找到 '{target_en}'，流水线终止。")
            return

        # 生成 2D 专业可视化结果
        self._visualize_pro(image_file, mask, target_en)

        # =========================================================
        # ⚠️ 显存隔离墙 (VRAM Isolation Wall) ⚠️
        # 必须在这里把 SAM 3 彻底赶出显存，为 10GB 的 SAM-3D 腾出空间
        # =========================================================
        print("\n[Memory System] �� 触发显存隔离墙：强制清空前置感知模型...")
        self.sam._unload_model() 
        
        # ---------------------------------------------------------
        # 阶段 4: 3D 重建 (SAM-3D + MoGe 深度估计)
        # ---------------------------------------------------------
        print(f"\n[Step 4/4] �� 开始将 '{target_en}' 重建为 3D 模型...")
        success = self.sam3d.generate_3d(image_file, mask)
        
        if success:
            print("\n" + "█"*60)
            print("�� 恭喜！端到端具身智能抓取视觉系统跑通全流程！")
            print("█"*60)
        else:
            print("\n❌ 3D 重建失败，请检查报错日志。")


    def _visualize_pro(self, img_path, mask, label):
        """
        专业级 2D 结果可视化：高亮掩码 + 红色轮廓 + 青色 BBox
        """
        img = cv2.imread(img_path)
        if img is None: return
            
        mask_bool = mask > 0.5
        mask_uint8 = (mask_bool).astype(np.uint8) * 255
        
        overlay = img.copy()
        overlay[mask_bool] = [0, 255, 0] # 亮绿色
        result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2) 
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > 500:
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.rectangle(result, (x, y - 30), (x + w, y), (255, 255, 0), -1)
                cv2.putText(result, label.upper(), (x + 5, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        save_name = "pipeline_output_2D_result.jpg"
        cv2.imwrite(save_name, result)
        print(f"[Pipeline] �� 2D 定位结果图已存至: {save_name}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # �� 用户配置区
    # ---------------------------------------------------------
    # 请填入你从 DeepSeek 官网获取的真实 API Key
    DEEPSEEK_KEY = "sk-872267e1032c4cfc9a592552223f879b" 
    
    # 测试文件路径
    TEST_AUDIO = "/home/zhn/3d_grasping_project/modules/test_audio1.m4a"
    TEST_IMAGE = "/home/zhn/3d_grasping_project/data/test_rgb_image.jpeg"
    # ---------------------------------------------------------

    # 启动流水线
    pipeline = GraspingPipeline(ds_key=DEEPSEEK_KEY)
    pipeline.run(TEST_AUDIO, TEST_IMAGE)