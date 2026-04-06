import os
import gc
import torch
import whisper

class WhisperASR:
    def __init__(self, model_size="small"):
        """
        初始化 Whisper 语音识别模块。
        可选 model_size: 'tiny', 'base', 'small', 'medium', 'large'
        出于显存限制的考虑，默认使用 'small' 模型。
        """
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 初始状态下不加载模型，避免长期占用显存
        self.model = None 

    def _load_model(self):
        """内部方法：动态加载模型到显存"""
        if self.model is None:
            print(f"[WhisperASR] 正在加载 {self.model_size} 模型到 {self.device}...")
            # 加载模型，FP16 精度可以进一步节省显存
            self.model = whisper.load_model(self.model_size, device=self.device)
            print("[WhisperASR] 模型加载完成。")

    def _unload_model(self):
        """内部方法：卸载模型并深度清理显存碎片"""
        if self.model is not None:
            print("[WhisperASR] 正在卸载模型释放显存...")
            del self.model
            self.model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[WhisperASR] 显存清理完毕。")

    def transcribe(self, audio_path: str) -> str:
        """
        核心功能：将音频文件转换为文本。
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"找不到音频文件: {audio_path}")

        try:
            # 1. 加载模型
            self._load_model()
            
            # 2. 执行推理转录
            print(f"[WhisperASR] 开始转录音频: {audio_path}")
            # fp16=True 确保在支持的 GPU 上以半精度运行
            result = self.model.transcribe(audio_path, fp16=torch.cuda.is_available())
            
            text_output = result["text"].strip()
            print(f"[WhisperASR] 转录成功。识别结果: '{text_output}'")
            
            return text_output

        except Exception as e:
            print(f"[WhisperASR] 转录过程中发生错误: {e}")
            return ""
            
        finally:
            # 3. 无论是否发生异常，强制卸载模型释放 11GB 有限的显存池
            self._unload_model()

# ==========================================
# 模块独立测试入口
# ==========================================
if __name__ == "__main__":
    # 这个测试代码块只有在直接运行此文件时才会执行，方便你独立调试
    asr_module = WhisperASR(model_size="small")
    
    # 请确保同级目录下有一个测试音频文件，你可以自己录一段 .wav 或 .mp3
    test_audio_file = "/home/zhn/3d_grasping_project/modules/test_audio1.m4a" 
    
    # 创建一个空的测试音频文件以防止 os.path.exists 报错（仅供占位）
    if not os.path.exists(test_audio_file):
        with open(test_audio_file, "wb") as f:
            pass
        print(f"提示: 已生成空的占位文件 {test_audio_file}，请替换为真实的音频文件进行测试！")

    print("\n--- 开始模块独立测试 ---")
    try:
        result_text = asr_module.transcribe(test_audio_file)
        print(f"\n最终输出文本: {result_text}")
    except Exception as e:
        print(f"测试失败: {e}")