import os
import gc
import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class SenseVoiceASR:
    def __init__(self, model_name="iic/SenseVoiceSmall"):
        """
        初始化 SenseVoice ASR 模块。
        model_name: 使用 ModelScope 的模型 ID
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def _load_model(self):
        """动态加载模型到显存"""
        if self.model is None:
            print(f"[SenseVoiceASR] 正在加载模型 {self.model_name} 到 {self.device}...")
            # SenseVoice 默认包含 VAD (语音活动检测)，可以自动切割长音频
            self.model = AutoModel(
                model=self.model_name,
                vad_model="fsmn-vad", # 自动处理长短语音
                device=self.device,
                hub="ms", # 使用 ModelScope 国内镜像，速度更快
                disable_update=True
            )
            print("[SenseVoiceASR] 模型加载完成。")

    def _unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            print("[SenseVoiceASR] 正在释放显存...")
            del self.model
            self.model = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print("[SenseVoiceASR] 显存已清理。")

    def transcribe(self, audio_path: str) -> str:
        """核心转录功能"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"找不到音频文件: {audio_path}")

        try:
            self._load_model()
            print(f"[SenseVoiceASR] 开始识别: {audio_path}")
            
            # 执行推理
            # language="zh" 强制中文识别，也可以设为 "auto"
            res = self.model.generate(
                input=audio_path,
                cache={},
                language="zh", 
                use_itn=True, # 启用反向文本标准化（将“一二三”转为“123”）
                batch_size_s=60,
                merge_vad=True
            )

            # 后处理：SenseVoice 的输出包含标签（如 <|HAPPY|>），需要清理
            # 这里的 res[0]['text'] 就是识别出的富文本
            raw_text = res[0]['text']
            # 使用官方工具清理标签，只保留纯文本
            clean_text = rich_transcription_postprocess(raw_text)
            
            print(f"[SenseVoiceASR] 识别结果: '{clean_text}'")
            return clean_text

        except Exception as e:
            print(f"[SenseVoiceASR] 发生错误: {e}")
            return ""
        finally:
            self._unload_model()

# ==========================================
# 独立测试入口
# ==========================================
if __name__ == "__main__":
    asr = SenseVoiceASR()
    # 替换为你刚才录制的路径
    test_file = "/home/zhn/3d_grasping_project/modules/test_audio2.m4a"
    
    if os.path.exists(test_file):
        print("\n--- SenseVoice 测试开始 ---")
        result = asr.transcribe(test_file)
        print(f"\n最终输出文本: {result}")
    else:
        print(f"找不到测试文件: {test_file}")