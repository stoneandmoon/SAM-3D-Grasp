import os
from openai import OpenAI

class SemanticParser:
    def __init__(self, api_key):
        """
        初始化 DeepSeek 语义解析模块。
        """
        self.client = OpenAI(
            api_key="sk-872267e1032c4cfc9a592552223f879b", 
            base_url="https://api.deepseek.com"
        )
        self.model_name = "deepseek-chat" 

    def extract_target(self, raw_instruction: str) -> str:
        if not raw_instruction:
            return ""

        # 核心更新：要求模型输出英文短语，这是视觉模型最喜欢的格式
        system_prompt = """
        Task: You are a robotic vision instruction parser.
        Goal: Extract the "target object" and its "physical attributes" from the user's Chinese speech.
        Requirements:
        1. Correct speech recognition errors (e.g., "被子" on a table should be "cup").
        2. Translate the target into a concise English noun phrase.
        3. Remove all verbs (pick up, grab), stop words, and punctuation.
        4. Output ONLY the English phrase (e.g., "red cup", "blue phone").
        
        Examples:
        Input: "帮我抓一下那个红色的被子。" -> Output: "red cup"
        Input: "请把蓝色的手机拿给我。" -> Output: "blue phone"
        """

        try:
            print("[SLM] 正在请求 DeepSeek 进行语义跨语言解构...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_instruction}
                ],
                temperature=0.1,
                stream=False
            )
            
            clean_target_en = response.choices[0].message.content.strip().lower()
            # 去掉可能存在的末尾句号
            clean_target_en = clean_target_en.replace(".", "")
            
            print(f"[SLM] 原始指令: 【{raw_instruction}】")
            print(f"[SLM] 提纯后(EN): 【{clean_target_en}】")
            return clean_target_en

        except Exception as e:
            print(f"[SLM] DeepSeek 解析失败: {e}")
            return "object" # 兜底策略