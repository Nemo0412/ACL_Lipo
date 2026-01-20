"""
配置文件 - HuggingFace API
"""
import os

# API配置
API_BASE = "https://router.huggingface.co/v1"
API_KEY = os.environ.get("HF_TOKEN", "your-hf-token-here")  # 从环境变量读取

# 模型配置
MODEL = "Qwen/Qwen3-32B:groq"

# 预测参数
TEMPERATURE = 0.7  # 生成温度，范围0-1，较低值更确定，较高值更随机
MAX_TOKENS = 50  # 最大生成token数
MAX_RETRIES = 3  # 最大重试次数

# 批处理参数
BATCH_SIZE = 10  # 每批次处理的药物数量（用于中间保存）
DELAY_BETWEEN_REQUESTS = 0.5  # 请求之间的延迟（秒），避免API限流
