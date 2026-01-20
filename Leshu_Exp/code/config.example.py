"""
配置文件示例
复制此文件为 config.py 并填入你的实际配置
"""

# API配置
API_BASE = "https://api.example.com/v1"  # 你的API基础地址
API_KEY = "your-api-key-here"  # 你的API密钥

# 模型配置
MODEL = "Qwen/Qwen2.5-32B-Instruct"  # 如果使用Qwen3-32B，请修改为相应的模型名称

# 预测参数
TEMPERATURE = 0.7  # 生成温度，范围0-1，较低值更确定，较高值更随机
MAX_TOKENS = 50  # 最大生成token数
MAX_RETRIES = 3  # 最大重试次数

# 批处理参数
BATCH_SIZE = 10  # 每批次处理的药物数量（用于中间保存）
DELAY_BETWEEN_REQUESTS = 0.5  # 请求之间的延迟（秒），避免API限流
