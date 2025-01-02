import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

def get_env_var(key: str, default: str = None) -> str:
    """
    获取环境变量值，如果不存在则抛出异常或返回默认值。
    """
    value = os.getenv(key, default)
    if not value:
        raise EnvironmentError(f"Missing environment variable: {key}")
    return value

# 定义 API Key
DASHSCOPE_API_KEY = get_env_var("DASHSCOPE_API_KEY")
GROQ_API_KEY = get_env_var("GROQ_API_KEY")
DEEPSEEK_API_KEY = get_env_var("DEEPSEEK_API_KEY")
BING_SUBSCRIPTION_KEY = get_env_var("BING_SUBSCRIPTION_KEY")
