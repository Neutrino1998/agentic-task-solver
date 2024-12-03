import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import DashScopeEmbeddings
# get key
from config import DASHSCOPE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY
# Log
from my_logger import Logger, LOG_LEVEL, LOG_PATH, LOG_FILE
# Initialize Logger
logger = Logger(name="AgentLogger", level=LOG_LEVEL, log_path=LOG_PATH, log_file=LOG_FILE)

def get_llm(model_name: str="qwen2.5-72b-instruct"):
    logger.logger.debug(f"[LLM] Initializing Model: '{model_name}'...")
    if model_name == 'deepseek-chat':
        return ChatOpenAI(
                model='deepseek-chat', 
                openai_api_key=DEEPSEEK_API_KEY, 
                openai_api_base='https://api.deepseek.com',
                max_tokens=1024
            )
    elif model_name == 'llama-3.1-70b-versatile':
        return ChatGroq(
                model='llama-3.1-70b-versatile',
                temperature=0.0,
                max_retries=2,
                # other params...
            )
    else:
        return ChatTongyi(model=model_name)

def get_embeddings(embedding_name: str="text-embedding-v1"):
    logger.logger.debug(f"[EMBEDDING] Initializing Embedding: '{embedding_name}'...")
    return DashScopeEmbeddings(
            model=embedding_name, dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    # =======================================================
    # 测试用例
    test_llm = get_llm(
    # model_name="qwen-turbo",
    # model_name="qwen-plus",
    # model_name="qwen2-72b-instruct",
    model_name="qwen2.5-72b-instruct",
    # model_name="qwen-max"
    )
    query = """
    LangGraph的功能是什么?
    """
    print("> Running:", test_llm.model_name)
    res = test_llm.stream([HumanMessage(content=query)], streaming=True)
    for chunks in res:
        print(chunks.content, end="")