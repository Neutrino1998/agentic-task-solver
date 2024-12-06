import os
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from duckduckgo_search.exceptions import RatelimitException
from langchain_core.tools import tool
import time
# Log
from my_logger import Logger, LOG_LEVEL, LOG_PATH, LOG_FILE
# Initialize Logger
logger = Logger(name="AgentLogger", level=LOG_LEVEL, log_path=LOG_PATH, log_file=LOG_FILE)

@tool
def ddg_search_engine(query: str, workspace: dict = {}) -> list[dict]:
    """
    Search the internet using DuckDuckGo Search API.
    *Note: Use `pip install -U duckduckgo_search==5.3.1b1` to somewhat avoid rate limit.

    Args:
        query: The search query.

    Returns:
        The returned search results:
        [{"url":"page link", "title":"page title", "content":"page content", "date":"page date"}, ...]
    """
    max_results = 5
    max_retries = 3
    wait_time = 3 # wait for 1s before retry
    ddg_api = DuckDuckGoSearchAPIWrapper(time='m', source="text") # Time Options: d, w, m, y
    retries = 0
    while retries < max_retries:
        try:
            results = ddg_api.results(query, max_results=max_results)
            return {"result": [{"url": r.get("link"), "title": r.get("title"), "content": r.get("snippet"), "date": r.get("date", "")} for r in results]}
        except RatelimitException:
            retries += 1
            logger.logger.warning((f"Retrying search for '{query}' due to rate limits... (Attempt {retries})"))
            time.sleep(wait_time)  # Wait before retryin
    logger.logger.warning((f"Failed to search for '{query}' after {max_retries} retries."))
    return  {"result": f"Failed to search for '{query}' due to API rate limit. Please try again."}

    

if __name__ == "__main__":
    import asyncio
    # =======================================================
    # Test Example
    print("="*80 + "\n> Testing ddg_search_engine:")
    queries_example = [
                    "Impact of million-plus token context window language models on RAG",
                    "Advantages of large context window models over smaller ones in RAG",
                    "Information retrieval improvement with large context window models",
                ]
    query_results = asyncio.run(ddg_search_engine.abatch(queries_example, return_exceptions=True))
    print(query_results)