import os
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from duckduckgo_search.exceptions import RatelimitException
from langchain_core.tools import tool
import time
import requests
# get key
from config import BING_SUBSCRIPTION_KEY
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


class BingSearch:
    """
    具体请参考bing api手册：
    https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/filter-answers
    """
    def __init__(self, search_term: str = 'how is the weather today?', result_count: int = 5, search_sites: list[str] = [], set_freshness: str = ""):
        # 使用环境变量获取密钥
        self.subscription_key = BING_SUBSCRIPTION_KEY
        self.search_url_news = "https://api.bing.microsoft.com/v7.0/news/search"
        self.search_url_web = "https://api.bing.microsoft.com/v7.0/search"
        self.search_term = search_term
        self.count = result_count
        self.sites = search_sites
        self.freshness = set_freshness # Month, Week, Day, YYYY-MM-DD..YYYY-MM-DD

    def make_request(self, url, params):
        """发送API请求并处理HTTP异常"""
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # 抛出HTTP错误
            return response.json()
        except requests.HTTPError as e:
            logger.logger.error(f"⚠️ HTTP error occurred: {e.response.status_code} - {e.response.text}")
            return None
        except requests.RequestException as e:
            logger.logger.error(f"⚠️ Request failed: {e}")
            return None
    
    def news_search(self):
        """执行新闻搜索并返回格式化后的结果"""
        params = {
            "q": self.search_term,
            "textDecorations": True,
            "textFormat": "HTML",
            "freshness": "Month",
            "count": self.count
        }
        results = self.make_request(self.search_url_news, params)
        # return results["value"]
        if results and "value" in results:
            return [{'url': article["url"], 'title': article["name"], 'content': article["description"], 'date': article.get("datePublished", article.get("dateLastCrawled", ""))} for article in results["value"]]
        return []

    def web_search(self):
        """执行Web搜索并返回格式化后的结果"""
        params = {
            "q": self.search_term,
            "textDecorations": True,
            "textFormat": "HTML",
            "freshness": "Month",
            "count": self.count
        }

        results = self.make_request(self.search_url_web, params)
        # return results['webPages']["value"]
        if results and "webPages" in results and "value" in results["webPages"]:
            return [{'url': article["url"], 'title': article["name"], 'content': article["snippet"], 'date': article.get("datePublished", article.get("dateLastCrawled", ""))} for article in results["webPages"]["value"]]
        return []

    def web_search_site(self):
        """执行针对特定站点的Web搜索并返回格式化后的结果"""

        sites = self.sites  
        if isinstance(sites, str):
            sites = [sites]  
        
        # 构建包含多个站点的查询字符串
        if sites:
            sites_query = " OR ".join(f"site:{site}" for site in sites)
            query = f"{self.search_term} {sites_query}"
            
        else:
            query = self.search_term

        if self.freshness == "":
            params = {
                "q": query,
                "textDecorations": True,
                "textFormat": "HTML",
                "count": self.count
            }
        else:
            params = {
                "q": query,
                "textDecorations": True,
                "textFormat": "HTML",
                "freshness": self.freshness,
                "count": self.count
            }
        # print(f"> params: \n{params}")
        results = self.make_request(self.search_url_web, params)
        # return results['webPages']["value"]
        if results and "webPages" in results and "value" in results["webPages"]:
            return [{'url': article["url"], 'title': article["name"], 'content': article["snippet"], 'date': article.get("datePublished", article.get("dateLastCrawled", ""))} for article in results['webPages']["value"]]
        return []

@tool
def bing_search_engine(query: str, workspace: dict = {}) -> list[dict]:
    """
    A search tool that uses the Bing search engine to find relevant information on the web.

    Args:
        query: The search query.

    Returns:
        The returned search results:
        [{"url":"page link", "title":"page title", "content":"page content", "date":"page date"}, ...]
    """
    max_results = 5
    bingsearch = BingSearch(query, result_count=max_results)
    search_result = bingsearch.web_search_site()
    return {"result": search_result}

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
    query_results = asyncio.run(bing_search_engine.abatch(queries_example, return_exceptions=True))
    print(query_results)