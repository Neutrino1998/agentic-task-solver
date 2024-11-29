import os
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

@tool
def ddg_search_engine(query: str) -> list[dict]:
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
    ddg_api = DuckDuckGoSearchAPIWrapper(time='m', source="text") # Time Options: d, w, m, y
    results = ddg_api.results(query, max_results=max_results)
    return [{"url": r.get("link"), "title": r.get("title"), "content": r.get("snippet"), "date": r.get("date", "")} for r in results]

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