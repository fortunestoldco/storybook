import os
import asyncio
from typing import List, Dict, Any, Optional

from exa_py import Exa 
from tavily import AsyncTavilyClient
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langsmith import traceable

async def select_and_execute_search(search_api: str, queries: List[str], params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute searches using the specified API."""
    if search_api == "tavily":
        return await tavily_search_async(queries, **params)
    elif search_api == "perplexity":
        return perplexity_search(queries, **params)  
    elif search_api == "exa":
        return await exa_search(queries, **params)
    elif search_api == "arxiv":
        return await arxiv_search_async(queries, **params)
    elif search_api == "pubmed":
        return await pubmed_search_async(queries, **params)
    else:
        raise ValueError(f"Unsupported search API: {search_api}")

@traceable
async def tavily_search_async(queries: List[str], **kwargs):
    """Execute searches using Tavily API."""
    client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    search_tasks = []
    for query in queries:
        search_tasks.append(
            client.search(query, **kwargs)
        )
    
    return await asyncio.gather(*search_tasks)

# ... Implement other search functions from RESEARCHER-TO-INTEGRATE/utils.py ...