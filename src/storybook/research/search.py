import os
import asyncio
from typing import List, Dict, Any, Optional

from exa_py import Exa 
from tavily import AsyncTavilyClient
from langchain_community.retrievers import ArxivRetriever
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from langsmith import traceable

async def select_and_execute_search(search_api: str, query_list: list[str], params_to_pass: dict) -> List[Dict[str, Any]]:
    """Execute searches using the specified API."""
    if search_api == "tavily":
        return await tavily_search_async(query_list, **params_to_pass)
    elif search_api == "perplexity":
        return perplexity_search(query_list, **params_to_pass)  
    elif search_api == "exa":
        return await exa_search(query_list, **params_to_pass)
    elif search_api == "arxiv":
        return await arxiv_search_async(query_list, **params_to_pass)
    elif search_api == "pubmed":
        return await pubmed_search_async(query_list, **params_to_pass)
    elif search_api == "linkup":
        return await linkup_search(query_list, **params_to_pass)
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

async def exa_search(queries: List[str], **kwargs):
    """Execute searches using Exa API."""
    client = Exa(api_key=os.getenv("EXA_API_KEY"))
    
    search_tasks = []
    for query in queries:
        search_tasks.append(
            client.search(query, **kwargs)
        )
    
    return await asyncio.gather(*search_tasks)

# Define a basic perplexity search function 
def perplexity_search(queries: List[str], **kwargs):
    """Execute searches using Perplexity API."""
    # This would be implemented with the Perplexity API
    results = []
    for query in queries:
        results.append({
            "query": query,
            "results": [
                {
                    "title": f"Result for {query}",
                    "url": "https://example.com",
                    "content": f"This is a placeholder result for: {query}",
                    "score": 0.9
                }
            ]
        })
    return results

@traceable
async def arxiv_search_async(queries: List[str], load_max_docs=5, **kwargs):
    """Execute searches on arXiv."""
    # Initialize a retriever for arXiv
    retriever = ArxivRetriever(load_max_docs=load_max_docs)
    
    # Process each query
    results = []
    for query in queries:
        docs = retriever.get_relevant_documents(query)
        
        query_results = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            query_results.append({
                "title": metadata.get("Title", ""),
                "url": metadata.get("entry_id", ""),
                "content": doc.page_content,
                "score": 1.0 - (i * 0.1)  # Simulate decreasing relevance
            })
        
        results.append({
            "query": query,
            "results": query_results
        })
    
    return results

@traceable
async def pubmed_search_async(queries: List[str], **kwargs):
    """Execute searches on PubMed."""
    # Initialize PubMed API wrapper
    pubmed = PubMedAPIWrapper()
    
    # Process each query
    results = []
    for query in queries:
        docs = pubmed.run(query)
        
        query_results = []
        for i, doc in enumerate(docs):
            query_results.append({
                "title": doc.get("title", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc.get('uid', '')}/",
                "content": doc.get("summary", ""),
                "score": 1.0 - (i * 0.1)  # Simulate decreasing relevance
            })
        
        results.append({
            "query": query,
            "results": query_results
        })
    
    return results

@traceable
async def linkup_search(queries: List[str], **kwargs):
    """Execute searches using Linkup."""
    # This would be implemented with the Linkup API
    results = []
    for query in queries:
        results.append({
            "query": query,
            "results": [
                {
                    "title": f"Linkup result for {query}",
                    "url": "https://example.com",
                    "content": f"This is a placeholder Linkup result for: {query}",
                    "score": 0.9
                }
            ]
        })
    return results