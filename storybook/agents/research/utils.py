import os
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
from firecrawl import FireCrawler
from pymongo import MongoClient
from langchain.embeddings import OpenAIEmbeddings

class ResearchUtils:
    def __init__(self, config: Dict[str, Any]):
        self.tavily_client = TavilyClient(api_key=config["tavily_api_key"])
        self.mongo_client = MongoClient(config["mongo_uri"])
        self.fire_crawler = FireCrawler()
        self.embeddings = OpenAIEmbeddings()
        self.db = self.mongo_client.research_db
        
    @traceable
    async def search_and_store(self, query: str, collection_name: str) -> Dict[str, Any]:
        # Search with Tavily
        search_results = await self.tavily_client.search(query)
        
        # Crawl pages
        enriched_results = []
        for result in search_results:
            content = await self.fire_crawler.crawl(result["url"])
            if content:
                # Generate embeddings
                embedding = self.embeddings.embed_query(content)
                
                # Store in MongoDB
                doc = {
                    "url": result["url"],
                    "title": result["title"],
                    "content": content,
                    "embedding": embedding,
                    "query": query
                }
                self.db[collection_name].insert_one(doc)
                enriched_results.append(doc)
                
        return {"results": enriched_results}

    def analyze_coverage(self, results: List[Dict[str, Any]], llm) -> Dict[str, Any]:
        system_prompt = """Analyze the research results and identify:
        1. Key findings
        2. Missing information
        3. Potential biases
        4. Suggested additional research areas"""
        
        content = "\n".join([r["content"] for r in results])
        response = llm.generate_content(system_prompt, content)
        
        return {
            "analysis": response,
            "source_count": len(results),
            "coverage_score": self._calculate_coverage(results)
        }
    
    def _calculate_coverage(self, results: List[Dict[str, Any]]) -> float:
        # Implementation for calculating research coverage
        pass