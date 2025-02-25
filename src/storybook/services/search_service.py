"""Search services for the Storybook application."""

import os
from typing import Dict, List, Any, Optional
import json
import requests
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper, GoogleSerperAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

from config import (
    GOOGLE_API_KEY, GOOGLE_CSE_ID, 
    SERPER_API_KEY, TAVILY_API_KEY,
    GOOGLE_BOOKS_API_KEY
)

class GoogleSearchService:
    """Service for Google Search operations."""
    
    def __init__(self):
        """Initialize the Google Search service."""
        self.search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        )
    
    def run_search(self, query: str, num_results: int = 8) -> List[Dict[str, Any]]:
        """Run a Google search."""
        results = self.search.results(query, num_results)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": "google_search"
            })
        
        return formatted_results
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for Google Search."""
        return Tool(
            name="google_search",
            description="Search Google for information. Useful for finding general information, news, and facts.",
            func=self.run_search,
            coroutine=None
        )

class SerperSearchService:
    """Service for Google Search via Serper API."""
    
    def __init__(self):
        """Initialize the Serper Search service."""
        self.search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)
    
    def run_search(self, query: str, num_results: int = 8) -> List[Dict[str, Any]]:
        """Run a Google search via Serper."""
        results = self.search.results(query, num_results)
        
        formatted_results = []
        if "organic" in results:
            for result in results["organic"][:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": "serper_search"
                })
        
        return formatted_results
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for Serper Search."""
        return Tool(
            name="serper_search",
            description="Search the web for information using Serper API. Useful for finding current information.",
            func=self.run_search,
            coroutine=None
        )

class TavilySearchService:
    """Service for Tavily Search API."""
    
    def __init__(self):
        """Initialize the Tavily Search service."""
        self.search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
    
    def run_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Run a search using Tavily API."""
        search_results = self.search.results(query, max_results=max_results)
        
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "source": "tavily_search"
            })
        
        return formatted_results
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for Tavily Search."""
        return Tool(
            name="tavily_search",
            description="Search for information using Tavily Search API. Provides high-quality, recent information with content extraction.",
            func=self.run_search,
            coroutine=None
        )

class GoogleBooksService:
    """Service for Google Books API."""
    
    def __init__(self):
        """Initialize the Google Books service."""
        self.api_key = GOOGLE_BOOKS_API_KEY
        self.base_url = "https://www.googleapis.com/books/v1/volumes"
    
    def search_books(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for books using Google Books API."""
        params = {
            "q": query,
            "maxResults": max_results,
            "key": self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        results = []
        if "items" in data:
            for item in data["items"]:
                volume_info = item.get("volumeInfo", {})
                
                # Extract authors
                authors = volume_info.get("authors", [])
                authors_str = ", ".join(authors) if authors else "Unknown"
                
                # Extract description
                description = volume_info.get("description", "No description available")
                
                results.append({
                    "title": volume_info.get("title", "Untitled"),
                    "authors": authors_str,
                    "publisher": volume_info.get("publisher", "Unknown"),
                    "published_date": volume_info.get("publishedDate", "Unknown"),
                    "description": description,
                    "page_count": volume_info.get("pageCount", 0),
                    "categories": volume_info.get("categories", []),
                    "language": volume_info.get("language", "Unknown"),
                    "preview_link": volume_info.get("previewLink", ""),
                    "info_link": volume_info.get("infoLink", ""),
                    "source": "google_books"
                })
        
        return results
    
    def create_tool(self) -> Tool:
        """Create a LangChain tool for Google Books."""
        return Tool(
            name="google_books_search",
            description="Search for books, publications, and literary references using Google Books API.",
            func=self.search_books,
            coroutine=None
        )
