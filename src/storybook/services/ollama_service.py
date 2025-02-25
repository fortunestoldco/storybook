"""Ollama service for local LLM deployment with the Storybook application."""

import os
import json
import requests
from typing import Dict, List, Any, Optional, Union

from storybook.config import OLLAMA_BASE_URL

class OllamaService:
    """Service for interacting with Ollama local LLM instances."""
    
    def __init__(self, base_url: str = None):
        """Initialize the Ollama service."""
        self.base_url = base_url or OLLAMA_BASE_URL
        self.generate_endpoint = f"{self.base_url}/api/generate"
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.embeddings_endpoint = f"{self.base_url}/api/embeddings"
        self.models_endpoint = f"{self.base_url}/api/tags"
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in the Ollama instance."""
        try:
            response = requests.get(self.models_endpoint)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            print(f"Error listing Ollama models: {str(e)}")
            return []
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text from a prompt."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": max_tokens,
                }
            }
            
            if system:
                payload["system"] = system
                
            if stop:
                payload["options"]["stop"] = stop
            
            response = requests.post(self.generate_endpoint, json=payload)
            response.raise_for_status()
            
            return {
                "model": model,
                "generated_text": response.json().get("response", ""),
                "raw_response": response.json()
            }
        except Exception as e:
            print(f"Error generating text with Ollama: {str(e)}")
            return {"error": str(e)}
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Chat with the model using a message history."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": max_tokens,
                }
            }
            
            if stop:
                payload["options"]["stop"] = stop
            
            response = requests.post(self.chat_endpoint, json=payload)
            response.raise_for_status()
            
            return {
                "model": model,
                "message": response.json().get("message", {}),
                "raw_response": response.json()
            }
        except Exception as e:
            print(f"Error chatting with Ollama: {str(e)}")
            return {"error": str(e)}
    
    def get_embeddings(self, model: str, text: str) -> Dict[str, Any]:
        """Get embeddings for text using Ollama."""
        try:
            payload = {
                "model": model,
                "prompt": text
            }
            
            response = requests.post(self.embeddings_endpoint, json=payload)
            response.raise_for_status()
            
            return {
                "model": model,
                "embedding": response.json().get("embedding", []),
                "raw_response": response.json()
            }
        except Exception as e:
            print(f"Error getting embeddings from Ollama: {str(e)}")
            return {"error": str(e)}
