"""
Ollama service module.
"""

from langchain_community.chat_models import ChatOllama
from storybook.config import OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL

class OllamaService:
    def __init__(self):
        self.client = ChatOllama(
            model=OLLAMA_DEFAULT_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def generate_response(self, prompt: str) -> str:
        return self.client.generate(prompt)
