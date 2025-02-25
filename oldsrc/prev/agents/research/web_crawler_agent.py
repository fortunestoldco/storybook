import asyncio
import logging
from services.tools_service import ToolsService
import json

class WebCrawlerAgent:
    def __init__(self, tools_service: ToolsService):
        self.tools_service = tools_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_prompts = self._load_system_prompts()

    def _load_system_prompts(self) -> Dict[str, Any]:
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)
    
    async def crawl_web(self, query: str):
        self.logger.info("Starting web crawling.")
        web_crawler = self.tools_service.get_web_crawler()
        results = await web_crawler._arun(query)
        self.logger.info("Web crawling completed.")
        return results
    
    async def fetch_and_parse(self, url: str) -> Dict[str, Any]:
        self.logger.debug(f"Crawling URL: {url}")
        web_crawler = self.tools_service.get_web_crawler()
        result = await web_crawler._arun(url)
        if "error" not in result:
            self.logger.debug(f"Data from {url} stored.")
        else:
            self.logger.warning(f"Failed to retrieve {url}: {result['error']}")
        return result

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the quality and completeness of the web crawling result."""
        required_sections = ["text", "metadata"]

        # Check completeness
        if not all(section in result for section in required_sections):
            return False

        # Validate content quality
        if len(result["text"]) < 100:
            return False

        return True

    async def cleanup(self) -> None:
        """Cleanup after web crawling."""
        self.state.memory = None
