from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.mongodb_service import MongoDBService

class ImportationAgent(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manuscript parsing and storing chapters in MongoDB."""
        manuscript = task.get("manuscript", "")
        novel_id = task.get("novel_id", "")
        
        chapters = self._parse_manuscript(manuscript)
        await self._store_chapters(novel_id, chapters)
        await self._update_story_bible(novel_id, chapters)
        
        return {"status": "complete"}

    def _parse_manuscript(self, manuscript: str) -> Dict[int, str]:
        """Parse the manuscript text and split it into chapters."""
        chapters = {}
        current_chapter = 0
        chapter_lines = []

        for line in manuscript.split("\n"):
            if line.strip().startswith("CHAPTER"):
                if current_chapter > 0:
                    chapters[current_chapter] = "\n".join(chapter_lines)
                current_chapter += 1
                chapter_lines = []
            chapter_lines.append(line)

        if current_chapter > 0:
            chapters[current_chapter] = "\n".join(chapter_lines)

        return chapters

    async def _store_chapters(self, novel_id: str, chapters: Dict[int, str]) -> None:
        """Store the parsed chapters in MongoDB."""
        for chapter_number, chapter_text in chapters.items():
            await self.mongodb_service.update_chapter(
                novel_id,
                chapter_number,
                {"chapter_number": chapter_number, "text": chapter_text}
            )

    async def _update_story_bible(self, novel_id: str, chapters: Dict[int, str]) -> None:
        """Update the story bible with the imported chapters."""
        story_bible = await self.mongodb_service.get_story_bible(novel_id)
        story_bible["chapters"] = chapters
        await self.mongodb_service.store_story_bible(novel_id, story_bible)
