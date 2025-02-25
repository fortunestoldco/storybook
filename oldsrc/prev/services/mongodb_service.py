from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from config import MONGODB_CONFIG
from langchain_mongodb import MongoDBChatMessageHistory

class MongoDBService:
    def __init__(self):
        self.client = MongoClient(MONGODB_CONFIG.connection_string)
        self.db = self.client[MONGODB_CONFIG.database_name]

    async def create_project(self, project_data: Dict[str, Any]) -> str:
        """Create a new novel writing project."""
        result = await self.db[MONGODB_CONFIG.projects_collection].insert_one(project_data)
        return str(result.inserted_id)

    async def store_story_bible(self, novel_id: str, bible_data: Dict[str, Any]) -> bool:
        """Store the story bible for a project."""
        result = await self.db[MONGODB_CONFIG.story_bible_collection].insert_one({
            "novel_id": novel_id,
            **bible_data
        })
        return bool(result.inserted_id)

    async def store_research_report(
        self,
        novel_id: str,
        report_type: str,
        report_data: Dict[str, Any]
    ) -> bool:
        """Store a research report."""
        result = await self.db[MONGODB_CONFIG.research_collection].insert_one({
            "novel_id": novel_id,
            "report_type": report_type,
            **report_data
        })
        return bool(result.inserted_id)

    async def store_draft(
        self,
        novel_id: str,
        draft_number: int,
        chapters: List[Dict[str, Any]]
    ) -> bool:
        """Store a manuscript draft."""
        result = await self.db[MONGODB_CONFIG.drafts_collection].insert_one({
            "novel_id": novel_id,
            "draft_number": draft_number,
            "chapters": chapters,
            "status": "in_progress"
        })
        return bool(result.inserted_id)

    async def update_draft_status(
        self,
        novel_id: str,
        draft_number: int,
        status: str
    ) -> bool:
        """Update the status of a draft."""
        result = await self.db[MONGODB_CONFIG.drafts_collection].update_one(
            {"novel_id": novel_id, "draft_number": draft_number},
            {"$set": {"status": status}}
        )
        return result.modified_count > 0

    async def store_feedback(
        self,
        novel_id: str,
        feedback_data: Dict[str, Any]
    ) -> bool:
        """Store user feedback."""
        result = await self.db[MONGODB_CONFIG.feedback_collection].insert_one({
            "novel_id": novel_id,
            **feedback_data
        })
        return bool(result.inserted_id)

    async def get_project(self, novel_id: str) -> Optional[Dict[str, Any]]:
        """Get project details."""
        return await self.db[MONGODB_CONFIG.projects_collection].find_one({"_id": novel_id})

    async def get_story_bible(self, novel_id: str) -> Optional[Dict[str, Any]]:
        """Get the story bible for a project."""
        return await self.db[MONGODB_CONFIG.story_bible_collection].find_one({"novel_id": novel_id})

    async def store_character_profiles(
        self,
        novel_id: str,
        character_profiles: Dict[str, Any]
    ) -> bool:
        """Store character profiles."""
        result = await self.db[MONGODB_CONFIG.character_collection].insert_one({
            "novel_id": novel_id,
            **character_profiles
        })
        return bool(result.inserted_id)

    async def store_world_spec(
        self,
        novel_id: str,
        world_spec: Dict[str, Any]
    ) -> bool:
        """Store world specifications."""
        result = await self.db[MONGODB_CONFIG.world_building_collection].insert_one({
            "novel_id": novel_id,
            **world_spec
        })
        return bool(result.inserted_id)

    async def update_chapter(
        self,
        novel_id: str,
        chapter_number: int,
        chapter_data: Dict[str, Any]
    ) -> bool:
        """Update a chapter with new data."""
        result = await self.db[MONGODB_CONFIG.drafts_collection].update_one(
            {"novel_id": novel_id, "chapters.chapter_number": chapter_number},
            {"$set": {"chapters.$": chapter_data}}
        )
        return result.modified_count > 0

    async def store_final_manuscript(
        self,
        novel_id: str,
        final_manuscript: Dict[str, Any]
    ) -> bool:
        """Store the final manuscript."""
        result = await self.db[MONGODB_CONFIG.drafts_collection].insert_one({
            "novel_id": novel_id,
            **final_manuscript
        })
        return bool(result.inserted_id)

    async def store_review_report(
        self,
        novel_id: str,
        review_report: Dict[str, Any]
    ) -> bool:
        """Store the review report for multi-writer review."""
        result = await self.db[MONGODB_CONFIG.feedback_collection].insert_one({
            "novel_id": novel_id,
            **review_report
        })
        return bool(result.inserted_id)

    async def get_previous_chapters(
        self,
        novel_id: str,
        chapter_number: int
    ) -> List[Dict[str, Any]]:
        """Get previous chapters for context."""
        result = await self.db[MONGODB_CONFIG.drafts_collection].find_one(
            {"novel_id": novel_id},
            {"chapters": {"$slice": chapter_number - 1}}
        )
        return result.get("chapters", []) if result else []

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
