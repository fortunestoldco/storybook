from typing import Dict, Any, List, Optional
from datetime import datetime
from ..db_config import get_collection, COLLECTIONS

class ResearchStorage:
    """Handles storage and retrieval of research data."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.collection = get_collection(COLLECTIONS["research"])
    
    async def store_research(self, data: Dict[str, Any]) -> str:
        """Store research data."""
        document = {
            "project_id": self.project_id,
            "data": data,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)
    
    async def get_research(self, research_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve research data by ID."""
        result = self.collection.find_one({"_id": research_id})
        return result if result else None
    
    async def list_research(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List research documents matching query."""
        base_query = {"project_id": self.project_id}
        if query:
            base_query.update(query)
        return list(self.collection.find(base_query))
    
    async def update_research(self, research_id: str, data: Dict[str, Any]) -> bool:
        """Update existing research document."""
        update_data = {
            "data": data,
            "updated_at": datetime.utcnow()
        }
        result = self.collection.update_one(
            {"_id": research_id},
            {"$set": update_data}
        )
        return result.modified_count > 0