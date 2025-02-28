from __future__ import annotations

# Standard library imports
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# Third-party imports
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MongoDBStore:
    """MongoDB storage for manuscripts and related data."""

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "storybook"):
        """Initialize MongoDB connection."""
        try:
            self.client = MongoClient(uri)
            self.db: Database = self.client[db_name]
            self.manuscripts: Collection = self.db.manuscripts
            logger.info(f"Connected to MongoDB at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def store_manuscript(self, manuscript_id: str, content: Dict[str, Any]) -> bool:
        """Store or update a manuscript."""
        try:
            content["updated_at"] = datetime.utcnow()
            result = self.manuscripts.update_one(
                {"_id": manuscript_id},
                {"$set": content},
                upsert=True
            )
            return result.acknowledged
        except Exception as e:
            logger.error(f"Failed to store manuscript {manuscript_id}: {e}")
            return False

    def get_manuscript(self, manuscript_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a manuscript by ID."""
        try:
            return self.manuscripts.find_one({"_id": manuscript_id})
        except Exception as e:
            logger.error(f"Failed to retrieve manuscript {manuscript_id}: {e}")
            return None

    def list_manuscripts(self) -> List[Dict[str, Any]]:
        """List all manuscripts."""
        try:
            return list(self.manuscripts.find())
        except Exception as e:
            logger.error(f"Failed to list manuscripts: {e}")
            return []

    def delete_manuscript(self, manuscript_id: str) -> bool:
        """Delete a manuscript by ID."""
        try:
            result = self.manuscripts.delete_one({"_id": manuscript_id})
            return result.acknowledged
        except Exception as e:
            logger.error(f"Failed to delete manuscript {manuscript_id}: {e}")
            return False
