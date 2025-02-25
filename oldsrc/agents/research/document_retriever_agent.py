from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
from services.tools_service import ToolsService
from services.mongodb_service import MongoDBService
import json
from config import MONGODB_CONFIG
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np

class DocumentRetrieverAgent:
    def __init__(self, tools_service: ToolsService):
        self.tools_service = tools_service
        self.logger = logging.getLogger(self.__class__.__name__)
        self.system_prompts = self._load_system_prompts()
        self.mongo_client = AsyncIOMotorClient(MONGODB_CONFIG.connection_string)
        self.db = self.mongo_client[MONGODB_CONFIG.database_name]
        self.vector_collection = self.db[MONGODB_CONFIG.collections["vectors"]]
        
        # Ensure vector search index exists
        self._ensure_vector_search_index()

    def _load_system_prompts(self) -> Dict[str, Any]:
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    async def _ensure_vector_search_index(self):
        """Ensure the vector search index exists in MongoDB Atlas."""
        try:
            # Check if index exists
            indexes = await self.vector_collection.list_indexes()
            index_exists = any(idx.get("name") == MONGODB_CONFIG.vector_search["index"] 
                             for idx in await indexes.to_list(length=None))
            
            if not index_exists:
                # Create vector search index
                await self.db.command({
                    "createIndexes": MONGODB_CONFIG.collections["vectors"],
                    "indexes": [{
                        "name": MONGODB_CONFIG.vector_search["index"],
                        "key": {
                            "vector": "vectorSearch"
                        },
                        "vectorSearchOptions": {
                            "dimension": MONGODB_CONFIG.vector_search["embedding_dimension"],
                            "similarity": MONGODB_CONFIG.vector_search["similarity_metric"]
                        }
                    }]
                })
                self.logger.info("Vector search index created successfully")
        except Exception as e:
            self.logger.error(f"Error ensuring vector search index: {str(e)}")

    async def retrieve_documents(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve documents using vector similarity search."""
        try:
            self.logger.info("Retrieving documents")
            
            # Get query embedding from OpenAI
            embeddings = self.tools_service.embeddings
            query_embedding = await embeddings.embed_query(query)
            
            # Prepare aggregation pipeline for vector search
            pipeline = [
                {
                    "$search": {
                        "index": MONGODB_CONFIG.vector_search["index"],
                        "vectorSearch": {
                            "queryVector": query_embedding,
                            "path": "vector",
                            "numCandidates": MONGODB_CONFIG.vector_search["num_candidates"],
                            "limit": 10
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "score": {
                            "$meta": "vectorSearchScore"
                        }
                    }
                }
            ]
            
            # Add filters if provided
            if filters:
                pipeline.insert(1, {"$match": filters})
            
            # Execute search
            cursor = self.vector_collection.aggregate(pipeline)
            documents = await cursor.to_list(length=10)
            
            # Process results
            results = []
            for doc in documents:
                result = {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc["score"],
                    "timestamp": datetime.utcnow().isoformat()
                }
                results.append(result)
            
            self.logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {str(e)}")
            return []

    async def store_document(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Store a document with its vector embedding."""
        try:
            # Get document embedding
            embeddings = self.tools_service.embeddings
            vector = await embeddings.embed_documents([content])
            
            # Prepare document
            document = {
                "content": content,
                "metadata": metadata,
                "vector": vector[0],
                "timestamp": datetime.utcnow()
            }
            
            # Store in MongoDB
            result = await self.vector_collection.insert_one(document)
            return bool(result.inserted_id)
            
        except Exception as e:
            self.logger.error(f"Error storing document: {str(e)}")
            return False

    async def reflect(self, result: Dict[str, Any]) -> bool:
        """Reflect on the quality and completeness of the retrieved documents."""
        required_sections = ["content", "metadata"]

        # Check completeness
        if not all(section in result for section in required_sections):
            return False

        # Validate content quality
        if len(result["content"]) < 100:
            return False

        return True

    async def cleanup(self) -> None:
        """Cleanup after document retrieval."""
        try:
            # Close MongoDB connection
            if self.mongo_client:
                self.mongo_client.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")