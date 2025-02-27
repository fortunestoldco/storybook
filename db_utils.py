import os
from typing import Dict, List, Optional, Union

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from pymongo import MongoClient

from models import Novel


class MongoDBManager:
    def __init__(self):
        # Load environment variables
        self.uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("MONGODB_DATABASE_NAME")
        self.doc_collection = os.getenv("DOCUMENT_COLLECTION_NAME")
        self.vector_collection = os.getenv("VECTOR_COLLECTION_NAME")
        
        # Initialize MongoDB client
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.manuscripts = self.db[self.doc_collection]
        self.vectors = self.db[self.vector_collection]
        
        # Initialize vector store
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.vectors,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding"
        )
    
    def store_novel(self, novel: Novel) -> str:
        """Store a novel in the database and return its ID"""
        novel_dict = novel.model_dump()
        result = self.manuscripts.insert_one(novel_dict)
        return str(result.inserted_id)
    
    def update_novel(self, novel_id: str, novel: Novel) -> bool:
        """Update a novel in the database"""
        novel_dict = novel.model_dump()
        result = self.manuscripts.update_one(
            {"_id": novel_id},
            {"$set": novel_dict}
        )
        return result.modified_count > 0
    
    def get_novel(self, novel_id: str) -> Optional[Novel]:
        """Retrieve a novel from the database"""
        novel_dict = self.manuscripts.find_one({"_id": novel_id})
        if novel_dict:
            return Novel(**novel_dict)
        return None
    
    def store_document_vectors(self, documents: List[Document], metadata: Dict) -> List[str]:
        """Store documents with their vector embeddings"""
        ids = []
        for doc in documents:
            embedding = self.embeddings.embed_query(doc.page_content)
            result = self.vectors.insert_one({
                "text": doc.page_content,
                "embedding": embedding,
                "metadata": {**doc.metadata, **metadata}
            })
            ids.append(str(result.inserted_id))
        return ids
    
    def search_similar_content(self, query: str, metadata_filter: Dict = None, limit: int = 5) -> List[Document]:
        """Search for similar content in the vector store"""
        filter_dict = metadata_filter or {}
        results = self.vector_store.similarity_search(
            query=query,
            k=limit,
            filter=filter_dict
        )
        return results
