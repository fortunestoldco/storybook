"""
Vector store service module.
"""

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from storybook.config import MONGODB_CONNECTION_STRING, DB_NAME


class VectorStoreService:
    def __init__(self):
        self.vector_store = MongoDBAtlasVectorSearch(
            database=DB_NAME, uri=MONGODB_CONNECTION_STRING
        )

    def add_vector(self, vector: dict):
        return self.vector_store.add_vector(vector)

    def search_vectors(self, query: str):
        return self.vector_store.search(query)
