"""
Database service module.
"""

from langchain_mongodb import MongoDBAtlasVectorSearch
from storybook.config import MONGODB_CONNECTION_STRING, DB_NAME


class DatabaseService:
    def __init__(self):
        self.vector_store = MongoDBAtlasVectorSearch(
            database=DB_NAME, uri=MONGODB_CONNECTION_STRING
        )

    def get_collection(self, collection_name: str):
        return self.vector_store.get_collection(collection_name)

    def insert_one(self, collection_name: str, document: dict):
        collection = self.get_collection(collection_name)
        return collection.insert_one(document)

    def find_one(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.find_one(query)

    def update_one(self, collection_name: str, query: dict, update: dict):
        collection = self.get_collection(collection_name)
        return collection.update_one(query, update)

    def delete_one(self, collection_name: str, query: dict):
        collection = self.get_collection(collection_name)
        return collection.delete_one(query)
