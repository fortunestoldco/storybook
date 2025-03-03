import unittest
from unittest.mock import patch, MagicMock
from mongodb import MongoDBManager

class TestMongoDBManager(unittest.TestCase):

    @patch('mongodb.MongoClient')
    def setUp(self, MockMongoClient):
        self.mock_client = MockMongoClient.return_value
        self.mock_db = self.mock_client[MONGODB_CONFIG["database_name"]]
        self.manager = MongoDBManager()

    def test_save_state(self):
        project_id = "test_project"
        state = {"key": "value"}
        self.manager.save_state(project_id, state)
        collection = self.mock_db[MONGODB_CONFIG["collections"]["project_state"]]
        collection.update_one.assert_called_once_with(
            {"project_id": project_id},
            {"$set": state},
            upsert=True
        )

    def test_load_state(self):
        project_id = "test_project"
        expected_state = {"project_id": project_id, "key": "value"}
        collection = self.mock_db[MONGODB_CONFIG["collections"]["project_state"]]
        collection.find_one.return_value = expected_state
        state = self.manager.load_state(project_id)
        self.assertEqual(state, expected_state)

    def test_save_document(self):
        document = {"_id": "test_id", "key": "value"}
        self.manager.save_document(document)
        collection = self.mock_db[MONGODB_CONFIG["collections"]["documents"]]
        collection.replace_one.assert_called_once_with(
            {"_id": document["_id"]}, document, upsert=True
        )

    def test_load_document(self):
        document_id = "test_id"
        expected_document = {"_id": document_id, "key": "value"}
        collection = self.mock_db[MONGODB_CONFIG["collections"]["documents"]]
        collection.find_one.return_value = expected_document
        document = self.manager.load_document(document_id)
        self.assertEqual(document, expected_document)

    def test_save_research(self):
        research = {"_id": "test_id", "key": "value"}
        self.manager.save_research(research)
        collection = self.mock_db[MONGODB_CONFIG["collections"]["research"]]
        collection.replace_one.assert_called_once_with(
            {"_id": research["_id"]}, research, upsert=True
        )

if __name__ >= '__main__':
    unittest.main()
