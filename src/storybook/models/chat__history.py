from typing import Dict, Any, List, Optional
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from storybook.db_config import MONGODB_URI, DB_NAME, COLLECTIONS

class ChatHistoryManager:
    """Manages chat message history using MongoDB."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize with MongoDB connection string."""
        self.connection_string = connection_string or MONGODB_URI
        self.database_name = DB_NAME
        self.collection_name = COLLECTIONS["chat_history"]
        
    def get_chat_history(self, session_id: str) -> MongoDBChatMessageHistory:
        """Get chat history for a specific session."""
        return MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=self.connection_string,
            database_name=self.database_name,
            collection_name=self.collection_name
        )
    
    def add_user_message(self, session_id: str, message: str) -> None:
        """Add a user message to the chat history."""
        chat_history = self.get_chat_history(session_id)
        chat_history.add_user_message(message)
    
    def add_ai_message(self, session_id: str, message: str) -> None:
        """Add an AI message to the chat history."""
        chat_history = self.get_chat_history(session_id)
        chat_history.add_ai_message(message)
    
    def add_system_message(self, session_id: str, message: str) -> None:
        """Add a system message to the chat history."""
        chat_history = self.get_chat_history(session_id)
        chat_history.add_message(SystemMessage(content=message))
    
    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Get all messages for a session."""
        chat_history = self.get_chat_history(session_id)
        return chat_history.messages
    
    def clear(self, session_id: str) -> None:
        """Clear chat history for a session."""
        chat_history = self.get_chat_history(session_id)
        chat_history.clear()