from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message passed between agents."""

    sender: str
    recipient: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentCommunication:
    """Utilities for agent communication."""

    @staticmethod
    def create_message(
        sender: str,
        recipient: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Create a new message."""
        if metadata is None:
            metadata = {}

        return Message(
            sender=sender, recipient=recipient, content=content, metadata=metadata
        )

    @staticmethod
    def format_for_llm(messages: List[Message]) -> str:
        """Format a list of messages for LLM consumption."""
        formatted = ""
        for msg in messages:
            formatted += f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {msg.sender} to {msg.recipient}:\n{msg.content}\n\n"
        return formatted

    @staticmethod
    def extract_task_assignments(message: Message) -> Dict[str, Any]:
        """Extract task assignments from a message."""
        # This would be implemented with actual NLP or pattern matching
        # Simplified version for demonstration
        assignments = {}
        if "assignments" in message.metadata:
            return message.metadata["assignments"]

        # Simple parsing fallback
        if "TASK:" in message.content:
            parts = message.content.split("TASK:")
            for part in parts[1:]:
                task_content = part.strip().split("\n")[0]
                if ":" in task_content:
                    agent, task = task_content.split(":", 1)
                    assignments[agent.strip()] = task.strip()

        return assignments
