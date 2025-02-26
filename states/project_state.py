# storybook/states/project_state.py

"""
Project state management for the Storybook system.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import json
import uuid
import datetime

class ProjectStatus(str, Enum):
    """Enum for project status."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    RESEARCHING = "researching"
    DRAFTING = "drafting"
    REVISING = "revising"
    EDITING = "editing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"

class ProjectState:
    """State management for novel generation projects."""
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        Initialize project state.
        
        Args:
            initial_data: Optional initial state data
        """
        self.state = initial_data or self._create_default_state()
        
    def _create_default_state(self) -> Dict[str, Any]:
        """
        Create default project state.
        
        Returns:
            Default state dictionary
        """
        return {
            "project_id": str(uuid.uuid4()),
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "status": ProjectStatus.INITIALIZED,
            "title": "Untitled Novel",
            "genre": "",
            "target_audience": "",
            "word_count_target": 80000,
            "description": "",
            "metadata": {},
            "market_research": {},
            "outline": [],
            "characters": {},
            "settings": {},
            "themes": [],
            "chapters": {},
            "current_draft_number": 0,
            "revision_history": [],
            "feedback": []
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.
        
        Returns:
            Current project state
        """
        return self.state
    
    def update_state(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the project state.
        
        Args:
            updates: Dictionary of state updates
            
        Returns:
            Updated state
        """
        # Update the state dictionary
        for key, value in updates.items():
            if key in self.state:
                if isinstance(self.state[key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    self.state[key].update(value)
                else:
                    # Replace value
                    self.state[key] = value
            else:
                # Add new key
                self.state[key] = value
                
        # Update the updated_at timestamp
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return self.state
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get project metadata.
        
        Returns:
            Project metadata
        """
        return {
            "project_id": self.state["project_id"],
            "title": self.state["title"],
            "genre": self.state["genre"],
            "status": self.state["status"],
            "created_at": self.state["created_at"],
            "updated_at": self.state["updated_at"],
            "word_count_target": self.state["word_count_target"],
            "current_draft_number": self.state["current_draft_number"]
        }
    
    def update_status(self, status: ProjectStatus) -> Dict[str, Any]:
        """
        Update project status.
        
        Args:
            status: New project status
            
        Returns:
            Updated state
        """
        self.state["status"] = status
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        # Add status change to revision history
        self.state["revision_history"].append({
            "timestamp": self.state["updated_at"],
            "type": "status_change",
            "old_status": self.state.get("status"),
            "new_status": status
        })
        
        return self.state
    
    def add_chapter(self, chapter_data: Dict[str, Any]) -> str:
        """
        Add a new chapter to the project.
        
        Args:
            chapter_data: Chapter data
            
        Returns:
            Chapter ID
        """
        chapter_id = chapter_data.get("id", str(uuid.uuid4()))
        
        # Ensure the chapter has all required fields
        if "title" not in chapter_data:
            chapter_data["title"] = f"Chapter {len(self.state['chapters']) + 1}"
        if "content" not in chapter_data:
            chapter_data["content"] = ""
        if "sequence_number" not in chapter_data:
            chapter_data["sequence_number"] = len(self.state["chapters"]) + 1
        if "status" not in chapter_data:
            chapter_data["status"] = "draft"
        if "created_at" not in chapter_data:
            chapter_data["created_at"] = datetime.datetime.now().isoformat()
        if "updated_at" not in chapter_data:
            chapter_data["updated_at"] = datetime.datetime.now().isoformat()
            
        # Add chapter to state
        self.state["chapters"][chapter_id] = chapter_data
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return chapter_id
    
    def update_chapter(self, chapter_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a chapter.
        
        Args:
            chapter_id: ID of the chapter to update
            updates: Updates to apply
            
        Returns:
            Updated chapter data
        """
        if chapter_id not in self.state["chapters"]:
            raise ValueError(f"Chapter {chapter_id} not found")
            
        # Update chapter
        for key, value in updates.items():
            self.state["chapters"][chapter_id][key] = value
            
        # Update timestamps
        self.state["chapters"][chapter_id]["updated_at"] = datetime.datetime.now().isoformat()
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return self.state["chapters"][chapter_id]
    
    def get_chapter(self, chapter_id: str) -> Dict[str, Any]:
        """
        Get a chapter.
        
        Args:
            chapter_id: ID of the chapter to retrieve
            
        Returns:
            Chapter data
        """
        if chapter_id not in self.state["chapters"]:
            raise ValueError(f"Chapter {chapter_id} not found")
            
        return self.state["chapters"][chapter_id]
    
    def get_chapters(self, ordered: bool = True) -> List[Dict[str, Any]]:
        """
        Get all chapters.
        
        Args:
            ordered: Whether to order chapters by sequence number
            
        Returns:
            List of chapter data
        """
        chapters = list(self.state["chapters"].values())
        
        if ordered:
            chapters.sort(key=lambda x: x["sequence_number"])
            
        return chapters
    
    def add_character(self, character_data: Dict[str, Any]) -> str:
        """
        Add a character to the project.
        
        Args:
            character_data: Character data
            
        Returns:
            Character ID
        """
        character_id = character_data.get("id", str(uuid.uuid4()))
        
        # Ensure character has required fields
        if "name" not in character_data:
            character_data["name"] = f"Character {len(self.state['characters']) + 1}"
        if "created_at" not in character_data:
            character_data["created_at"] = datetime.datetime.now().isoformat()
        if "updated_at" not in character_data:
            character_data["updated_at"] = datetime.datetime.now().isoformat()
            
        # Add character to state
        self.state["characters"][character_id] = character_data
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return character_id
    
    def update_character(self, character_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a character.
        
        Args:
            character_id: ID of the character to update
            updates: Updates to apply
            
        Returns:
            Updated character data
        """
        if character_id not in self.state["characters"]:
            raise ValueError(f"Character {character_id} not found")
            
        # Update character
        for key, value in updates.items():
            self.state["characters"][character_id][key] = value
            
        # Update timestamps
        self.state["characters"][character_id]["updated_at"] = datetime.datetime.now().isoformat()
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return self.state["characters"][character_id]
    
    def get_character(self, character_id: str) -> Dict[str, Any]:
        """
        Get a character.
        
        Args:
            character_id: ID of the character to retrieve
            
        Returns:
            Character data
        """
        if character_id not in self.state["characters"]:
            raise ValueError(f"Character {character_id} not found")
            
        return self.state["characters"][character_id]
    
    def get_characters(self) -> List[Dict[str, Any]]:
        """
        Get all characters.
        
        Returns:
            List of character data
        """
        return list(self.state["characters"].values())
    
    def add_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Add feedback to the project.
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Feedback ID
        """
        feedback_id = feedback_data.get("id", str(uuid.uuid4()))
        
        # Ensure feedback has required fields
        if "content" not in feedback_data:
            raise ValueError("Feedback must include content")
        if "source" not in feedback_data:
            feedback_data["source"] = "unknown"
        if "timestamp" not in feedback_data:
            feedback_data["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add feedback ID
        feedback_data["id"] = feedback_id
            
        # Add feedback to state
        self.state["feedback"].append(feedback_data)
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        return feedback_id
    
    def get_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all feedback.
        
        Returns:
            List of feedback data
        """
        return self.state["feedback"]
    
    def increment_draft_number(self) -> int:
        """
        Increment the current draft number.
        
        Returns:
            New draft number
        """
        self.state["current_draft_number"] += 1
        self.state["updated_at"] = datetime.datetime.now().isoformat()
        
        # Add draft increment to revision history
        self.state["revision_history"].append({
            "timestamp": self.state["updated_at"],
            "type": "draft_increment",
            "old_draft": self.state["current_draft_number"] - 1,
            "new_draft": self.state["current_draft_number"]
        })
        
        return self.state["current_draft_number"]
    
    def get_full_manuscript(self) -> str:
        """
        Get the full manuscript content.
        
        Returns:
            Full manuscript text
        """
        chapters = self.get_chapters(ordered=True)
        return "\n\n".join(chapter["content"] for chapter in chapters if "content" in chapter)
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save the project state to a file.
        
        Args:
            filepath: Path to save the file
        """
        with open(filepath, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ProjectState':
        """
        Load project state from a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            ProjectState instance
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        return cls(state)
