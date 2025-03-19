from typing import Dict, Any, List, Optional
from datetime import datetime

from pymongo import MongoClient
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tools import ResearchTools
from utils import check_quality_gate, split_manuscript, extract_chunk_references
from graph import create_phase_graph
from config import MONGODB_URI, QUALITY_GATES

class Storybook:
    def __init__(self, title: str, author: str, synopsis: str, manuscript: str, model_config: Optional[Dict[str, Any]] = None):
        self.title = title
        self.author = author
        self.synopsis = synopsis
        self.manuscript = manuscript
        self.model_config = model_config or {}
        self.mongo_client = None
        self.text_splitter = None
        self.agent_factory = None
        self.storybook_graph = None

        # Initialize MongoDB connection if URI is available
        if MONGODB_URI:
            try:
                self.mongo_client = MongoClient(MONGODB_URI)
                # Test connection
                self.mongo_client.admin.command('ping')
            except Exception as e:
                print(f"MongoDB connection failed: {str(e)}")

    def run_storybook_phase(self, state: Dict[str, Any], phase: str, progress_callback=None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a specific phase of the storybook workflow."""
        try:
            # Validate state
            if not state or not isinstance(state, dict):
                raise ValueError("Invalid state provided")

            # Create a copy of state to avoid modifying the original
            current_state = state.copy()

            # Set the phase in the state
            current_state["phase"] = phase
            if "current_input" not in current_state:
                current_state["current_input"] = {}
            current_state["current_input"]["phase"] = phase

            # Create phase graph
            phase_graph = create_phase_graph(config or {
                "configurable": {
                    "phase": phase,
                    "model_config": self.model_config,
                    "agent_factory": self.agent_factory
                }
            })
            
            if not phase_graph:
                raise ValueError(f"Could not create graph for phase: {phase}")

            # Execute the phase graph
            if progress_callback:
                progress_callback(f"Executing {phase} phase graph...")

            try:
                result_state = phase_graph.invoke(current_state)
                if progress_callback:
                    progress_callback(f"Completed {phase} phase execution")
                return result_state
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error executing {phase} phase: {str(e)}")
                raise

        except Exception as e:
            # Create error state
            error_state = state.copy()
            error_state["error"] = f"Error in {phase} phase: {str(e)}"
            error_state["lnode"] = "error"
            return error_state

    def initialize_storybook_project(self, title: str, synopsis: str, manuscript: str, notes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize a new storybook project with the given inputs."""
        project_id = f"proj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Split manuscript into chunks
        manuscript_chunks = split_manuscript(manuscript, self.text_splitter)
        
        # Create initial project data
        project_data = {
            "id": project_id,
            "title": title,
            "synopsis": synopsis,
            "manuscript": manuscript,
            "manuscript_chunks": manuscript_chunks,
            "notes": notes or {},
            "type": "new",
            "quality_assessment": {
                "clarity": 0.0,
                "coherence": 0.0,
                "engagement": 0.0,
                "style": 0.0,
                "overall": 0.0
            },
            "created_at": datetime.now().isoformat()
        }

        # Create initial state
        initial_state = {
            "project": project_data,
            "phase": "initialization",
            "phase_history": {},
            "current_input": {
                "project_type": "new",
                "project_data": project_data,
                "task": "Initialize and analyze project"
            },
            "messages": [],
            "count": 0,
            "lnode": "executive_director"
        }

        return initial_state

    def update_model_config(self, model_config: Dict[str, Any]):
        """Update the model configuration."""
        self.model_config = model_config
        # Update agent factory if it exists
        if self.agent_factory:
            self.agent_factory.model_config = model_config

    def set_text_splitter(self, splitter):
        """Set the text splitter for manuscript processing."""
        self.text_splitter = splitter

    def get_manuscript_statistics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for the current manuscript."""
        try:
            manuscript = state.get("project", {}).get("manuscript", "")
            if not manuscript:
                return {"error": "No manuscript found in state"}

            # Calculate basic statistics
            words = manuscript.split()
            sentences = manuscript.split('.')
            paragraphs = manuscript.split('\n\n')
            
            stats = {
                "word_count": len(words),
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "average_words_per_sentence": len(words) / max(len(sentences), 1),
                "average_words_per_paragraph": len(words) / max(len(paragraphs), 1),
            }

            return stats
        except Exception as e:
            return {"error": f"Error calculating statistics: {str(e)}"}

    def save_checkpoint(self, state: Dict[str, Any], checkpoint_id: str) -> bool:
        """Save current state as a checkpoint."""
        if not self.mongo_client:
            return False

        try:
            db = self.mongo_client["storybook"]
            checkpoints = db["checkpoints"]
            
            checkpoint = {
                "checkpoint_id": checkpoint_id,
                "project_id": state.get("project", {}).get("id", "unknown"),
                "phase": state.get("phase", "unknown"),
                "state": state,
                "last_modified": datetime.now()
            }
            
            checkpoints.update_one(
                {"checkpoint_id": checkpoint_id},
                {"$set": checkpoint},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            return False

    def load_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load a state from a checkpoint."""
        if not self.mongo_client:
            return {}

        try:
            db = self.mongo_client["storybook"]
            checkpoints = db["checkpoints"]
            
            checkpoint = checkpoints.find_one({"checkpoint_id": checkpoint_id})
            if not checkpoint:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
                
            return checkpoint.get("state", {})
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return {}

    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available checkpoints."""
        if not self.mongo_client:
            return []

        try:
            db = self.mongo_client["storybook"]
            checkpoints = db["checkpoints"]
            
            cursor = checkpoints.find(
                {},
                {
                    "checkpoint_id": 1,
                    "project_id": 1,
                    "phase": 1,
                    "last_modified": 1
                }
            ).sort("last_modified", -1)
            
            return list(cursor)
        except Exception as e:
            print(f"Error getting checkpoints: {str(e)}")
            return []

    def export_manuscript(self, state: Dict[str, Any], format: str = "text") -> str:
        """Export manuscript in various formats."""
        try:
            manuscript = state.get("project", {}).get("manuscript", "")
            if not manuscript:
                return "No manuscript found in state"

            if format == "text":
                return manuscript
            elif format == "markdown":
                # Convert to markdown format
                title = state.get("project", {}).get("title", "Untitled")
                return f"# {title}\n\n{manuscript}"
            elif format == "html":
                # Convert to HTML format
                title = state.get("project", {}).get("title", "Untitled")
                return f"<h1>{title}</h1>\n<div class='manuscript'>\n{manuscript}\n</div>"
            else:
                return manuscript
        except Exception as e:
            return f"Error exporting manuscript: {str(e)}"
