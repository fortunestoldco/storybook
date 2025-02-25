from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field

@dataclass
class StoryCreationState:
    """
    Represents the state of the story creation workflow at any given time.
    """
    title: str  # Title of the story
    genre: str  # Genre of the story
    themes: List[str]  # List of themes to be included
    story_bible: str = ""  # Central knowledge repository, initially empty
    brainstorm_summary: Optional[str] = None # Summary of brainstorming session
    research_reports: List[str] = field(default_factory=list) # Accumulation of research reports.
    market_research_report: Optional[str] = None # Market research report
    contextual_research_report: Optional[str] = None # Contextual research report
    consumer_research_report: Optional[str] = None  # Consumer Research Report
    outline: Optional[str] = None # Story outline
    world_specification: Optional[str] = None  # The world specification document
    character_development: Optional[str] = None  # The character development document
    story_plan: Optional[str] = None # The overall story plan after world/character development
    chapter_drafts: Dict[int, str] = field(default_factory=dict) # Store chapter drafts (chapter number -> content)
    editorial_notes: Dict[int, List[str]] = field(default_factory=dict)  # Store editorial notes for each chapter
    final_manuscript: Optional[str] = None # The final, complete manuscript
    sections: Dict[str, str] = field(default_factory=dict)  # Story sections, each with a unique identifier.  Values are the content.
    edits: Dict[str, str] = field(default_factory=dict) # Edits to the story sections
    final_story: Optional[str] = None  # The complete, final story
    metadata: Dict[str, str] = field(default_factory=dict)  # Metadata for publishing
    human_validation: bool = False  # Flag to pause for human validation
    current_step: str = "Project Initiation" # String representing the workflow stage.
    writer_assignments: Dict[str, List[str]] = field(default_factory=dict) # mapping of sections to assigned writers. List of writer names.
    editor_assignments: Dict[str, List[str]] = field(default_factory=dict) # mapping of sections to assigned editors. List of editor names.
    memory: List[BaseMessage] = field(default_factory=list) # Conversation history.
    input: str = ""  # The latest user input
    project_status: str = "Initial"  # Project status
    project_progress: int = 0 # Percentage of project complete
    project_goal: str = "Create a best-selling class novel that ticks all the boxes." # The overall goal of the project.
    overall_manuscript_draft_number: int = 0 # Track overall manuscript draft number (1, 2, 3)
    chapters_written: int = 0  # Number of chapters written
    all_chapters_finished: bool = False # Set to true when done all chapters

    def to_dict(self):
        return self.__dict__
