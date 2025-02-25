"""Configuration settings for the Storybook application."""

import os
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Model Configuration
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4-turbo")
RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4-turbo")
WRITING_MODEL = os.getenv("WRITING_MODEL", "gpt-4-turbo")
PUBLISHING_MODEL = os.getenv("PUBLISHING_MODEL", "gpt-4-turbo")
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "gpt-4-turbo")
AUTHOR_RELATIONS_MODEL = os.getenv("AUTHOR_RELATIONS_MODEL", "gpt-4-turbo")

# Ollama Models (when USE_OLLAMA=true)
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3")
OLLAMA_RESEARCH_MODEL = os.getenv("OLLAMA_RESEARCH_MODEL", "llama3")
OLLAMA_WRITING_MODEL = os.getenv("OLLAMA_WRITING_MODEL", "llama3")
OLLAMA_PUBLISHING_MODEL = os.getenv("OLLAMA_PUBLISHING_MODEL", "llama3")
OLLAMA_SUPERVISOR_MODEL = os.getenv("OLLAMA_SUPERVISOR_MODEL", "llama3")

# Database Configuration
DB_NAME = os.getenv("DB_NAME", "storybook")
STORIES_COLLECTION = os.getenv("STORIES_COLLECTION", "stories")
RESEARCH_COLLECTION = os.getenv("RESEARCH_COLLECTION", "research")
BIBLE_COLLECTION = os.getenv("BIBLE_COLLECTION", "bible")
FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION", "feedback")
USER_COLLECTION = os.getenv("USER_COLLECTION", "users")

# Application Settings
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BRAINSTORM_SESSION_TIMEOUT = int(os.getenv("BRAINSTORM_SESSION_TIMEOUT", "1800"))  # 30 minutes
HUMAN_REVIEW_TIMEOUT = int(os.getenv("HUMAN_REVIEW_TIMEOUT", "86400"))  # 24 hours
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_WRITERS = int(os.getenv("DEFAULT_WRITERS", "1"))
JOINT_LLM_THRESHOLD = int(os.getenv("JOINT_LLM_THRESHOLD", "2000"))  # Word count threshold for joint LLM

# Operation Modes
class OperationMode(str, Enum):
    CREATE = "create"       # Create new story from scratch
    IMPORT = "import"       # Import existing content
    EDIT = "edit"           # Edit existing story
    CONTINUE = "continue"   # Continue existing story

# Enums for workflow status
class StoryState(str, Enum):
    INITIATED = "initiated"
    BRIEFING = "briefing"
    RESEARCH = "research"
    PLANNING = "planning"
    WRITING = "writing"
    EDITING = "editing"
    REVIEW = "review"
    REVISION = "revision"
    READY_FOR_PUBLISHING = "ready_for_publishing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    ON_HOLD = "on_hold"
    ERROR = "error"

class AgentRole(str, Enum):
    RESEARCHER = "researcher"
    WRITER = "writer"
    EDITOR = "editor"
    PUBLISHER = "publisher"
    RESEARCH_SUPERVISOR = "research_supervisor"
    WRITING_SUPERVISOR = "writing_supervisor"
    PUBLISHING_SUPERVISOR = "publishing_supervisor"
    AUTHOR_RELATIONS = "author_relations"
    STYLE_GUIDE_EDITOR = "style_guide_editor"
    HUMAN_IN_LOOP = "human_in_loop"
    JOINT_WRITER = "joint_writer"   # Special role for collaborative writing

class TeamType(str, Enum):
    RESEARCH = "research"
    WRITING = "writing"
    PUBLISHING = "publishing"
    SUPERVISION = "supervision"
    COORDINATION = "coordination"

class BibleSectionType(str, Enum):
    STYLE_GUIDE = "style_guide"
    CHARACTER_PROFILES = "character_profiles"
    WORLD_BUILDING = "world_building"
    PLOT_ELEMENTS = "plot_elements"
    THEMES = "themes"
    REFERENCE_MATERIAL = "reference_material"
    AUDIENCE_NOTES = "audience_notes"

class FeedbackType(str, Enum):
    CONTENT = "content"
    STYLE = "style"
    STRUCTURE = "structure"
    RESEARCH = "research"
    TECHNICAL = "technical"
    GENERAL = "general"

class StoryStructure(str, Enum):
    THREE_ACT = "three_act"
    FIVE_ACT = "five_act"
    HEROS_JOURNEY = "heros_journey"
    CUSTOM = "custom"

# Story Structure Templates
THREE_ACT_STRUCTURE = {
    "name": "Three-Act Structure",
    "description": "Classic storytelling structure with setup, confrontation, and resolution",
    "acts": [
        {
            "name": "Act I: Setup",
            "description": "Introduce the main characters, setting, and central conflict",
            "components": [
                {"name": "Exposition", "description": "Establish the world and characters"},
                {"name": "Inciting Incident", "description": "Event that sets the story in motion"},
                {"name": "First Plot Point", "description": "Character commits to addressing the central conflict"}
            ]
        },
        {
            "name": "Act II: Confrontation",
            "description": "Character faces obstacles and evolves through conflict",
            "components": [
                {"name": "Rising Action", "description": "Character attempts to resolve conflict but faces complications"},
                {"name": "Midpoint", "description": "Major event that changes the character's perspective"},
                {"name": "Second Plot Point", "description": "Character faces a major setback"}
            ]
        },
        {
            "name": "Act III: Resolution",
            "description": "Climax and conclusion of the story",
            "components": [
                {"name": "Pre-Climax", "description": "Character makes final preparations for the climactic moment"},
                {"name": "Climax", "description": "Final confrontation that resolves the central conflict"},
                {"name": "Denouement", "description": "Wrap up loose ends and show the new normal"}
            ]
        }
    ]
}

FIVE_ACT_STRUCTURE = {
    "name": "Five-Act Structure",
    "description": "Expanded structure with more detailed dramatic arcs",
    "acts": [
        {
            "name": "Act I: Exposition",
            "description": "Set up the story world and introduce characters",
            "components": [
                {"name": "Introduction", "description": "Present the world, protagonist, and other key characters"},
                {"name": "Background", "description": "Provide context for the story"},
                {"name": "Inciting Incident", "description": "Event that disrupts the status quo"}
            ]
        },
        {
            "name": "Act II: Rising Action",
            "description": "Build complications and develop conflict",
            "components": [
                {"name": "Reaction", "description": "Protagonist reacts to the inciting incident"},
                {"name": "Action", "description": "Protagonist makes first attempts to address the situation"},
                {"name": "Complication", "description": "New obstacles emerge that complicate matters"}
            ]
        },
        {
            "name": "Act III: Climax",
            "description": "Central turning point in the story",
            "components": [
                {"name": "Preparation", "description": "Events leading up to the climactic moment"},
                {"name": "Climactic Moment", "description": "The highest point of tension"},
                {"name": "Immediate Aftermath", "description": "Immediate consequences of the climax"}
            ]
        },
        {
            "name": "Act IV: Falling Action",
            "description": "Deal with the consequences of the climax",
            "components": [
                {"name": "Outcomes", "description": "Effects of the climax unfold"},
                {"name": "Complications", "description": "New challenges arising from the climax"},
                {"name": "Approach to Resolution", "description": "Moving toward the story's conclusion"}
            ]
        },
        {
            "name": "Act V: Denouement",
            "description": "Resolve the story and provide closure",
            "components": [
                {"name": "Final Confrontation", "description": "Address any remaining conflicts"},
                {"name": "Resolution", "description": "Tie up loose ends"},
                {"name": "New Status Quo", "description": "Show the new state of the world/characters"}
            ]
        }
    ]
}

HEROS_JOURNEY_STRUCTURE = {
    "name": "Hero's Journey",
    "description": "Joseph Campbell's monomyth structure for transformative adventures",
    "acts": [
        {
            "name": "Act I: Departure",
            "description": "The hero's journey begins",
            "components": [
                {"name": "The Ordinary World", "description": "Establish the hero's normal life and limitations"},
                {"name": "The Call to Adventure", "description": "Hero is presented with a challenge or quest"},
                {"name": "Refusal of the Call", "description": "Hero initially hesitates or refuses"},
                {"name": "Meeting the Mentor", "description": "Hero gains guidance, encouragement, or items"},
                {"name": "Crossing the Threshold", "description": "Hero commits to the adventure"}
            ]
        },
        {
            "name": "Act II: Initiation",
            "description": "The hero faces trials and transformation",
            "components": [
                {"name": "Tests, Allies, and Enemies", "description": "Hero encounters challenges and forms relationships"},
                {"name": "Approach to the Inmost Cave", "description": "Preparations for major challenge"},
                {"name": "The Ordeal", "description": "Hero faces a central crisis and must overcome it"},
                {"name": "Reward", "description": "Hero gains something from the ordeal (object, knowledge, etc.)"}
            ]
        },
        {
            "name": "Act III: Return",
            "description": "The hero completes the journey and returns transformed",
            "components": [
                {"name": "The Road Back", "description": "Hero begins journey back to ordinary world"},
                {"name": "Resurrection", "description": "Final test that applies what the hero has learned"},
                {"name": "Return with the Elixir", "description": "Hero brings back something to benefit the ordinary world"}
            ]
        }
    ]
}

STORY_STRUCTURES = {
    StoryStructure.THREE_ACT: THREE_ACT_STRUCTURE,
    StoryStructure.FIVE_ACT: FIVE_ACT_STRUCTURE,
    StoryStructure.HEROS_JOURNEY: HEROS_JOURNEY_STRUCTURE
}

# Pydantic Models for data structures
class UserRequest(BaseModel):
    """User request for story creation."""
    title: Optional[str] = None
    theme: Optional[str] = None
    genre: Optional[str] = None
    target_audience: Optional[str] = None
    length: Optional[str] = None
    keywords: Optional[List[str]] = None
    style: Optional[str] = None
    special_requirements: Optional[str] = None
    user_id: Optional[str] = None
    deadline: Optional[str] = None
    references: Optional[List[str]] = None
    tone: Optional[str] = None
    story_structure: Optional[StoryStructure] = StoryStructure.THREE_ACT
    num_writers: Optional[int] = 1
    use_joint_llm: Optional[bool] = False
    operation_mode: Optional[OperationMode] = OperationMode.CREATE
    existing_content: Optional[str] = None
    sections_to_edit: Optional[List[str]] = None
    
    def to_prompt_string(self) -> str:
        """Convert request to a formatted string for prompts."""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.theme:
            parts.append(f"Theme: {self.theme}")
        if self.genre:
            parts.append(f"Genre: {self.genre}")
        if self.target_audience:
            parts.append(f"Target Audience: {self.target_audience}")
        if self.length:
            parts.append(f"Length: {self.length}")
        if self.style:
            parts.append(f"Style: {self.style}")
        if self.tone:
            parts.append(f"Tone: {self.tone}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")
        if self.special_requirements:
            parts.append(f"Special Requirements: {self.special_requirements}")
        if self.references:
            parts.append(f"References: {', '.join(self.references)}")
        if self.deadline:
            parts.append(f"Deadline: {self.deadline}")
        if self.story_structure:
            structure_name = STORY_STRUCTURES[self.story_structure]["name"] if self.story_structure in STORY_STRUCTURES else "Custom"
            parts.append(f"Story Structure: {structure_name}")
            
        return "\n".join(parts)

class ResearchItem(BaseModel):
    """Research item containing reference material."""
    source: str
    content: str
    relevance: float = 1.0
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    added_by: Optional[str] = None
    created_at: Optional[str] = None

class BibleSection(BaseModel):
    """A section of the story bible."""
    section_type: BibleSectionType
    title: str
    content: str
    created_by: str
    updated_by: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: int = 1
    tags: List[str] = Field(default_factory=list)
    
class StoryBible(BaseModel):
    """The complete story bible containing all sections."""
    story_id: str
    sections: Dict[str, List[BibleSection]] = Field(default_factory=dict)
    current_version: int = 1
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def add_section(self, section: BibleSection):
        """Add a section to the bible."""
        section_type = section.section_type.value
        if section_type not in self.sections:
            self.sections[section_type] = []
        self.sections[section_type].append(section)
        
    def get_section(self, section_type: BibleSectionType, title: Optional[str] = None) -> List[BibleSection]:
        """Get sections of a specific type, optionally filtered by title."""
        section_type_value = section_type.value
        if section_type_value not in self.sections:
            return []
            
        if title:
            return [s for s in self.sections[section_type_value] if s.title == title]
        return self.sections[section_type_value]

class Character(BaseModel):
    """Character in a story."""
    name: str
    role: str
    description: str
    background: Optional[str] = None
    motivations: Optional[List[str]] = None
    traits: Optional[List[str]] = None
    relationships: Optional[Dict[str, str]] = None

class PlotPoint(BaseModel):
    """Plot point in a story."""
    title: str
    description: str
    sequence: int
    act: Optional[str] = None  # Which act or section this belongs to
    importance: Optional[float] = 1.0  # 0.0 to 1.0
    characters_involved: Optional[List[str]] = None
    setting: Optional[str] = None

class Setting(BaseModel):
    """Setting in a story."""
    name: str
    description: str
    importance: Optional[float] = 1.0
    attributes: Optional[Dict[str, Any]] = None

class StoryOutline(BaseModel):
    """Structure for a story outline."""
    title: str
    summary: str
    structure: StoryStructure = StoryStructure.THREE_ACT
    acts: List[Dict[str, Any]] = Field(default_factory=list)  # Structured according to story structure
    characters: List[Character] = Field(default_factory=list)
    plot_points: List[PlotPoint] = Field(default_factory=list)
    settings: List[Setting] = Field(default_factory=list) 
    themes: List[str] = Field(default_factory=list)
    style_notes: Optional[str] = None
    target_audience: Optional[str] = None
    estimated_length: Optional[str] = None

class StorySection(BaseModel):
    """A section of the story content."""
    id: str
    title: str
    content: str
    sequence: int
    act: Optional[str] = None
    assigned_to: Optional[str] = None  # Agent ID responsible for writing
    status: str = "not_started"  # not_started, in_progress, completed, revised
    word_count: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: int = 1
    
class Feedback(BaseModel):
    """Feedback on story content."""
    feedback_type: FeedbackType
    content: str
    source: str  # agent_id or user_id
    source_role: str
    target_section: Optional[str] = None
    severity: Optional[int] = None  # 1-5, with 5 being critical
    suggestions: Optional[List[str]] = None
    created_at: Optional[str] = None
    
class PublishingMetadata(BaseModel):
    """Metadata for publishing a story."""
    formatted_content: Optional[str] = None
    summary: Optional[str] = None
    seo_keywords: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    feature_image: Optional[str] = None
    teaser: Optional[str] = None
    publish_platform: Optional[str] = None
    publish_date: Optional[str] = None
    audience_targeting: Optional[Dict[str, Any]] = None

class WriterAssignment(BaseModel):
    """Assignment of writer to story sections."""
    writer_id: str
    sections: List[str]  # Section IDs
    status: str = "assigned"  # assigned, in_progress, completed
    joint_llm: bool = False  # Whether to use joint LLM for difficult sections

class Story(BaseModel):
    """Complete story with metadata."""
    id: Optional[str] = None
    title: str
    content: str = ""
    outline: Optional[StoryOutline] = None
    sections: List[StorySection] = Field(default_factory=list)
    bible: Optional[StoryBible] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    state: StoryState = StoryState.INITIATED
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    user_id: Optional[str] = None
    research: List[ResearchItem] = Field(default_factory=list)
    feedback: List[Feedback] = Field(default_factory=list)
    publishing_metadata: Optional[PublishingMetadata] = None
    structure: StoryStructure = StoryStructure.THREE_ACT
    operation_mode: OperationMode = OperationMode.CREATE
    writer_assignments: List[WriterAssignment] = Field(default_factory=list)
    
    def add_research(self, research_item: ResearchItem):
        """Add a research item to the story."""
        self.research.append(research_item)
        
    def add_feedback(self, feedback_item: Feedback):
        """Add a feedback item to the story."""
        self.feedback.append(feedback_item)
        
    def add_section(self, section: StorySection):
        """Add a section to the story."""
        self.sections.append(section)
        
    def get_section_by_id(self, section_id: str) -> Optional[StorySection]:
        """Get a section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
        
    def assign_writer(self, writer_id: str, section_ids: List[str], joint_llm: bool = False):
        """Assign a writer to story sections."""
        # Check if writer already has assignments
        for assignment in self.writer_assignments:
            if assignment.writer_id == writer_id:
                # Update existing assignment
                assignment.sections.extend([s for s in section_ids if s not in assignment.sections])
                assignment.joint_llm = joint_llm
                return
                
        # Create new assignment
        self.writer_assignments.append(
            WriterAssignment(
                writer_id=writer_id,
                sections=section_ids,
                joint_llm=joint_llm
            )
        )
