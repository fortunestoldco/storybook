import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Union, Literal
import uuid
import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.checkpoint.memory import MemorySaver

from storybook.config import (
    StoryState,
    AgentRole,
    TeamType,
    BibleSectionType,
    FeedbackType,
    StoryStructure,
    OperationMode,
    UserRequest,
    ResearchItem,
    BibleSection,
    StoryBible,
    Character,
    PlotPoint,
    Setting,
    StoryOutline,
    StorySection,
    Feedback,
    PublishingMetadata,
    Story,
    WriterAssignment,
    STORY_STRUCTURES,
    USE_GPU,
    USE_OLLAMA,
)

from storybook.state import (
    GraphState,
    AgentState,
    ResearchAgentState,
    WritingAgentState,
    JointWriterAgentState,
    EditingAgentState,
    PublishingAgentState,
    SupervisorAgentState,
    AuthorRelationsAgentState,
    HumanInLoopState,
    StyleGuideEditorState,
)


from storybook.tools import ToolsService

from storybook.utils import (
    generate_id,
    current_timestamp,
    extract_json_from_text,
    format_message_history,
    clean_and_format_text,
    create_task_id,
    parse_feedback,
    format_agent_response,
    validate_story_structure,
    prepare_human_review_prompt,
    format_brainstorm_session,
    get_story_structure_template,
    create_model_instance,
    create_section_structure_from_template,
    distribute_sections_to_writers,
)
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(
    Image(
        self.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
    )
)
