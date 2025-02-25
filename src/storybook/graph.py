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
    SupervisorAgentState,
    AuthorRelationsAgentState,
    HumanInLoopState,
    StyleGuideEditorState,
)

from storybook.prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    EDITOR_SYSTEM_PROMPT,
    WRITING_SUPERVISOR_SYSTEM_PROMPT,
    AUTHOR_RELATIONS_SYSTEM_PROMPT,
    HUMAN_IN_LOOP_SYSTEM_PROMPT,
    STYLE_GUIDE_EDITOR_SYSTEM_PROMPT,
    RESEARCH_TASK_PROMPT,
    WRITING_TASK_PROMPT,
    EDITING_TASK_PROMPT,
    REVIEW_TASK_PROMPT,
    BIBLE_UPDATE_PROMPT,
    BRAINSTORM_SESSION_PROMPT,
    HUMAN_REVIEW_PROMPT,
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

# Initialize tool executor
tool_executor = ToolExecutor()


# Agent Factory Functions
def create_researcher_agent(agent_id: str) -> Any:
    """Create a researcher agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("research", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=RESEARCH_TOOLS, system_message=RESEARCHER_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=RESEARCH_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_research_supervisor_agent(agent_id: str) -> Any:
    """Create a research supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        system_message=RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    )

    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
    )


def create_writer_agent(agent_id: str) -> Any:
    """Create a writer agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("writing", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=WRITING_TOOLS, system_message=WRITER_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=WRITING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_joint_writer_agent(agent_id: str, component_writer_ids: List[str] = None) -> Any:
    """Create a joint writer agent that combines the power of multiple models."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    # This agent uses a more powerful model since it handles complex sections
    llm = create_model_instance(
        "writing", use_local=False
    )  # Always use API model for joint writing

    # Enhanced system prompt for joint writing
    joint_writer_prompt = f"""
    {WRITER_SYSTEM_PROMPT}
    
    As a joint writer, you are responsible for handling particularly complex or important story sections that
    require exceptional attention to detail and creative skill. You may need to integrate contributions from
    multiple individual writers to create a cohesive, high-quality section.
    
    Focus on:
    - Ensuring narrative continuity across writers' contributions
    - Maintaining consistent character voices and relationships
    - Creating seamless transitions between ideas and scenes
    - Elevating the overall literary quality
    
    Your work represents the highest standard of storytelling for this project.
    """

    prompt = create_openai_tools_agent(
        llm=llm, tools=WRITING_TOOLS, system_message=joint_writer_prompt
    )

    return AgentExecutor(
        agent=prompt, tools=WRITING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_editor_agent(agent_id: str) -> Any:
    """Create an editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("writing", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=EDITING_TOOLS, system_message=EDITOR_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=EDITING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_writing_supervisor_agent(agent_id: str) -> Any:
    """Create a writing supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        system_message=WRITING_SUPERVISOR_SYSTEM_PROMPT,
    )

    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
    )


def create_author_relations_agent(agent_id: str) -> Any:
    """Create an author relations agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("author_relations", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=AUTHOR_RELATIONS_TOOLS, system_message=AUTHOR_RELATIONS_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=AUTHOR_RELATIONS_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_style_guide_editor_agent(agent_id: str) -> Any:
    """Create a style guide editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("writing", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=BIBLE_EDITOR_TOOLS, system_message=STYLE_GUIDE_EDITOR_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=BIBLE_EDITOR_TOOLS, verbose=True, handle_parsing_errors=True
    )


# LangGraph Node Functions
def initialize_workflow(state: GraphState) -> GraphState:
    """Initialize the workflow with agents and initial state based on operation mode."""
    new_state = state.copy()

    # Set creation timestamp if not set
    if not new_state.created_at:
        new_state.created_at = current_timestamp()
        new_state.last_updated = current_timestamp()

    # Create a unique story ID if not provided
    if not new_state.story_id:
        new_state.story_id = generate_id("story")

    # Set operation mode if not set
    if not new_state.operation_mode and new_state.user_request:
        new_state.operation_mode = new_state.user_request.operation_mode or OperationMode.CREATE

    # Set number of writers from request
    num_writers = 1
    use_joint_llm = False
    if new_state.user_request:
        num_writers = new_state.user_request.num_writers or 1
        use_joint_llm = new_state.user_request.use_joint_llm or False

    new_state.num_writers = max(1, min(5, num_writers))  # Limit between 1-5 writers
    new_state.use_joint_llm = use_joint_llm

    # Set story structure
    if new_state.user_request and new_state.user_request.story_structure:
        new_state.story_structure = new_state.user_request.story_structure
        new_state.structure_template = get_story_structure_template(new_state.story_structure)
    else:
        new_state.story_structure = StoryStructure.THREE_ACT
        new_state.structure_template = get_story_structure_template(StoryStructure.THREE_ACT)

    # Initialize teams if they don't exist
    # Research team
    if not new_state.research_team:
        researcher_id = generate_id("researcher")
        new_state.research_team[researcher_id] = ResearchAgentState(
            agent_id=researcher_id, agent_role=AgentRole.RESEARCHER, status="idle"
        )

    # Research supervisor
    if not any(
        sup.agent_role == AgentRole.RESEARCH_SUPERVISOR for sup in new_state.supervisors.values()
    ):
        supervisor_id = generate_id("rsup")
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.RESEARCH_SUPERVISOR,
            team_type=TeamType.RESEARCH,
            status="idle",
            supervised_agents=list(new_state.research_team.keys()),
        )

    # Writing team - create multiple writer agents based on num_writers
    if not new_state.writing_team:
        for i in range(new_state.num_writers):
            writer_id = generate_id(f"writer_{i+1}")
            new_state.writing_team[writer_id] = WritingAgentState(
                agent_id=writer_id, agent_role=AgentRole.WRITER, status="idle"
            )

        # Create joint writer if enabled
        if new_state.use_joint_llm:
            joint_writer_id = generate_id("joint_writer")
            new_state.joint_writers[joint_writer_id] = JointWriterAgentState(
                agent_id=joint_writer_id,
                agent_role=AgentRole.JOINT_WRITER,
                status="idle",
                component_writers=list(new_state.writing_team.keys()),
                is_joint_llm=True,
            )

    # Editing team
    if not new_state.editing_team:
        editor_id = generate_id("editor")
        new_state.editing_team[editor_id] = EditingAgentState(
            agent_id=editor_id, agent_role=AgentRole.EDITOR, status="idle"
        )

    # Writing supervisor
    if not any(
        sup.agent_role == AgentRole.WRITING_SUPERVISOR for sup in new_state.supervisors.values()
    ):
        supervisor_id = generate_id("wsup")
        writer_ids = list(new_state.writing_team.keys())
        joint_writer_ids = list(new_state.joint_writers.keys())
        editor_ids = list(new_state.editing_team.keys())

        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.WRITING_SUPERVISOR,
            team_type=TeamType.WRITING,
            status="idle",
            supervised_agents=writer_ids + joint_writer_ids + editor_ids,
        )

    # Special agents
    # Author relations agent
    if not new_state.author_relations:
        author_relations_id = generate_id("author_relations")
        new_state.author_relations[author_relations_id] = AuthorRelationsAgentState(
            agent_id=author_relations_id, agent_role=AgentRole.AUTHOR_RELATIONS, status="idle"
        )

    # Style guide editor
    if not new_state.style_guide_editor:
        style_guide_id = generate_id("style_guide")
        new_state.style_guide_editor[style_guide_id] = StyleGuideEditorState(
            agent_id=style_guide_id, agent role=AgentRole.STYLE_GUIDE_EDITOR, status="idle"
        )

    # Human in the loop
    if not new_state.human_in_loop:
        human_id = generate_id("human")
        new_state.human_in_loop[human_id] = HumanInLoopState(
            agent_id=human_id, agent role=AgentRole.HUMAN_IN_LOOP, status="ready"
        )

    # Initialize story state based on operation mode
    if new_state.operation_mode == OperationMode.CREATE:
        # For new story creation
        new_state.current_state = StoryState.INITIATED
        if not new_state.story:
            title = new_state.user_request.title if new_state.user_request else "Untitled Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content="",
                state=StoryState.INITIATED,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    elif new_state.operation_mode == OperationMode.IMPORT:
        # For importing existing content
        new_state.current_state = StoryState.PLANNING
        if new_state.user_request and new_state.user_request.existing_content:
            new_state.imported_content = new_state.user_request.existing_content

            # Create a story object with the imported content
            title = new_state.user_request.title if new_state.user_request else "Imported Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content=new_state.imported_content,
                state=StoryState.PLANNING,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    elif new_state.operation_mode in [OperationMode.EDIT, OperationMode.CONTINUE]:
        # For editing or continuing an existing story
        new_state.current_state = StoryState.REVISION
        if new_state.user_request:
            if new_state.user_request.existing_content:
                new_state.imported_content = new_state.user_request.existing_content
            if new_state.user_request.sections_to_edit:
                new_state.sections_to_edit = new_state.user_request.sections_to_edit

            # Create a story object with the existing content
            title = new_state.user_request.title if new_state.user_request else "Edited Story"
            new_state.story = Story(
                id=new_state.story_id,
                title=title,
                content=new_state.imported_content or "",
                state=StoryState.REVISION,
                created_at=current_timestamp(),
                updated_at=current_timestamp(),
                user_id=new_state.user_request.user_id if new_state.user_request else None,
                structure=new_state.story_structure,
                operation_mode=new_state.operation_mode,
            )

    # Initialize empty bible if none exists
    if not new_state.bible:
        new_state.bible = StoryBible(
            story_id=new_state.story_id,
            created_at=current_timestamp(),
            updated_at=current_timestamp(),
        )

    return new_state


def process_user_request(state: GraphState, user_request: UserRequest) -> GraphState:
    """Process a new user request to initiate the workflow."""
    new_state = state.copy()

    # Store the user request
    new_state.user_request = user_request

    # Set the operation mode
    new_state.operation_mode = user_request.operation_mode or OperationMode.CREATE

    # Set story structure
    if user_request.story_structure:
        new_state.story_structure = user_request.story_structure
        new_state.structure_template = get_story_structure_template(user_request.story_structure)

    # Handle different operation modes
    if new_state.operation_mode == OperationMode.CREATE:
        # Standard creation flow
        new_state.update_story_state(StoryState.BRIEFING)

        # Create a task for the author relations agent to conduct initial briefing
        author_relations_id = next(iter(new_state.author_relations.keys()))
        briefing_task_id = create_task_id("briefing", author_relations_id)

        # Add task to state
        new_state.add_task(
            task_id=briefing_task_id,
            agent_id=author_relations_id,
            task_type="briefing",
            description="Conduct initial briefing with the client to gather story requirements",
            data={"user_request": user_request.dict()},
        )

        # Update author relations agent state
        author_agent = new_state.author_relations[author_relations_id]
        author_agent.update_status("working", "conducting_briefing")

        # Add initial message to communication log
        new_state.add_message(
            sender="system",
            recipient=author_relations_id,
            content=f"New story request received. Please conduct an initial briefing with the client.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nStory Structure: {new_state.story_structure}",
            metadata={"task_id": briefing_task_id, "task_type": "briefing"},
        )

    elif new_state.operation_mode == OperationMode.IMPORT:
        # Import existing content and start analyzing it
        new_state.update_story_state(StoryState.PLANNING)
        new_state.imported_content = user_request.existing_content

        # Create a task for analyzing the imported content
        style_guide_id = next(iter(new_state.style_guide_editor.keys()))
        analysis_task_id = create_task_id("analyze_import", style_guide_id)

        # Add task to state
        new_state.add_task(
            task_id=analysis_task_id,
            agent_id=style_guide_id,
            task_type="analyze_import",
            description="Analyze imported content and create initial bible entries",
            data={
                "imported_content": user_request.existing_content,
                "user_request": user_request.dict(),
            },
        )

        # Update style guide editor state
        style_guide = new_state.style_guide_editor[style_guide_id]
        style_guide.update_status("working", "analyzing_import")

        # Add initial message to communication log
        new_state.add_message(
            sender="system",
            recipient=style_guide_id,
            content=f"Imported content received. Please analyze it and create initial bible entries.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nImported Content Preview:\n{user_request.existing_content[:500]}...",
            metadata={"task_id": analysis_task_id, "task_type": "analyze_import"},
        )

    elif
