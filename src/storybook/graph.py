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
from langgraph.checkpoint import Checkpoint

from config import (
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

from agents.state import (
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

from agents.prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
    EDITOR_SYSTEM_PROMPT,
    WRITING_SUPERVISOR_SYSTEM_PROMPT,
    PUBLISHER_SYSTEM_PROMPT,
    PUBLISHING_SUPERVISOR_SYSTEM_PROMPT,
    AUTHOR_RELATIONS_SYSTEM_PROMPT,
    HUMAN_IN_LOOP_SYSTEM_PROMPT,
    STYLE_GUIDE_EDITOR_SYSTEM_PROMPT,
    RESEARCH_TASK_PROMPT,
    WRITING_TASK_PROMPT,
    EDITING_TASK_PROMPT,
    PUBLISHING_TASK_PROMPT,
    REVIEW_TASK_PROMPT,
    BIBLE_UPDATE_PROMPT,
    BRAINSTORM_SESSION_PROMPT,
    HUMAN_REVIEW_PROMPT,
)

from agents.tools import (
    RESEARCH_TOOLS,
    WRITING_TOOLS,
    EDITING_TOOLS,
    PUBLISHING_TOOLS,
    SUPERVISOR_TOOLS,
    AUTHOR_RELATIONS_TOOLS,
    BIBLE_EDITOR_TOOLS,
    HUMAN_IN_LOOP_TOOLS,
)

from agents.utils import (
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


def create_publisher_agent(agent_id: str) -> Any:
    """Create a publisher agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("publishing", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm, tools=PUBLISHING_TOOLS, system_message=PUBLISHER_SYSTEM_PROMPT
    )

    return AgentExecutor(
        agent=prompt, tools=PUBLISHING_TOOLS, verbose=True, handle_parsing_errors=True
    )


def create_publishing_supervisor_agent(agent_id: str) -> Any:
    """Create a publishing supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent

    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)

    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
        system_message=PUBLISHING_SUPERVISOR_SYSTEM_PROMPT,
    )

    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
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

    # Publishing team
    if not new_state.publishing_team:
        publisher_id = generate_id("publisher")
        new_state.publishing_team[publisher_id] = PublishingAgentState(
            agent_id=publisher_id, agent_role=AgentRole.PUBLISHER, status="idle"
        )

    # Publishing supervisor
    if not any(
        sup.agent_role == AgentRole.PUBLISHING_SUPERVISOR for sup in new_state.supervisors.values()
    ):
        supervisor_id = generate_id("psup")
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.PUBLISHING_SUPERVISOR,
            team_type=TeamType.PUBLISHING,
            status="idle",
            supervised_agents=list(new_state.publishing_team.keys()),
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
            agent_id=style_guide_id, agent_role=AgentRole.STYLE_GUIDE_EDITOR, status="idle"
        )

    # Human in the loop
    if not new_state.human_in_loop:
        human_id = generate_id("human")
        new_state.human_in_loop[human_id] = HumanInLoopState(
            agent_id=human_id, agent_role=AgentRole.HUMAN_IN_LOOP, status="ready"
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
            content=f"New story request received. Please conduct an initial briefing with the client.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nStory Structure: {new_state.story_structure.value}",
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

    elif new_state.operation_mode in [OperationMode.EDIT, OperationMode.CONTINUE]:
        # Start the edit/continue process
        new_state.update_story_state(StoryState.REVISION)
        new_state.imported_content = user_request.existing_content
        new_state.sections_to_edit = user_request.sections_to_edit or []

        # Create a task for planning the edits
        writing_supervisor_id = next(
            sup_id
            for sup_id, sup in new_state.supervisors.items()
            if sup.agent_role == AgentRole.WRITING_SUPERVISOR
        )

        planning_task_id = create_task_id("plan_edits", writing_supervisor_id)

        # Add task to state
        new_state.add_task(
            task_id=planning_task_id,
            agent_id=writing_supervisor_id,
            task_type="plan_edits",
            description="Plan edits or continuation for the existing content",
            data={
                "existing_content": user_request.existing_content,
                "sections_to_edit": user_request.sections_to_edit or [],
                "user_request": user_request.dict(),
                "is_continuation": new_state.operation_mode == OperationMode.CONTINUE,
            },
        )

        # Update supervisor state
        supervisor = new_state.supervisors[writing_supervisor_id]
        supervisor.update_status("working", "planning_edits")

        # Add initial message to communication log
        operation_type = (
            "continuation" if new_state.operation_mode == OperationMode.CONTINUE else "edits"
        )
        new_state.add_message(
            sender="system",
            recipient=writing_supervisor_id,
            content=f"Request for {operation_type} received. Please plan the necessary work.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nExisting Content Preview:\n{user_request.existing_content[:500]}...\n\nSections to Edit: {', '.join(user_request.sections_to_edit or ['All sections' if new_state.operation_mode == OperationMode.EDIT else 'Continue from end'])}",
            metadata={"task_id": planning_task_id, "task_type": f"plan_{operation_type}"},
        )

    return new_state


def analyze_imported_content(state: GraphState) -> GraphState:
    """Analyze imported content and create initial bible entries."""
    new_state = state.copy()

    # This node handles the IMPORT operation mode
    if new_state.operation_mode != OperationMode.IMPORT:
        return new_state

    # Get style guide editor and their task
    style_guide_id = next(iter(new_state.style_guide_editor.keys()))
    style_guide = new_state.style_guide_editor[style_guide_id]

    # Find the analysis task message
    analysis_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == style_guide_id
        and msg.get("metadata", {}).get("task_type") == "analyze_import"
    ]

    if not analysis_messages:
        # No analysis task found
        return new_state

    latest_analysis_message = analysis_messages[-1]
    task_id = latest_analysis_message.get("metadata", {}).get("task_id")

    # Create the style guide editor agent
    style_guide_agent = create_style_guide_editor_agent(style_guide_id)

    try:
        # Execute analysis task
        response = style_guide_agent.invoke(
            {
                "input": latest_analysis_message["content"],
                "agent_id": style_guide_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update style guide editor state
            style_guide.update_status("completed", "import_analyzed")

            # Add message to communication log
            new_state.add_message(
                sender=style_guide_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "import_analysis_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"analysis_output": response["output"]}
            )

            # Create a reverse outline task for the writing supervisor
            supervisor_id = next(
                sup_id
                for sup_id, sup in new_state.supervisors.items()
                if sup.agent_role == AgentRole.WRITING_SUPERVISOR
            )

            outline_task_id = create_task_id("reverse_outline", supervisor_id)

            # Format the task
            outline_prompt = f"""
            # Reverse Outline Task
            
            An existing story has been imported, and we need to create a structural outline from it.
            
            ## Imported Content
            {new_state.imported_content[:2000]}...
            
            ## Initial Analysis
            {response["output"]}
            
            Please create a reverse outline of this story that includes:
            1. Story structure (identify acts, key scenes, plot points)
            2. Character list and descriptions based on the content
            3. Setting information extracted from the text
            4. Themes and motifs present in the work
            5. Style and tone analysis
            
            This reverse outline will guide our work in extending or modifying the content.
            """

            # Add task to state
            new_state.add_task(
                task_id=outline_task_id,
                agent_id=supervisor_id,
                task_type="reverse_outline",
                description="Create a reverse outline from imported content",
                data={
                    "imported_content": new_state.imported_content,
                    "analysis": response["output"],
                },
            )

            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", "creating_reverse_outline")

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=outline_prompt,
                metadata={"task_id": outline_task_id, "task_type": "reverse_outline"},
            )
    except Exception as e:
        error_msg = f"Error analyzing imported content: {str(e)}"
        new_state.add_error("style_guide_editor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        style_guide.update_status("error", "import_analysis_error")

        # Still try to move to reverse outlining
        supervisor_id = next(
            sup_id
            for sup_id, sup in new_state.supervisors.items()
            if sup.agent_role == AgentRole.WRITING_SUPERVISOR
        )

        outline_task_id = create_task_id("reverse_outline", supervisor_id)

        outline_prompt = f"""
        # Reverse Outline Task - Error Recovery
        
        There was an error analyzing the imported content, but we still need to create a structural outline.
        
        ## Imported Content
        {new_state.imported_content[:2000]}...
        
        Please create a reverse outline of this story that includes:
        1. Story structure (identify acts, key scenes, plot points)
        2. Character list and descriptions based on the content
        3. Setting information extracted from the text
        4. Themes and motifs present in the work
        5. Style and tone analysis
        
        This reverse outline will guide our work in extending or modifying the content.
        """

        # Add task to state
        new_state.add_task(
            task_id=outline_task_id,
            agent_id=supervisor_id,
            task_type="reverse_outline",
            description="Create a reverse outline from imported content (error recovery)",
            data={"imported_content": new_state.imported_content, "error_recovery": True},
        )

        # Update supervisor state
        supervisor = new_state.supervisors[supervisor_id]
        supervisor.update_status("working", "creating_reverse_outline")

        # Add message to communication log
        new_state.add_message(
            sender="system",
            recipient=supervisor_id,
            content=outline_prompt,
            metadata={"task_id": outline_task_id, "task_type": "reverse_outline"},
        )

    return new_state


def create_reverse_outline(state: GraphState) -> GraphState:
    """Create a reverse outline from imported content."""
    new_state = state.copy()

    # This node is for the IMPORT operation mode
    if new_state.operation_mode != OperationMode.IMPORT:
        return new_state

    # Get supervisor and their task
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.WRITING_SUPERVISOR
    )
    supervisor = new_state.supervisors[supervisor_id]

    # Find the reverse outline task message
    outline_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == supervisor_id
        and msg.get("metadata", {}).get("task_type") == "reverse_outline"
    ]

    if not outline_messages:
        # No outline task found
        return new_state

    latest_outline_message = outline_messages[-1]
    task_id = latest_outline_message.get("metadata", {}).get("task_id")

    # Create the supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)

    try:
        # Execute reverse outline task
        response = supervisor_agent.invoke(
            {
                "input": latest_outline_message["content"],
                "agent_id": supervisor_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "reverse_outline_completed")

            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "reverse_outline_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"outline_output": response["output"]}
            )

            # Create bible sections from the reverse outline
            style_guide_id = next(iter(new_state.style_guide_editor.keys()))

            # Create bible update task
            bible_task_id = create_task_id("bible_update", style_guide_id)

            # Format bible update task
            update_prompt = BIBLE_UPDATE_PROMPT.format(
                task_description="Create bible sections based on the reverse outline of imported content.",
                section_type=BibleSectionType.REFERENCE_MATERIAL.value,
                section_title="Imported Content Analysis",
                current_content="No existing content for this section.",
                new_information=response["output"],
                related_sections="",
            )

            # Add task to state
            new_state.add_task(
                task_id=bible_task_id,
                agent_id=style_guide_id,
                task_type="bible_update",
                description="Create bible sections from reverse outline",
                data={"reverse_outline": response["output"]},
            )

            # Update style guide editor state
            style_guide = new_state.style_guide_editor[style_guide_id]
            style_guide.update_status("working", "updating_bible")

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=style_guide_id,
                content=update_prompt,
                metadata={"task_id": bible_task_id, "task_type": "bible_update"},
            )

            # After reverse outline, set up for continuation or extension
            new_state.update_story_state(StoryState.PLANNING)
    except Exception as e:
        error_msg = f"Error creating reverse outline: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        supervisor.update_status("error", "reverse_outline_error")

        # Still try to move to planning phase
        new_state.update_story_state(StoryState.PLANNING)

    return new_state


def plan_edit_continuation(state: GraphState) -> GraphState:
    """Plan edits or continuation for existing content."""
    new_state = state.copy()

    # This node handles the EDIT or CONTINUE operation modes
    if new_state.operation_mode not in [OperationMode.EDIT, OperationMode.CONTINUE]:
        return new_state

    # Get supervisor and their task
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.WRITING_SUPERVISOR
    )
    supervisor = new_state.supervisors[supervisor_id]

    # Find the planning task message
    is_continuation = new_state.operation_mode == OperationMode.CONTINUE
    task_type = "plan_continuation" if is_continuation else "plan_edits"

    planning_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == supervisor_id
        and msg.get("metadata", {}).get("task_type") in ["plan_edits", "plan_continuation"]
    ]

    if not planning_messages:
        # No planning task found
        return new_state

    latest_planning_message = planning_messages[-1]
    task_id = latest_planning_message.get("metadata", {}).get("task_id")

    # Create the supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)

    try:
        # Execute planning task
        response = supervisor_agent.invoke(
            {
                "input": latest_planning_message["content"],
                "agent_id": supervisor_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "edit_plan_completed")

            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"plan_output": response["output"]}
            )

            # Create a task for the style guide editor to update the bible
            style_guide_id = next(iter(new_state.style_guide_editor.keys()))

            bible_task_id = create_task_id("bible_update", style_guide_id)

            # Format bible update task
            update_prompt = BIBLE_UPDATE_PROMPT.format(
                task_description=f"Create or update bible sections based on the {'continuation' if is_continuation else 'edit'} plan.",
                section_type=BibleSectionType.REFERENCE_MATERIAL.value,
                section_title=f"{'Continuation' if is_continuation else 'Edit'} Plan",
                current_content="No existing content for this section.",
                new_information=response["output"],
                related_sections="",
            )

            # Add task to state
            new_state.add_task(
                task_id=bible_task_id,
                agent_id=style_guide_id,
                task_type="bible_update",
                description=f"Update bible for {'continuation' if is_continuation else 'edits'}",
                data={"plan": response["output"], "is_continuation": is_continuation},
            )

            # Update style guide editor state
            style_guide = new_state.style_guide_editor[style_guide_id]
            style_guide.update_status("working", "updating_bible")

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=style_guide_id,
                content=update_prompt,
                metadata={"task_id": bible_task_id, "task_type": "bible_update"},
            )

            # Identify sections to edit/continue and assign to writers
            sections_to_edit = new_state.sections_to_edit
            if not sections_to_edit and is_continuation:
                sections_to_edit = ["continuation"]

            # We'll parse these from the supervisor's response
            # For now, assuming we're editing/continuing the whole content
            writer_ids = list(new_state.writing_team.keys())

            if len(writer_ids) > 0:
                # For simplicity, assign all work to the first writer
                # In a real implementation, you'd distribute work more intelligently
                primary_writer_id = writer_ids[0]

                if is_continuation:
                    # For continuation, create a single new section
                    new_state.add_task(
                        task_id=create_task_id("continuation", primary_writer_id),
                        agent_id=primary_writer_id,
                        task_type="write_continuation",
                        description="Continue the story from where it left off",
                        data={
                            "existing_content": new_state.imported_content,
                            "continuation_plan": response["output"],
                        },
                    )

                    # Update writer state
                    new_state.writing_team[primary_writer_id].update_status(
                        "assigned", "continuation_assigned"
                    )
                else:
                    # For editing, assign the edit task
                    new_state.add_task(
                        task_id=create_task_id("edit", primary_writer_id),
                        agent_id=primary_writer_id,
                        task_type="edit_content",
                        description="Edit the existing content based on the plan",
                        data={
                            "existing_content": new_state.imported_content,
                            "sections_to_edit": sections_to_edit,
                            "edit_plan": response["output"],
                        },
                    )

                    # Update writer state
                    new_state.writing_team[primary_writer_id].update_status(
                        "assigned", "edit_assigned"
                    )
    except Exception as e:
        error_msg = f"Error planning {'continuation' if is_continuation else 'edits'}: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        supervisor.update_status("error", "planning_error")

    return new_state


def execute_edit_continuation(state: GraphState) -> GraphState:
    """Execute the edit or continuation task with a writer agent."""
    new_state = state.copy()

    # This node handles the EDIT or CONTINUE operation modes
    if new_state.operation_mode not in [OperationMode.EDIT, OperationMode.CONTINUE]:
        return new_state

    # Get assigned writer and their task
    is_continuation = new_state.operation_mode == OperationMode.CONTINUE
    task_type = "write_continuation" if is_continuation else "edit_content"

    # Find writers with assigned edit/continuation tasks
    writer_id = None
    assigned_writers = [
        w_id
        for w_id, w in new_state.writing_team.items()
        if w.status in ["assigned", "working"]
        and ("continuation" in w.last_action if is_continuation else "edit" in w.last_action)
    ]

    if not assigned_writers:
        # No assigned writers found
        return new_state

    writer_id = assigned_writers[0]
    writer = new_state.writing_team[writer_id]

    # Find the task message
    edit_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == writer_id and msg.get("metadata", {}).get("task_type") == task_type
    ]

    if not edit_messages:
        # Create a task message if none exists
        if is_continuation:
            prompt = f"""
            # Continuation Task
            
            Please continue the story from where it left off. Here is the existing content for context:
            
            {new_state.imported_content[:1000]}...
            
            Continue the narrative in a way that maintains consistency with the existing style, characters, and plot.
            """
        else:
            prompt = f"""
            # Edit Task
            
            Please edit the following content according to the requirements:
            
            {new_state.imported_content[:1000]}...
            
            Sections to edit: {', '.join(new_state.sections_to_edit) if new_state.sections_to_edit else 'All content'}
            
            Make edits that improve the quality while maintaining the core story elements.
            """

        task_id = create_task_id(task_type, writer_id)

        new_state.add_message(
            sender="system",
            recipient=writer_id,
            content=prompt,
            metadata={"task_id": task_id, "task_type": task_type},
        )

        # Update state and return - will process in next iteration
        writer.update_status("working", f"{task_type}_in_progress")
        return new_state

    # Process the existing task
    latest_task_message = edit_messages[-1]
    task_id = latest_task_message.get("metadata", {}).get("task_id")

    # Create writer agent
    writer_agent = create_writer_agent(writer_id)

    try:
        # Execute edit/continuation task
        response = writer_agent.invoke(
            {
                "input": latest_task_message["content"],
                "agent_id": writer_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update writer state
            writer.update_status("completed", f"{task_type}_completed")

            # Add message to communication log
            new_state.add_message(
                sender=writer_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"content": response["output"]}
            )

            # Send to supervisor for review
            supervisor_id = next(
                sup_id
                for sup_id, sup in new_state.supervisors.items()
                if sup.agent_role == AgentRole.WRITING_SUPERVISOR
            )

            review_task_id = create_task_id("review", supervisor_id)

            # Format review task
            review_prompt = REVIEW_TASK_PROMPT.format(
                task_description=f"Review the {'continued' if is_continuation else 'edited'} content.",
                content_type=f"{'Continuation' if is_continuation else 'Edit'}",
                content_summary=clean_and_format_text(response["output"]),
                review_criteria=(
                    f"1. {'Continuation flows naturally from the original' if is_continuation else 'Edits improve the content while maintaining integrity'}\n"
                    f"2. Consistency with existing characters, tone, and style\n"
                    f"3. Overall quality and coherence\n"
                    f"4. Adherence to the {'continuation' if is_continuation else 'edit'} plan\n"
                    f"5. Any issues or areas for improvement"
                ),
                context=f"Original content: {new_state.imported_content[:500]}...",
                previous_feedback="",
            )

            # Add review task to state
            new_state.add_task(
                task_id=review_task_id,
                agent_id=supervisor_id,
                task_type=f"review_{task_type}",
                description=f"Review {'continuation' if is_continuation else 'edits'}",
                data={
                    "original_content": new_state.imported_content,
                    "new_content": response["output"],
                    "writer_id": writer_id,
                },
            )

            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", f"reviewing_{task_type}")

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=review_prompt,
                metadata={"task_id": review_task_id, "task_type": f"review_{task_type}"},
            )
    except Exception as e:
        error_msg = f"Error during {task_type}: {str(e)}"
        new_state.add_error("writer", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        writer.update_status("error", f"{task_type}_error")

    return new_state


def review_edit_continuation(state: GraphState) -> GraphState:
    """Review edits or continuation with the writing supervisor."""
    new_state = state.copy()

    # This node handles the EDIT or CONTINUE operation modes
    if new_state.operation_mode not in [OperationMode.EDIT, OperationMode.CONTINUE]:
        return new_state

    # Get supervisor and their task
    is_continuation = new_state.operation_mode == OperationMode.CONTINUE
    task_type = f"review_{'write_continuation' if is_continuation else 'edit_content'}"

    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.WRITING_SUPERVISOR
    )
    supervisor = new_state.supervisors[supervisor_id]

    # Find the review task message
    review_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == supervisor_id
        and msg.get("metadata", {}).get("task_type") == task_type
    ]

    if not review_messages:
        # No review task found
        return new_state

    latest_review_message = review_messages[-1]
    task_id = latest_review_message.get("metadata", {}).get("task_id")

    # Create supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)

    try:
        # Execute review task
        response = supervisor_agent.invoke(
            {
                "input": latest_review_message["content"],
                "agent_id": supervisor_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", f"{task_type}_completed")

            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"review_output": response["output"]}
            )

            # Check if edits/continuation are approved
            is_approved = (
                "approved" in response["output"].lower() or "accept" in response["output"].lower()
            )

            if is_approved:
                # Update final content
                content_type = (
                    "write_continuation_complete" if is_continuation else "edit_content_complete"
                )
                content_messages = [
                    msg
                    for msg in new_state.messages
                    if msg.get("metadata", {}).get("task_type") == content_type
                ]

                if content_messages:
                    final_content = content_messages[-1]["content"]

                    # For continuation, append to existing content
                    if is_continuation:
                        updated_content = f"{new_state.imported_content}\n\n{final_content}"
                    else:
                        # For edits, replace content
                        updated_content = final_content

                    # Update story object
                    if new_state.story:
                        new_state.story.content = updated_content
                        new_state.story.updated_at = current_timestamp()

                    # Move to next appropriate phase
                    if is_continuation:
                        new_state.update_story_state(StoryState.EDITING)
                    else:
                        new_state.update_story_state(StoryState.READY_FOR_PUBLISHING)

                    # Should we request human approval?
                    human_approval_needed = True

                    if human_approval_needed:
                        human_id = next(iter(new_state.human_in_loop.keys()))

                        review_type = (
                            "continuation_approval" if is_continuation else "edit_approval"
                        )

                        # Request human review
                        from agents.tools import request_human_review

                        human_review_result = request_human_review(
                            story_id=new_state.story_id,
                            review_type=review_type,
                            content=updated_content[:2000] + "...",  # Preview
                            options=[
                                {
                                    "text": "Approve",
                                    "description": f"Approve the {'continuation' if is_continuation else 'edits'}",
                                },
                                {
                                    "text": "Request Revisions",
                                    "description": "Request specific revisions",
                                },
                                {
                                    "text": "Reject",
                                    "description": f"Reject the {'continuation' if is_continuation else 'edits'} entirely",
                                },
                            ],
                            context=f"Supervisor's review:\n\n{response['output']}",
                            deadline=None,
                        )

                        if human_review_result.get("status") == "success":
                            review_id = human_review_result.get("review_id")

                            # Update state to awaiting human input
                            new_state.request_human_input(
                                input_type=review_type,
                                data={
                                    "review_id": review_id,
                                    "content": updated_content,
                                    "supervisor_feedback": response["output"],
                                },
                            )
            else:
                # Edits/continuation need revision
                writer_id = None
                for msg in new_state.messages:
                    if msg.get("metadata", {}).get("task_type") in [
                        "write_continuation_complete",
                        "edit_content_complete",
                    ]:
                        writer_id = msg.get("sender")
                        break

                if writer_id and writer_id in new_state.writing_team:
                    # Create revision task
                    revision_task_id = create_task_id("revision", writer_id)

                    # Format revision task
                    revision_prompt = f"""
                    # {'Continuation' if is_continuation else 'Edit'} Revision Request
                    
                    Your {'continuation' if is_continuation else 'edits'} has been reviewed and requires some revisions. Here is the feedback:
                    
                    {response["output"]}
                    
                    Please address these points and revise accordingly.
                    """

                    # Add revision task to state
                    new_state.add_task(
                        task_id=revision_task_id,
                        agent_id=writer_id,
                        task_type=f"revise_{task_type.replace('review_', '')}",
                        description=f"Revise {'continuation' if is_continuation else 'edits'} based on supervisor feedback",
                        data={"feedback": response["output"]},
                    )

                    # Update writer state
                    writer = new_state.writing_team[writer_id]
                    writer.update_status("working", f"revising_{task_type.replace('review_', '')}")

                    # Add message to communication log
                    new_state.add_message(
                        sender=supervisor_id,
                        recipient=writer_id,
                        content=revision_prompt,
                        metadata={
                            "task_id": revision_task_id,
                            "task_type": f"revise_{task_type.replace('review_', '')}",
                        },
                    )
    except Exception as e:
        error_msg = f"Error during {task_type}: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        supervisor.update_status("error", f"{task_type}_error")

    return new_state


def process_human_edit_approval(state: GraphState, human_input: Dict[str, Any]) -> GraphState:
    """Process human input on edit/continuation approval."""
    new_state = state.copy()

    # This node handles the EDIT or CONTINUE operation modes
    if new_state.operation_mode not in [OperationMode.EDIT, OperationMode.CONTINUE]:
        return new_state

    # Extract data from human input
    decision = human_input.get("decision", "")
    comments = human_input.get("comments", "")
    review_id = human_input.get("review_id", "")

    # Update human review in database
    if review_id:
        from agents.tools import check_human_review_status

        check_human_review_status(review_id)

    # Process based on decision
    if decision.lower() == "approve":
        # Human approved the edits/continuation
        if new_state.operation_mode == OperationMode.CONTINUE:
            new_state.update_story_state(StoryState.EDITING)
        else:
            new_state.update_story_state(StoryState.READY_FOR_PUBLISHING)

        # Log the approval
        new_state.add_message(
            sender="human",
            recipient="system",
            content=f"Human approved the {'continuation' if new_state.operation_mode == OperationMode.CONTINUE else 'edits'}.\nComments: {comments}",
            metadata={"review_id": review_id, "decision": "approve"},
        )

        # For edit mode, we might want to move to publishing
        if new_state.operation_mode == OperationMode.EDIT:
            start_publishing_phase(new_state)

    elif decision.lower() == "request revisions":
        # Human requested revisions
        writer_id = None
        for msg in new_state.messages:
            if msg.get("metadata", {}).get("task_type") in [
                "write_continuation_complete",
                "edit_content_complete",
            ]:
                writer_id = msg.get("sender")
                break

        if writer_id and writer_id in new_state.writing_team:
            # Create revision task
            revision_task_id = create_task_id("human_revision", writer_id)

            is_continuation = new_state.operation_mode == OperationMode.CONTINUE

            # Format revision task
            revision_prompt = f"""
            # Human-Requested {'Continuation' if is_continuation else 'Edit'} Revisions
            
            The client/reviewer has requested revisions to your {'continuation' if is_continuation else 'edits'}. Here is their feedback:
            
            {comments}
            
            Please make these revisions carefully, focusing on addressing all of the client's concerns.
            This is a high-priority revision as it comes directly from the client.
            """

            # Add revision task to state
            new_state.add_task(
                task_id=revision_task_id,
                agent_id=writer_id,
                task_type="human_requested_revision",
                description=f"Make revisions to {'continuation' if is_continuation else 'edits'} based on human feedback",
                data={"feedback": comments},
            )

            # Update writer state
            writer = new_state.writing_team[writer_id]
            writer.update_status("working", "human_revisions")

            # Add message to communication log
            new_state.add_message(
                sender="human",
                recipient=writer_id,
                content=revision_prompt,
                metadata={"task_id": revision_task_id, "task_type": "human_requested_revision"},
            )

            # Log feedback
            from agents.tools import add_feedback

            add_feedback(
                story_id=new_state.story_id,
                feedback_type=FeedbackType.CONTENT.value,
                content=comments,
                source="human",
                source_role="client",
                severity=4,
            )

    elif decision.lower() == "reject":
        # Human rejected the edits/continuation
        writing_supervisor_id = next(
            sup_id
            for sup_id, sup in new_state.supervisors.items()
            if sup.agent_role == AgentRole.WRITING_SUPERVISOR
        )

        is_continuation = new_state.operation_mode == OperationMode.CONTINUE

        # Create a task to handle major revision
        task_id = create_task_id("major_revision", writing_supervisor_id)

        # Format task
        task_prompt = f"""
        # {'Continuation' if is_continuation else 'Edit'} Rejected
        
        The client/reviewer has rejected the {'continuation' if is_continuation else 'edits'}. Here is their feedback:
        
        {comments}
        
        This requires a completely new approach. Please develop a new plan for {'continuing' if is_continuation else 'editing'} the story that addresses these concerns.
        """

        # Add task to state
        new_state.add_task(
            task_id=task_id,
            agent_id=writing_supervisor_id,
            task_type=f"new_{'continuation' if is_continuation else 'edit'}_plan",
            description=f"Plan new {'continuation' if is_continuation else 'edit'} approach based on rejection",
            data={"feedback": comments},
        )

        # Update supervisor state
        supervisor = new_state.supervisors[writing_supervisor_id]
        supervisor.update_status(
            "working", f"planning_new_{'continuation' if is_continuation else 'edit'}"
        )

        # Add message to communication log
        new_state.add_message(
            sender="human",
            recipient=writing_supervisor_id,
            content=task_prompt,
            metadata={
                "task_id": task_id,
                "task_type": f"new_{'continuation' if is_continuation else 'edit'}_plan",
            },
        )

        # Log feedback
        from agents.tools import add_feedback

        add_feedback(
            story_id=new_state.story_id,
            feedback_type=FeedbackType.CONTENT.value,
            content=comments,
            source="human",
            source_role="client",
            severity=5,
        )

    return new_state


def conduct_briefing(state: GraphState) -> GraphState:
    """Conduct initial briefing with the client using the author relations agent."""
    new_state = state.copy()

    # Get author relations agent and their current task
    author_relations_id = next(iter(new_state.author_relations.keys()))
    author_agent = new_state.author_relations[author_relations_id]

    # Find the briefing task message
    briefing_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == author_relations_id
        and msg.get("metadata", {}).get("task_type") == "briefing"
    ]

    if not briefing_messages:
        # No briefing task found
        return new_state

    latest_briefing_message = briefing_messages[-1]
    task_id = latest_briefing_message.get("metadata", {}).get("task_id")

    # Create the agent
    author_relations_agent = create_author_relations_agent(author_relations_id)

    try:
        # Prepare brainstorming session prompt
        user_request = new_state.user_request
        story_id = new_state.story_id

        # Include story structure information in the prompt
        structure_info = ""
        if new_state.story_structure and new_state.structure_template:
            structure_name = new_state.structure_template.get(
                "name", new_state.story_structure.value
            )
            structure_description = new_state.structure_template.get("description", "")
            structure_info = (
                f"\nStory Structure: {structure_name}\nDescription: {structure_description}\n"
            )

        prompt = BRAINSTORM_SESSION_PROMPT.format(
            topic="Initial Story Briefing",
            story_request=(
                user_request.to_prompt_string()
                if user_request
                else "No specific request details provided"
            ),
            current_status=f"Initial briefing phase\n{structure_info}",
            key_questions=(
                "1. What are the most important elements of this story to the client?\n"
                "2. Are there any specific research areas that would benefit the story?\n"
                "3. What tone and style is the client looking for?\n"
                "4. Any specific characters or plot elements that must be included?\n"
                "5. What would make this story particularly successful for the client?"
            ),
            previous_ideas="",
        )

        # Execute briefing task
        response = author_relations_agent.invoke(
            {"input": prompt, "agent_id": author_relations_id, "story_id": story_id}
        )

        if "output" in response:
            # Create a brainstorming session
            session_id = generate_id("session")

            # Update author relations agent state
            author_agent.start_session(session_id, "Initial Briefing")
            author_agent.add_session_message(session_id, "agent", response["output"])
            author_agent.update_status("awaiting_feedback", "briefing_session_started")

            # Add message to communication log
            new_state.add_message(
                sender=author_relations_id,
                recipient="human",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "briefing", "session_id": session_id},
            )

            # Request human input
            new_state.request_human_input(
                input_type="briefing_session",
                data={
                    "session_id": session_id,
                    "agent_id": author_relations_id,
                    "current_message": response["output"],
                    "story_id": story_id,
                },
            )

            # Update task status
            new_state.update_task_status(task_id=task_id, status="awaiting_human_input")
    except Exception as e:
        error_msg = f"Error during briefing: {str(e)}"
        new_state.add_error("author_relations", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        author_agent.update_status("error", "briefing_error")

    return new_state


def process_human_briefing_input(state: GraphState, human_input: Dict[str, Any]) -> GraphState:
    """Process human input during the briefing phase."""
    new_state = state.copy()

    # Extract data from human input
    session_id = human_input.get("session_id")
    agent_id = human_input.get("agent_id")
    message = human_input.get("message", "")
    continue_session = human_input.get("continue", True)

    if not session_id or not agent_id:
        new_state.add_error("system", "Missing session ID or agent ID in human input")
        return new_state

    # Update author relations agent state
    if agent_id in new_state.author_relations:
        author_agent = new_state.author_relations[agent_id]
        author_agent.add_session_message(session_id, "human", message)

        if continue_session:
            author_agent.update_status("working", "continuing_briefing")

            # Create the agent
            author_relations_agent = create_author_relations_agent(agent_id)

            try:
                # Add message to communication log
                new_state.add_message(
                    sender="human",
                    recipient=agent_id,
                    content=message,
                    metadata={"session_id": session_id},
                )

                # Get the session history for context
                session_messages = [
                    msg
                    for msg in author_agent.session_history.get(session_id, [])
                    if msg.get("role") in ["agent", "human"]
                ]

                formatted_history = "\n\n".join(
                    [f"{msg['role'].upper()}: {msg['content']}" for msg in session_messages]
                )

                # Include story structure information in the prompt
                structure_info = ""
                if new_state.story_structure and new_state.structure_template:
                    structure_name = new_state.structure_template.get(
                        "name", new_state.story_structure.value
                    )
                    structure_description = new_state.structure_template.get("description", "")
                    structure_info = f"\nStory Structure: {structure_name}\nDescription: {structure_description}\n"

                prompt = f"""
                # Continuing Briefing Session
                
                ## Story Structure
                {structure_info}
                
                ## Session History
                {formatted_history}
                
                Continue the briefing session based on the client's response above. 
                Focus on gathering all necessary information to create a detailed story plan
                that aligns with the selected story structure.
                """

                # Process the human input with the agent
                response = author_relations_agent.invoke(
                    {
                        "input": prompt,
                        "agent_id": agent_id,
                        "session_id": session_id,
                        "story_id": new_state.story_id,
                    }
                )

                if "output" in response:
                    # Record agent response
                    author_agent.add_session_message(session_id, "agent", response["output"])

                    # Add message to communication log
                    new_state.add_message(
                        sender=agent_id,
                        recipient="human",
                        content=response["output"],
                        metadata={"session_id": session_id},
                    )

                    # Request further human input
                    new_state.request_human_input(
                        input_type="briefing_session",
                        data={
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "current_message": response["output"],
                            "story_id": new_state.story_id,
                        },
                    )

                    author_agent.update_status("awaiting_feedback", "awaiting_briefing_response")
            except Exception as e:
                error_msg = f"Error processing human input: {str(e)}"
                new_state.add_error(agent_id, error_msg)
                author_agent.update_status("error", "briefing_response_error")
        else:
            # End the briefing session
            author_agent.end_session(session_id)
            author_agent.update_status("completed", "briefing_completed")

            # Add message to communication log
            new_state.add_message(
                sender="human",
                recipient=agent_id,
                content=f"{message}\n\n[BRIEFING SESSION ENDED]",
                metadata={"session_id": session_id, "session_ended": True},
            )

            # Create a summary of the briefing
            try:
                author_relations_agent = create_author_relations_agent(agent_id)

                # Get all session messages
                session_messages = author_agent.session_history.get(session_id, [])
                formatted_history = "\n\n".join(
                    [f"{msg['role'].upper()}: {msg['content']}" for msg in session_messages]
                )

                # Include story structure in the prompt
                structure_info = ""
                if new_state.story_structure and new_state.structure_template:
                    structure_name = new_state.structure_template.get(
                        "name", new_state.story_structure.value
                    )
                    acts_info = ""
                    if "acts" in new_state.structure_template:
                        for act in new_state.structure_template["acts"]:
                            acts_info += f"- {act.get('name', '')}: {act.get('description', '')}\n"

                    structure_info = f"""
                    # Story Structure: {structure_name}
                    
                    {new_state.structure_template.get("description", "")}
                    
                    ## Acts
                    {acts_info}
                    """

                summary_prompt = f"""
                # Briefing Session Summary
                
                ## Story Structure
                {structure_info}
                
                ## Complete Briefing Conversation
                {formatted_history}
                
                Please create a structured summary of this briefing session that:
                1. Identifies the core requirements for the story
                2. Lists key elements that must be included
                3. Notes specific style, tone, and audience preferences
                4. Suggests priority research areas
                5. Captures any special requests or constraints
                6. Maps how these elements align with the selected story structure
                
                This summary will be used to guide the story creation process.
                """

                summary_response = author_relations_agent.invoke(
                    {"input": summary_prompt, "agent_id": agent_id, "story_id": new_state.story_id}
                )

                if "output" in summary_response:
                    summary = summary_response["output"]

                    # Create a bible section with the briefing summary
                    from agents.tools import create_bible_section

                    result = create_bible_section(
                        story_id=new_state.story_id,
                        section_type=BibleSectionType.REFERENCE_MATERIAL.value,
                        title="Briefing Summary",
                        content=summary,
                        agent_id=agent_id,
                        tags=["briefing", "requirements"],
                    )

                    # Update state to move to research phase
                    new_state.update_story_state(StoryState.RESEARCH)

                    # Create a research task
                    start_research_phase(new_state)
            except Exception as e:
                error_msg = f"Error creating briefing summary: {str(e)}"
                new_state.add_error(agent_id, error_msg)
                # Still move to research phase despite error
                new_state.update_story_state(StoryState.RESEARCH)
                start_research_phase(new_state)

    return new_state


def start_research_phase(state: GraphState) -> GraphState:
    """Start the research phase by assigning tasks to researchers."""
    new_state = state.copy()

    # Update story state
    new_state.update_story_state(StoryState.RESEARCH)

    # Get researcher and research supervisor
    researcher_id = next(iter(new_state.research_team.keys()))
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR
    )

    # Get briefing summary from bible if available
    bible_sections = new_state.bible.sections if new_state.bible else {}
    briefing_summary = ""

    if BibleSectionType.REFERENCE_MATERIAL.value in bible_sections:
        for section in bible_sections[BibleSectionType.REFERENCE_MATERIAL.value]:
            if section["title"] == "Briefing Summary":
                briefing_summary = section["content"]
                break

    # Include story structure information
    structure_info = ""
    if new_state.story_structure and new_state.structure_template:
        structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
        structure_info = f"\n\nStory Structure: {structure_name}\n{new_state.structure_template.get('description', '')}"

        # Add act information
        if "acts" in new_state.structure_template:
            structure_info += "\n\nActs:"
            for act in new_state.structure_template["acts"]:
                structure_info += f"\n- {act.get('name', '')}: {act.get('description', '')}"

    # Create a research task
    task_id = create_task_id("research", researcher_id)

    # Format the task using the research task prompt template
    task_prompt = RESEARCH_TASK_PROMPT.format(
        task_description=f"Conduct comprehensive research for the story based on the client's requirements, with special attention to elements needed for a {new_state.story_structure.value} structure.",
        story_request=(
            new_state.user_request.to_prompt_string()
            if new_state.user_request
            else "No specific request details provided"
        ),
        research_focus=(
            "1. Background information relevant to the story's setting and time period\n"
            "2. Subject matter expertise needed for authentic details\n"
            "3. Cultural, historical, or scientific context\n"
            "4. Similar works for inspiration and differentiation\n"
            "5. Target audience preferences and expectations\n"
            f"6. Research specific to each act in the {new_state.story_structure.value} structure"
        ),
        bible_entries=f"{briefing_summary}{structure_info}" or "No bible entries available yet.",
        existing_research="No prior research exists for this story.",
    )

    # Add task to state
    new_state.add_task(
        task_id=task_id,
        agent_id=researcher_id,
        task_type="initial_research",
        description="Conduct initial comprehensive research for the story",
        data={
            "user_request": new_state.user_request.dict() if new_state.user_request else {},
            "briefing_summary": briefing_summary,
            "story_structure": new_state.story_structure.value,
        },
    )

    # Update researcher state
    researcher = new_state.research_team[researcher_id]
    researcher.update_status("working", "conducting_research")

    # Add message to communication log
    new_state.add_message(
        sender="system",
        recipient=researcher_id,
        content=task_prompt,
        metadata={"task_id": task_id, "task_type": "initial_research"},
    )

    return new_state


def conduct_research(state: GraphState) -> GraphState:
    """Conduct research using the researcher agent."""
    new_state = state.copy()

    # Get researcher and their task
    researcher_id = next(iter(new_state.research_team.keys()))
    researcher = new_state.research_team[researcher_id]

    # Find the research task message
    research_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == researcher_id
        and msg.get("metadata", {}).get("task_type") == "initial_research"
    ]

    if not research_messages:
        # No research task found
        return new_state

    latest_research_message = research_messages[-1]
    task_id = latest_research_message.get("metadata", {}).get("task_id")

    # Create the researcher agent
    researcher_agent = create_researcher_agent(researcher_id)

    try:
        # Execute research task
        response = researcher_agent.invoke(
            {
                "input": latest_research_message["content"],
                "agent_id": researcher_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update researcher state
            researcher.update_status("completed", "research_completed")

            # Add message to communication log
            new_state.add_message(
                sender=researcher_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "research_results"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"research_output": response["output"]}
            )

            # Send research to supervisor for review
            supervisor_id = next(
                sup_id
                for sup_id, sup in new_state.supervisors.items()
                if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR
            )

            # Create review task
            review_task_id = create_task_id("review", supervisor_id)

            # Include story structure context
            structure_context = ""
            if new_state.story_structure and new_state.structure_template:
                structure_context = f"\n\nResearch should support a {new_state.structure_template.get('name', '')} structure."

            # Format review task
            review_prompt = REVIEW_TASK_PROMPT.format(
                task_description=f"Review the initial research conducted for the story with the {new_state.story_structure.value} structure.",
                content_type="Research",
                content_summary=clean_and_format_text(response["output"]),
                review_criteria=(
                    "1. Comprehensiveness - Does the research cover all necessary areas?\n"
                    "2. Relevance - Is the research directly applicable to the story requirements?\n"
                    "3. Accuracy - Does the research appear accurate and from reliable sources?\n"
                    "4. Usability - Is the research organized in a way that writers can easily use?\n"
                    "5. Structure Support - Does the research adequately support each act in the story structure?\n"
                    "6. Gaps - Are there any obvious gaps or missing areas of research?"
                ),
                context=f"{new_state.user_request.to_prompt_string() if new_state.user_request else 'No specific request details provided'}{structure_context}",
                previous_feedback="",
            )

            # Add review task to state
            new_state.add_task(
                task_id=review_task_id,
                agent_id=supervisor_id,
                task_type="research_review",
                description="Review initial research for completeness and quality",
                data={
                    "research_output": response["output"],
                    "researcher_id": researcher_id,
                    "story_structure": new_state.story_structure.value,
                },
            )

            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", "reviewing_research")
            supervisor.pending_reviews.append(task_id)

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=review_prompt,
                metadata={"task_id": review_task_id, "task_type": "research_review"},
            )
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        new_state.add_error("researcher", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        researcher.update_status("error", "research_error")

        # Retry logic could be implemented here
        retry_count = new_state.increment_retry(f"research_{researcher_id}")
        if retry_count <= 3:
            # Reset researcher for retry
            researcher.update_status("idle", "ready_for_retry")
            # Could add a new task with modified instructions based on error

    return new_state


def review_research(state: GraphState) -> GraphState:
    """Review research with the research supervisor agent."""
    new_state = state.copy()

    # Get supervisor and their task
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR
    )
    supervisor = new_state.supervisors[supervisor_id]

    # Find the review task message
    review_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == supervisor_id
        and msg.get("metadata", {}).get("task_type") == "research_review"
    ]

    if not review_messages:
        # No review task found
        return new_state

    latest_review_message = review_messages[-1]
    task_id = latest_review_message.get("metadata", {}).get("task_id")

    # Create the supervisor agent
    supervisor_agent = create_research_supervisor_agent(supervisor_id)

    try:
        # Execute review task
        response = supervisor_agent.invoke(
            {
                "input": latest_review_message["content"],
                "agent_id": supervisor_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "review_completed")

            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "research_review_results"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"review_output": response["output"]}
            )

            # Determine if research is approved or needs revision
            is_approved = (
                "approved" in response["output"].lower()
                or "satisfactory" in response["output"].lower()
            )

            if is_approved:
                # Research is approved, move to planning phase
                complete_research_phase(new_state, response["output"])
            else:
                # Research needs revision
                researcher_id = next(iter(new_state.research_team.keys()))

                # Create revision task
                revision_task_id = create_task_id("revision", researcher_id)

                # Format revision task
                revision_prompt = f"""
                # Research Revision Request
                
                Your initial research has been reviewed and requires some revisions. Here is the feedback:
                
                {response["output"]}
                
                Please address these points and revise your research accordingly. Focus particularly on any gaps or areas that need more depth.
                Remember to consider how the research supports each act in the {new_state.story_structure.value} structure.
                """

                # Add revision task to state
                new_state.add_task(
                    task_id=revision_task_id,
                    agent_id=researcher_id,
                    task_type="research_revision",
                    description="Revise research based on supervisor feedback",
                    data={
                        "feedback": response["output"],
                        "story_structure": new_state.story_structure.value,
                    },
                )

                # Update researcher state
                researcher = new_state.research_team[researcher_id]
                researcher.update_status("working", "revising_research")

                # Add message to communication log
                new_state.add_message(
                    sender=supervisor_id,
                    recipient=researcher_id,
                    content=revision_prompt,
                    metadata={"task_id": revision_task_id, "task_type": "research_revision"},
                )

                # Log feedback
                from agents.tools import add_feedback

                add_feedback(
                    story_id=new_state.story_id,
                    feedback_type=FeedbackType.RESEARCH.value,
                    content=response["output"],
                    source=supervisor_id,
                    source_role=AgentRole.RESEARCH_SUPERVISOR.value,
                    severity=3,
                )
    except Exception as e:
        error_msg = f"Error during research review: {str(e)}"
        new_state.add_error("research_supervisor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        supervisor.update_status("error", "review_error")

    return new_state


def complete_research_phase(state: GraphState, review_output: str) -> GraphState:
    """Complete the research phase and move to planning."""
    new_state = state.copy()

    # Mark research phase as complete
    new_state.mark_phase_complete(StoryState.RESEARCH)

    # Create bible sections from research
    style_guide_id = next(iter(new_state.style_guide_editor.keys()))

    # Create bible update task
    task_id = create_task_id("bible_update", style_guide_id)

    # Format bible update task, including story structure references
    structure_context = ""
    if new_state.story_structure and new_state.structure_template:
        structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
        structure_context = (
            f"\n\nThis research should be organized to support the {structure_name} structure."
        )

    update_prompt = BIBLE_UPDATE_PROMPT.format(
        task_description=f"Create reference material sections in the story bible based on approved research for a {new_state.story_structure.value} structure story.",
        section_type=BibleSectionType.REFERENCE_MATERIAL.value,
        section_title="Research Findings",
        current_content="No existing content for this section.",
        new_information=f"{review_output}{structure_context}",
        related_sections="",
    )

    # Add task to state
    new_state.add_task(
        task_id=task_id,
        agent_id=style_guide_id,
        task_type="bible_update",
        description="Create bible sections from research",
        data={"research_output": review_output, "story_structure": new_state.story_structure.value},
    )

    # Update style guide editor state
    style_guide = new_state.style_guide_editor[style_guide_id]
    style_guide.update_status("working", "updating_bible")

    # Add message to communication log
    new_state.add_message(
        sender="system",
        recipient=style_guide_id,
        content=update_prompt,
        metadata={"task_id": task_id, "task_type": "bible_update"},
    )

    # Update story state to planning
    new_state.update_story_state(StoryState.PLANNING)

    return new_state


def update_story_bible(state: GraphState) -> GraphState:
    """Update the story bible with the style guide editor agent."""
    new_state = state.copy()

    # Get style guide editor and their task
    style_guide_id = next(iter(new_state.style_guide_editor.keys()))
    style_guide = new_state.style_guide_editor[style_guide_id]

    # Find the bible update task message
    update_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == style_guide_id
        and msg.get("metadata", {}).get("task_type") == "bible_update"
    ]

    if not update_messages:
        # No update task found
        return new_state

    latest_update_message = update_messages[-1]
    task_id = latest_update_message.get("metadata", {}).get("task_id")

    # Create the style guide editor agent
    style_guide_agent = create_style_guide_editor_agent(style_guide_id)

    try:
        # Execute bible update task
        response = style_guide_agent.invoke(
            {
                "input": latest_update_message["content"],
                "agent_id": style_guide_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update style guide editor state
            style_guide.update_status("completed", "bible_updated")

            # Add message to communication log
            new_state.add_message(
                sender=style_guide_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "bible_update_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"bible_update_output": response["output"]},
            )

            # Start the planning phase (outline creation)
            start_planning_phase(new_state)
    except Exception as e:
        error_msg = f"Error updating story bible: {str(e)}"
        new_state.add_error("style_guide_editor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        style_guide.update_status("error", "bible_update_error")

        # Continue to planning phase despite error
        start_planning_phase(new_state)

    return new_state


def start_planning_phase(state: GraphState) -> GraphState:
    """Start the planning phase by assigning outline creation task."""
    new_state = state.copy()

    # Update story state
    new_state.update_story_state(StoryState.PLANNING)

    # Get writer and writing supervisor
    writer_id = next(iter(new_state.writing_team.keys()))
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.WRITING_SUPERVISOR
    )

    # Get bible sections for reference
    from agents.tools import get_bible_sections

    bible_result = get_bible_sections(new_state.story_id)

    bible_sections_text = ""
    if bible_result.get("status") != "error" and "sections" in bible_result:
        sections = bible_result["sections"]
        for section_type, section_list in sections.items():
            for section in section_list:
                bible_sections_text += (
                    f"\n## {section['title']} ({section_type})\n{section['content']}\n"
                )

    # Get research items
    from agents.tools import get_research_for_story

    research_items = get_research_for_story(new_state.story_id)

    research_text = ""
    if isinstance(research_items, list) and not any("error" in item for item in research_items):
        for i, item in enumerate(research_items, 1):
            research_text += f"\n### Research Item {i}\nSource: {item.get('source', 'Unknown')}\n{item.get('content', '')}\n"

    # Add story structure details to the prompt
    structure_details = ""
    if new_state.story_structure and new_state.structure_template:
        structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
        structure_description = new_state.structure_template.get("description", "")

        structure_details = f"""
        # Story Structure: {structure_name}
        
        {structure_description}
        
        ## Acts and Components
        """

        for act in new_state.structure_template.get("acts", []):
            act_name = act.get("name", "")
            act_desc = act.get("description", "")
            structure_details += f"\n### {act_name}\n{act_desc}\n"

            components = act.get("components", [])
            if components:
                structure_details += "\nComponents:\n"
                for comp in components:
                    comp_name = comp.get("name", "")
                    comp_desc = comp.get("description", "")
                    structure_details += f"- {comp_name}: {comp_desc}\n"

    # Create outline task
    task_id = create_task_id("outline", writer_id)

    # Format the task using the writing task prompt template
    task_prompt = WRITING_TASK_PROMPT.format(
        task_description=f"Create a detailed story outline based on the {new_state.story_structure.value} structure, the research, and client requirements.",
        story_request=(
            new_state.user_request.to_prompt_string()
            if new_state.user_request
            else "No specific request details provided"
        ),
        section="outline",
        target_length="N/A for outline",
        tone_style=new_state.user_request.style if new_state.user_request else "Not specified",
        reference_materials=research_text or "No specific research materials available.",
        bible_entries=bible_sections_text or "No bible entries available yet.",
        outline_elements=(
            f"Your outline should follow the {new_state.story_structure.value} structure as detailed below:\n\n"
            f"{structure_details}\n\n"
            "Include the following elements in your outline:\n"
            "1. A compelling title that captures the essence of the story\n"
            "2. A brief summary of the overall narrative\n"
            "3. Character profiles with motivation and arcs\n"
            "4. Major plot points organized according to the structure's acts and components\n"
            "5. Key settings with descriptions\n"
            "6. Central themes and motifs\n"
            "7. How each act will be developed"
        ),
        previous_feedback="",
    )

    # Add task to state
    new_state.add_task(
        task_id=task_id,
        agent_id=writer_id,
        task_type="create_outline",
        description="Create detailed story outline",
        data={
            "user_request": new_state.user_request.dict() if new_state.user_request else {},
            "bible_sections": bible_sections_text,
            "research": research_text,
            "story_structure": new_state.story_structure.value,
            "structure_template": new_state.structure_template,
        },
    )

    # Update writer state
    writer = new_state.writing_team[writer_id]
    writer.update_status("working", "creating_outline")
    writer.current_section = "outline"
    writer.assigned_sections.append("outline")

    # Add message to communication log
    new_state.add_message(
        sender="system",
        recipient=writer_id,
        content=task_prompt,
        metadata={"task_id": task_id, "task_type": "create_outline"},
    )

    return new_state


def create_story_outline(state: GraphState) -> GraphState:
    """Create a story outline with the writer agent."""
    new_state = state.copy()

    # Get writer and their task
    writer_id = next(
        w_id for w_id, w in new_state.writing_team.items() if w.agent_role == AgentRole.WRITER
    )
    writer = new_state.writing_team[writer_id]

    # Find the outline task message
    outline_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == writer_id
        and msg.get("metadata", {}).get("task_type") == "create_outline"
    ]

    if not outline_messages:
        # No outline task found
        return new_state

    latest_outline_message = outline_messages[-1]
    task_id = latest_outline_message.get("metadata", {}).get("task_id")

    # Create the writer agent
    writer_agent = create_writer_agent(writer_id)

    try:
        # Execute outline creation task
        response = writer_agent.invoke(
            {
                "input": latest_outline_message["content"],
                "agent_id": writer_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update writer state
            writer.update_status("completed", "outline_created")
            writer.mark_section_complete("outline")
            writer.add_draft_content("outline", response["output"])

            # Add message to communication log
            new_state.add_message(
                sender=writer_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "outline_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"outline": response["output"]}
            )

            # Send outline to supervisor for review
            supervisor_id = next(
                sup_id
                for sup_id, sup in new_state.supervisors.items()
                if sup.agent_role == AgentRole.WRITING_SUPERVISOR
            )

            # Create review task
            review_task_id = create_task_id("review", supervisor_id)

            # Format review task with structure reference
            structure_reference = f"adherence to the {new_state.story_structure.value} structure"

            review_prompt = REVIEW_TASK_PROMPT.format(
                task_description=f"Review the story outline for quality, {structure_reference}, and alignment with requirements.",
                content_type="Story Outline",
                content_summary=clean_and_format_text(response["output"]),
                review_criteria=(
                    f"1. {structure_reference.capitalize()}\n"
                    "2. Adherence to client requirements\n"
                    "3. Character development and dimension\n"
                    "4. Plot coherence and engagement\n"
                    "5. Setting detail and integration\n"
                    "6. Theme depth and resonance\n"
                    "7. Overall creative quality and potential"
                ),
                context=(
                    new_state.user_request.to_prompt_string()
                    if new_state.user_request
                    else "No specific request details provided"
                ),
                previous_feedback="",
            )

            # Add review task to state
            new_state.add_task(
                task_id=review_task_id,
                agent_id=supervisor_id,
                task_type="outline_review",
                description="Review story outline",
                data={
                    "outline": response["output"],
                    "writer_id": writer_id,
                    "story_structure": new_state.story_structure.value,
                },
            )

            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", "reviewing_outline")
            supervisor.pending_reviews.append(task_id)

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=review_prompt,
                metadata={"task_id": review_task_id, "task_type": "outline_review"},
            )
    except Exception as e:
        error_msg = f"Error creating outline: {str(e)}"
        new_state.add_error("writer", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        writer.update_status("error", "outline_error")

    return new_state


def review_outline(state: GraphState) -> GraphState:
    """Review story outline with the writing supervisor agent."""
    new_state = state.copy()

    # Get supervisor and their task
    supervisor_id = next(
        sup_id
        for sup_id, sup in new_state.supervisors.items()
        if sup.agent_role == AgentRole.WRITING_SUPERVISOR
    )
    supervisor = new_state.supervisors[supervisor_id]

    # Find the review task message
    review_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == supervisor_id
        and msg.get("metadata", {}).get("task_type") == "outline_review"
    ]

    if not review_messages:
        # No review task found
        return new_state

    latest_review_message = review_messages[-1]
    task_id = latest_review_message.get("metadata", {}).get("task_id")

    # Create the supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)

    try:
        # Execute review task
        response = supervisor_agent.invoke(
            {
                "input": latest_review_message["content"],
                "agent_id": supervisor_id,
                "story_id": new_state.story_id,
            }
        )

        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "review_completed")

            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "outline_review_complete"},
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id, status="completed", result={"review_output": response["output"]}
            )

            # Should we request human feedback on the outline?
            request_human_feedback = True

            if request_human_feedback:
                # Request human review of outline
                human_id = next(iter(new_state.human_in_loop.keys()))

                # Get the outline
                outline_messages = [
                    msg
                    for msg in new_state.messages
                    if msg.get("metadata", {}).get("task_type") == "outline_complete"
                ]

                if outline_messages:
                    outline = outline_messages[-1]["content"]

                    # Create human review request
                    from agents.tools import request_human_review

                    human_review_result = request_human_review(
                        story_id=new_state.story_id,
                        review_type="outline_approval",
                        content=outline,
                        options=[
                            {
                                "text": "Approve",
                                "description": "Approve the outline and continue to writing phase",
                            },
                            {
                                "text": "Request Revisions",
                                "description": "Request specific revisions to the outline",
                            },
                            {
                                "text": "Reject",
                                "description": "Reject the outline and restart the planning phase",
                            },
                        ],
                        context=f"Supervisor's review:\n\n{response['output']}\n\nStory Structure: {new_state.story_structure.value}",
                        deadline=None,
                    )

                    if human_review_result.get("status") == "success":
                        review_id = human_review_result.get("review_id")

                        # Update state to awaiting human input
                        new_state.request_human_input(
                            input_type="outline_review",
                            data={
                                "review_id": review_id,
                                "outline": outline,
                                "supervisor_feedback": response["output"],
                            },
                        )

                        return new_state

            # Determine if outline is approved or needs revision
            is_approved = (
                "approved" in response["output"].lower() or "accept" in response["output"].lower()
            )

            if is_approved:
                # Outline is approved, move to writing phase
                complete_planning_phase(new_state, response["output"])
            else:
                # Outline needs revision
                writer_id = next(
                    w_id
                    for w_id, w in new_state.writing_team.items()
                    if w.agent_role == AgentRole.WRITER
                )

                # Create revision task
                revision_task_id = create_task_id("revision", writer_id)

                # Format revision task
                revision_prompt = f"""
                # Outline Revision Request
                
                Your story outline has been reviewed and requires some revisions. Here is the feedback:
                
                {response["output"]}
                
                Please address these points and revise your outline accordingly. Ensure it follows the {new_state.story_structure.value} structure properly.
                Maintain the same format but improve the content based on the feedback.
                """

                # Add revision task to state
                new_state.add_task(
                    task_id=revision_task_id,
                    agent_id=writer_id,
                    task_type="outline_revision",
                    description="Revise outline based on supervisor feedback",
                    data={
                        "feedback": response["output"],
                        "story_structure": new_state.story_structure.value,
                    },
                )

                # Update writer state
                writer = new_state.writing_team[writer_id]
                writer.update_status("working", "revising_outline")
                writer.increment_revision("outline")

                # Add message to communication log
                new_state.add_message(
                    sender=supervisor_id,
                    recipient=writer_id,
                    content=revision_prompt,
                    metadata={"task_id": revision_task_id, "task_type": "outline_revision"},
                )

                # Log feedback
                from agents.tools import add_feedback

                add_feedback(
                    story_id=new_state.story_id,
                    feedback_type=FeedbackType.STRUCTURE.value,
                    content=response["output"],
                    source=supervisor_id,
                    source_role=AgentRole.WRITING_SUPERVISOR.value,
                    target_section="outline",
                    severity=3,
                )
    except Exception as e:
        error_msg = f"Error during outline review: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        supervisor.update_status("error", "review_error")

    return new_state


def process_human_outline_review(state: GraphState, human_input: Dict[str, Any]) -> GraphState:
    """Process human input on the outline review."""
    new_state = state.copy()

    # Extract data from human input
    decision = human_input.get("decision", "")
    comments = human_input.get("comments", "")

    # Process based on decision
    if decision.lower() == "approve":
        # Human approved the outline, move to writing phase
        complete_planning_phase(new_state, f"Human approved the outline. {comments}")

    elif decision.lower() == "request revisions":
        # Human requested revisions to the outline
        writer_id = next(
            w_id for w_id, w in new_state.writing_team.items() if w.agent_role == AgentRole.WRITER
        )

        # Create revision task
        revision_task_id = create_task_id("human_revision", writer_id)

        # Format revision task
        revision_prompt = f"""
        # Human-Requested Outline Revisions
        
        The client/reviewer has requested revisions to your outline. Here is their feedback:
        
        {comments}
        
        Please make these revisions carefully, focusing on addressing all of the client's concerns.
        This is a high-priority revision as it comes directly from the client.
        
        Remember to maintain the {new_state.story_structure.value} structure while incorporating these changes.
        """

        # Add revision task to state
        new_state.add_task(
            task_id=revision_task_id,
            agent_id=writer_id,
            task_type="human_outline_revision",
            description="Revise outline based on human feedback",
            data={"feedback": comments, "story_structure": new_state.story_structure.value},
        )

        # Update writer state
        writer = new_state.writing_team[writer_id]
        writer.update_status("working", "human_outline_revision")
        writer.increment_revision("outline")

        # Add message to communication log
        new_state.add_message(
            sender="human",
            recipient=writer_id,
            content=revision_prompt,
            metadata={"task_id": revision_task_id, "task_type": "human_outline_revision"},
        )

        # Log feedback
        from agents.tools import add_feedback

        add_feedback(
            story_id=new_state.story_id,
            feedback_type=FeedbackType.STRUCTURE.value,
            content=comments,
            source="human",
            source_role="client",
            target_section="outline",
            severity=4,
        )

    elif decision.lower() == "reject":
        # Human rejected the outline, restart planning
        supervisor_id = next(
            sup_id
            for sup_id, sup in new_state.supervisors.items()
            if sup.agent_role == AgentRole.WRITING_SUPERVISOR
        )

        # Create a task for planning a new outline
        task_id = create_task_id("new_outline", supervisor_id)

        # Format task
        task_prompt = f"""
        # Outline Rejected - New Approach Needed
        
        The client/reviewer has rejected the outline. Here is their feedback:
        
        {comments}
        
        We need to start over with a completely new outline approach. 
        
        Please analyze this feedback carefully and develop a new strategy for the outline. 
        Consider if a different approach to the {new_state.story_structure.value} structure might work better,
        or if we need to emphasize different aspects of the story.
        """

        # Add task to state
        new_state.add_task(
            task_id=task_id,
            agent_id=supervisor_id,
            task_type="plan_new_outline",
            description="Plan new outline approach based on rejection",
            data={"feedback": comments, "story_structure": new_state.story_structure.value},
        )

        # Update supervisor state
        supervisor = new_state.supervisors[supervisor_id]
        supervisor.update_status("working", "planning_new_outline")

        # Add message to communication log
        new_state.add_message(
            sender="human",
            recipient=supervisor_id,
            content=task_prompt,
            metadata={"task_id": task_id, "task_type": "plan_new_outline"},
        )

        # Log feedback
        from agents.tools import add_feedback

        add_feedback(
            story_id=new_state.story_id,
            feedback_type=FeedbackType.STRUCTURE.value,
            content=comments,
            source="human",
            source_role="client",
            target_section="outline",
            severity=5,
        )

    return new_state


def complete_planning_phase(state: GraphState, review_output: str) -> GraphState:
    """Complete the planning phase and move to writing."""
    new_state = state.copy()

    # Mark planning phase as complete
    new_state.mark_phase_complete(StoryState.PLANNING)

    # Update story state to writing
    new_state.update_story_state(StoryState.WRITING)

    # Get the outline
    outline_messages = [
        msg
        for msg in new_state.messages
        if msg.get("metadata", {}).get("task_type") == "outline_complete"
    ]

    if not outline_messages:
        # No outline found
        new_state.add_error("system", "No outline found when trying to complete planning phase")
        return new_state

    outline = outline_messages[-1]["content"]

    # Save the outline to the story in database
    try:
        # Extract outline components for structured storage
        # This is a simplified example and would need more robust parsing in production
        title_match = re.search(r"# (?:Title|Story Title):\s*(.+)", outline)
        title = title_match.group(1).strip() if title_match else "Untitled Story"

        summary_match = re.search(r"## (?:Summary|Synopsis)(?:\s*):?\s*([\s\S]+?)(?=##|$)", outline)
        summary = summary_match.group(1).strip() if summary_match else ""

        # Extract characters (simplified)
        characters_section = re.search(r"## Characters(?:\s*):?\s*([\s\S]+?)(?=##|$)", outline)
        characters = []
        if characters_section:
            character_blocks = re.findall(
                r"### (.+?)(?:\s*):?\s*([\s\S]+?)(?=###|$)", characters_section.group(1)
            )
            for name, desc in character_blocks:
                characters.append(
                    Character(
                        name=name.strip(),
                        role="Main character",  # Default
                        description=desc.strip(),
                    )
                )

        # Extract plot points with act structure
        plot_points = []

        # Try to extract act-structured plot points
        acts_data = []

        # Parse the outline based on the story structure
        if new_state.story_structure == StoryStructure.THREE_ACT:
            act_pattern = (
                r"## Act (?:I|II|III|1|2|3)(?:\s*):?\s*(.+?)(?:\s*):?\s*([\s\S]+?)(?=## Act|$)"
            )
            acts = re.findall(act_pattern, outline)

            for i, (name, content) in enumerate(acts):
                act_name = f"Act {i+1}: {name.strip()}" if name.strip() else f"Act {i+1}"

                # Extract plot points within this act
                plot_pattern = r"(?:\d+\.\s*|\*\s*|-\s*|### )(.+?)(?:\s*):?\s*([\s\S]+?)(?=\d+\.\s*|\*\s*|-\s*|###|$)"
                plots = re.findall(plot_pattern, content)

                act_data = {"name": act_name, "description": "", "components": []}

                for j, (plot_name, plot_desc) in enumerate(plots):
                    if plot_name.strip():
                        plot_points.append(
                            PlotPoint(
                                title=plot_name.strip(),
                                description=plot_desc.strip(),
                                sequence=(i * 100) + j,
                                act=act_name,
                            )
                        )

                        act_data["components"].append(
                            {"name": plot_name.strip(), "description": plot_desc.strip()}
                        )

                acts_data.append(act_data)

        elif new_state.story_structure == StoryStructure.FIVE_ACT:
            act_pattern = r"## Act (?:I|II|III|IV|V|1|2|3|4|5)(?:\s*):?\s*(.+?)(?:\s*):?\s*([\s\S]+?)(?=## Act|$)"
            acts = re.findall(act_pattern, outline)

            for i, (name, content) in enumerate(acts):
                act_name = f"Act {i+1}: {name.strip()}" if name.strip() else f"Act {i+1}"

                # Extract plot points within this act
                plot_pattern = r"(?:\d+\.\s*|\*\s*|-\s*|### )(.+?)(?:\s*):?\s*([\s\S]+?)(?=\d+\.\s*|\*\s*|-\s*|###|$)"
                plots = re.findall(plot_pattern, content)

                act_data = {"name": act_name, "description": "", "components": []}

                for j, (plot_name, plot_desc) in enumerate(plots):
                    if plot_name.strip():
                        plot_points.append(
                            PlotPoint(
                                title=plot_name.strip(),
                                description=plot_desc.strip(),
                                sequence=(i * 100) + j,
                                act=act_name,
                            )
                        )

                        act_data["components"].append(
                            {"name": plot_name.strip(), "description": plot_desc.strip()}
                        )

                acts_data.append(act_data)

        elif new_state.story_structure == StoryStructure.HEROS_JOURNEY:
            journey_stages = [
                "The Ordinary World",
                "The Call to Adventure",
                "Refusal of the Call",
                "Meeting the Mentor",
                "Crossing the Threshold",
                "Tests, Allies, and Enemies",
                "Approach to the Inmost Cave",
                "The Ordeal",
                "Reward",
                "The Road Back",
                "Resurrection",
                "Return with the Elixir",
            ]

            # Try to find each stage
            act_data = {
                "name": "Hero's Journey",
                "description": "Joseph Campbell's monomyth structure",
                "components": [],
            }

            for i, stage in enumerate(journey_stages):
                stage_pattern = (
                    f"(?:##|###) {re.escape(stage)}(?:\s*):?\s*([\s\S]+?)(?=(?:##|###)|$)"
                )
                stage_match = re.search(stage_pattern, outline)

                if stage_match:
                    stage_content = stage_match.group(1).strip()

                    plot_points.append(
                        PlotPoint(
                            title=stage, description=stage_content, sequence=i, act="Hero's Journey"
                        )
                    )

                    act_data["components"].append({"name": stage, "description": stage_content})

            acts_data.append(act_data)

        # If no structured acts found, try to extract general plot points
        if not plot_points:
            plot_section = re.search(
                r"## (?:Plot Points|Plot)(?:\s*):?\s*([\s\S]+?)(?=##|$)", outline
            )
            if plot_section:
                plot_blocks = re.findall(
                    r"(?:\d+\.\s*|\*\s*|-\s*|###\s*)([\s\S]+?)(?=\d+\.\s*|\*\s*|-\s*|###\s*|$)",
                    plot_section.group(1),
                )
                for i, plot in enumerate(plot_blocks):
                    if plot.strip():
                        plot_title_content = (
                            plot.split(":", 1) if ":" in plot else (f"Plot Point {i+1}", plot)
                        )
                        if isinstance(plot_title_content, tuple) and len(plot_title_content) == 2:
                            title, content = plot_title_content
                        else:
                            title = f"Plot Point {i+1}"
                            content = plot_title_content

                        plot_points.append(
                            PlotPoint(
                                title=(
                                    title.strip() if isinstance(title, str) else f"Plot Point {i+1}"
                                ),
                                description=(
                                    content.strip() if isinstance(content, str) else plot.strip()
                                ),
                                sequence=i + 1,
                            )
                        )

        # Extract settings (simplified)
        settings_section = re.search(
            r"## (?:Settings|Setting|Locations)(?:\s*):?\s*([\s\S]+?)(?=##|$)", outline
        )
        settings = []
        if settings_section:
            setting_blocks = re.findall(
                r"(?:\d+\.\s*|\*\s*|-\s*|###\s*)([\s\S]+?)(?=\d+\.\s*|\*\s*|-\s*|###\s*|$)",
                settings_section.group(1),
            )
            for i, setting in enumerate(setting_blocks):
                if setting.strip():
                    setting_title_content = (
                        setting.split(":", 1) if ":" in setting else (f"Setting {i+1}", setting)
                    )
                    if isinstance(setting_title_content, tuple) and len(setting_title_content) == 2:
                        title, content = setting_title_content
                    else:
                        title = f"Setting {i+1}"
                        content = setting_title_content

                    settings.append(
                        Setting(
                            name=title.strip() if isinstance(title, str) else f"Setting {i+1}",
                            description=(
                                content.strip() if isinstance(content, str) else setting.strip()
                            ),
                        )
                    )

        # Extract themes (simplified)
        themes_section = re.search(r"## (?:Themes|Theme)(?:\s*):?\s*([\s\S]+?)(?=##|$)", outline)
        themes = []
        if themes_section:
            theme_blocks = re.findall(
                r"(?:\d+\.\s*|\*\s*|-\s*)([\s\S]+?)(?=\d+\.\s*|\*\s*|-\s*|$)",
                themes_section.group(1),
            )
            themes = [theme.strip() for theme in theme_blocks if theme.strip()]

        # Create structured outline
        story_outline = StoryOutline(
            title=title,
            summary=summary,
            structure=new_state.story_structure,
            acts=acts_data,
            characters=characters,
            plot_points=plot_points,
            settings=settings,
            themes=themes,
        )

        # Update story in database
        from agents.tools import update_story

        update_result = update_story(
            new_state.story_id,
            {"title": title, "outline": story_outline.dict(), "state": StoryState.WRITING.value},
        )

        if update_result.get("status") != "success":
            new_state.add_error(
                "system", f"Failed to update story with outline: {update_result.get('message')}"
            )

        # Add outline to story bible
        style_guide_id = next(iter(new_state.style_guide_editor.keys()))

        # Create separate bible sections for different outline components

        # Characters section
        from agents.tools import create_bible_section

        for character in characters:
            create_bible_section(
                story_id=new_state.story_id,
                section_type=BibleSectionType.CHARACTER_PROFILES.value,
                title=character.name,
                content=character.description,
                agent_id=style_guide_id,
                tags=["character", "outline"],
            )

        # Plot section
        plot_content = "\n\n".join(
            [
                f"**{plot.title}** ({plot.act or 'Main Plot'}):\n{plot.description}"
                for plot in plot_points
            ]
        )
        create_bible_section(
            story_id=new_state.story_id,
            section_type=BibleSectionType.PLOT_ELEMENTS.value,
            title="Plot Outline",
            content=plot_content,
            agent_id=style_guide_id,
            tags=["plot", "outline", new_state.story_structure.value],
        )

        # Settings section
        settings_content = "\n\n".join(
            [f"**{setting.name}**:\n{setting.description}" for setting in settings]
        )
        create_bible_section(
            story_id=new_state.story_id,
            section_type=BibleSectionType.WORLD_BUILDING.value,
            title="Settings",
            content=settings_content,
            agent_id=style_guide_id,
            tags=["settings", "outline"],
        )

        # Themes section
        themes_content = "The story explores the following themes:\n\n" + "\n".join(
            [f"- {theme}" for theme in themes]
        )
        create_bible_section(
            story_id=new_state.story_id,
            section_type=BibleSectionType.THEMES.value,
            title="Themes and Motifs",
            content=themes_content,
            agent_id=style_guide_id,
            tags=["themes", "outline"],
        )

        # Story structure section
        structure_content = (
            f"# {new_state.story_structure.value.replace('_', ' ').title()} Structure\n\n"
        )

        for act in acts_data:
            structure_content += f"## {act['name']}\n"
            if act.get("description"):
                structure_content += f"{act['description']}\n\n"

            for component in act.get("components", []):
                structure_content += f"### {component['name']}\n"
                structure_content += f"{component['description']}\n\n"

        create_bible_section(
            story_id=new_state.story_id,
            section_type=BibleSectionType.PLOT_ELEMENTS.value,
            title=f"{new_state.story_structure.value.replace('_', ' ').title()} Structure",
            content=structure_content,
            agent_id=style_guide_id,
            tags=["structure", new_state.story_structure.value],
        )

        # Create story sections based on the structure
        story_sections = []

        if new_state.story_structure == StoryStructure.THREE_ACT:
            # Create sections based on three-act structure
            for act_index, act in enumerate(acts_data):
                act_name = act.get("name", f"Act {act_index + 1}")

                for comp_index, component in enumerate(act.get("components", [])):
                    comp_name = component.get("name", f"Section {act_index}.{comp_index}")
                    section = new_state.create_section_from_template(
                        act_index, comp_index, comp_name
                    )
                    story_sections.append(section)

        elif new_state.story_structure == StoryStructure.FIVE_ACT:
            # Create sections based on five-act structure
            for act_index, act in enumerate(acts_data):
                act_name = act.get("name", f"Act {act_index + 1}")

                for comp_index, component in enumerate(act.get("components", [])):
                    comp_name = component.get("name", f"Section {act_index}.{comp_index}")
                    section = new_state.create_section_from_template(
                        act_index, comp_index, comp_name
                    )
                    story_sections.append(section)

        elif new_state.story_structure == StoryStructure.HEROS_JOURNEY:
            # Create sections based on hero's journey
            act_index = 0
            for comp_index, component in enumerate(acts_data[0].get("components", [])):
                comp_name = component.get("name", f"Journey Stage {comp_index + 1}")
                section = new_state.create_section_from_template(act_index, comp_index, comp_name)
                story_sections.append(section)

        # Add sections to story in database
        if story_sections:
            for section in story_sections:
                # Update the story object
                if new_state.story:
                    new_state.story.add_section(section)

            # Update the database
            from agents.tools import update_story

            update_story(
                new_state.story_id, {"sections": [section.dict() for section in story_sections]}
            )

        # Distribute sections among writers
        if new_state.num_writers > 1 or new_state.use_joint_llm:
            writer_ids = list(new_state.writing_team.keys())

            # Convert story sections to dictionary format for distribution
            sections_for_distribution = [
                {
                    "id": section.id,
                    "title": section.title,
                    "description": f"Act: {section.act}, Sequence: {section.sequence}",
                    "sequence": section.sequence,
                }
                for section in story_sections
            ]

            # Distribute sections among writers
            distribution = distribute_sections_to_writers(
                sections_for_distribution, writer_ids, new_state.use_joint_llm
            )

            # Process distribution and assign writers
            for writer_id, sections in distribution.items():
                if writer_id == "joint_writer" and new_state.use_joint_llm:
                    # These sections will be handled by the joint writer
                    joint_writer_id = next(iter(new_state.joint_writers.keys()), None)

                    if joint_writer_id:
                        section_ids = [section["id"] for section in sections]
                        new_state.assign_writer_to_sections(joint_writer_id, section_ids, True)
                else:
                    # Normal writer assignments
                    section_ids = [section["id"] for section in sections]
                    new_state.assign_writer_to_sections(writer_id, section_ids, False)
        else:
            # Single writer - assign all sections to the first writer
            writer_id = next(iter(new_state.writing_team.keys()))
            section_ids = [section.id for section in story_sections]
            new_state.assign_writer_to_sections(writer_id, section_ids, False)

    except Exception as e:
        error_msg = f"Error processing outline: {str(e)}"
        new_state.add_error("system", error_msg)

    # Start the writing phase
    start_writing_phase(new_state)

    return new_state


def start_writing_phase(state: GraphState) -> GraphState:
    """Start the writing phase by assigning draft creation tasks to writers."""
    new_state = state.copy()

    # Update story state if not already set
    if new_state.current_state != StoryState.WRITING:
        new_state.update_story_state(StoryState.WRITING)

    # If we have writer assignments, use those
    if new_state.writer_assignments:
        # Process each writer assignment
        for assignment in new_state.writer_assignments:
            writer_id = assignment.writer_id
            section_ids = assignment.sections
            is_joint_llm = assignment.joint_llm

            # Skip if no sections to write
            if not section_ids:
                continue

            # Get the writer
            if is_joint_llm:
                # This is for the joint writer
                if writer_id in new_state.joint_writers:
                    writer = new_state.joint_writers[writer_id]
                else:
                    # Skip if joint writer doesn't exist
                    continue
            else:
                # Regular writer
                if writer_id in new_state.writing_team:
                    writer = new_state.writing_team[writer_id]
                else:
                    # Skip if writer doesn't exist
                    continue

            # Get the first section to write
            first_section_id = section_ids[0] if section_ids else None

            if not first_section_id:
                continue

            # Get the section from the story
            section = None
            if new_state.story and new_state.story.sections:
                section = next(
                    (s for s in new_state.story.sections if s.id == first_section_id), None
                )

            if not section:
                # Try to create a generic section if not found
                section_title = f"Section {first_section_id}"

                # Create a task to write this section
                task_id = create_task_id("write_section", writer_id)

                # Use a generic prompt
                prompt = f"""
                # Writing Task: {section_title}
                
                Please write the content for this section of the story based on the approved outline.
                
                ## Story Structure
                The story follows a {new_state.story_structure.value} structure.
                
                ## Your Assignment
                You are assigned to write section ID: {first_section_id}
                
                Please write engaging, high-quality content that follows the story outline and fits within the overall narrative flow.
                """
            else:
                # Create a task to write this specific section
                task_id = create_task_id("write_section", writer_id)

                # Get relevant bible sections and outline
                bible_sections = ""
                from agents.tools import get_bible_sections

                bible_result = get_bible_sections(
                    new_state.story_id, BibleSectionType.PLOT_ELEMENTS.value
                )

                if bible_result.get("status") != "error" and "sections" in bible_result:
                    plot_sections = bible_result["sections"].get(
                        BibleSectionType.PLOT_ELEMENTS.value, []
                    )
                    for plot_section in plot_sections:
                        bible_sections += (
                            f"\n## {plot_section['title']}\n{plot_section['content']}\n"
                        )

                # Get outline for context
                outline = ""
                outline_messages = [
                    msg
                    for msg in new_state.messages
                    if msg.get("metadata", {}).get("task_type") == "outline_complete"
                ]

                if outline_messages:
                    outline = outline_messages[-1]["content"]

                # Format the task
                prompt = WRITING_TASK_PROMPT.format(
                    task_description=f"Write the content for the section: {section.title}",
                    story_request=(
                        new_state.user_request.to_prompt_string()
                        if new_state.user_request
                        else "No specific request details provided"
                    ),
                    section=section.title,
                    target_length="Appropriate length for this section - typically 1000-1500 words",
                    tone_style=(
                        new_state.user_request.style if new_state.user_request else "Not specified"
                    ),
                    reference_materials="",
                    bible_entries=bible_sections,
                    outline_elements=f"## Section Context\nAct: {section.act}\nSequence: {section.sequence}\n\n{outline}",
                    previous_feedback="",
                )

            # Add task to state
            new_state.add_task(
                task_id=task_id,
                agent_id=writer_id,
                task_type="write_section",
                description=f"Write section: {section_title if 'section_title' in locals() else section.title}",
                data={"section_id": first_section_id, "is_joint_llm": is_joint_llm},
            )

            # Update writer state
            writer.current_section = first_section_id
            writer.update_status("working", f"writing_section_{first_section_id}")

            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=writer_id,
                content=prompt,
                metadata={
                    "task_id": task_id,
                    "task_type": "write_section",
                    "section_id": first_section_id,
                    "is_joint_llm": is_joint_llm,
                },
            )
    else:
        # No writer assignments - use default approach
        # Get the first writer
        writer_id = next(iter(new_state.writing_team.keys()))
        writer = new_state.writing_team[writer_id]

        # Create a simple section structure
        sections = ["beginning", "middle", "end"]

        # Assign the first section
        current_section = sections[0]

        # Create draft task
        task_id = create_task_id("draft", writer_id)

        # Get the outline
        outline = ""
        outline_messages = [
            msg
            for msg in new_state.messages
            if msg.get("metadata", {}).get("task_type") == "outline_complete"
        ]

        if outline_messages:
            outline = outline_messages[-1]["content"]

        # Format the task
        prompt = WRITING_TASK_PROMPT.format(
            task_description=f"Write the {current_section} section of the story based on the approved outline.",
            story_request=(
                new_state.user_request.to_prompt_string()
                if new_state.user_request
                else "No specific request details provided"
            ),
            section=current_section,
            target_length="Appropriate length for this section, typically 1000-1500 words",
            tone_style=new_state.user_request.style if new_state.user_request else "Not specified",
            reference_materials="",
            bible_entries="",
            outline_elements=outline,
            previous_feedback="",
        )

        # Add task to state
        new_state.add_task(
            task_id=task_id,
            agent_id=writer_id,
            task_type="write_draft",
            description=f"Write {current_section} section of story",
            data={
                "section": current_section,
                "outline": outline,
                "story_structure": new_state.story_structure.value,
            },
        )

        # Update writer state
        writer.update_status("working", f"writing_{current_section}")
        writer.current_section = current_section
        writer.assigned_sections = sections

        # Add message to communication log
        new_state.add_message(
            sender="system",
            recipient=writer_id,
            content=prompt,
            metadata={"task_id": task_id, "task_type": "write_draft", "section": current_section},
        )

    return new_state


def write_story_section(state: GraphState) -> GraphState:
    """Write a section of the story with the assigned writer agent."""
    new_state = state.copy()

    # Find active writer agents with assigned tasks
    writer_states = []

    # Check regular writers
    for writer_id, writer in new_state.writing_team.items():
        if writer.status == "working" and writer.current_section:
            writer_states.append((writer_id, writer, False))  # False = not joint LLM

    # Check joint writers
    for writer_id, writer in new_state.joint_writers.items():
        if writer.status == "working" and writer.current_section:
            writer_states.append((writer_id, writer, True))  # True = joint LLM

    if not writer_states:
        # No writers actively working
        return new_state

    # Process the first active writer
    writer_id, writer, is_joint_llm = writer_states[0]

    # Find the writing task message
    section_id = writer.current_section

    section_messages = [
        msg
        for msg in new_state.messages
        if msg["recipient"] == writer_id
        and (
            msg.get("metadata", {}).get("task_type") == "write_section"
            or msg.get("metadata", {}).get("task_type") == "write_draft"
        )
        and (
            msg.get("metadata", {}).get("section_id") == section_id
            or msg.get("metadata", {}).get("section") == section_id
        )
    ]

    if not section_messages:
        # No writing task found for this section
        return new_state

    latest_section_message = section_messages[-1]
    task_id = latest_section_message.get("metadata", {}).get("task_id")

    # Create the appropriate writer agent
    if is_joint_llm:
        # Create joint writer agent
        writer_agent = create_joint_writer_agent(writer_id)
    else:
        # Create regular writer agent
        writer_agent = create_writer_agent(writer_id)

    try:
        # Execute section writing task
        response = writer_agent.invoke(
            {
                "input": latest_section_message["content"],
                "agent_id": writer_id,
                "story_id": new_state.story_id,
                "section_id": section_id,
            }
        )

        if "output" in response:
            # Update writer state
            writer.update_status("completed", f"completed_section_{section_id}")
            writer.mark_section_complete(section_id)
            writer.add_draft_content(section_id, response["output"])

            # Add message to communication log
            new_state.add_message(
                sender=writer_id,
                recipient="system",
                content=response["output"],
                metadata={
                    "task_id": task_id,
                    "task_type": "section_complete",
                    "section_id": section_id,
                    "is_joint_llm": is_joint_llm,
                },
            )

            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"section_id": section_id, "content": response["output"]},
            )

            # Update the section in the story object
            if new_state.story:
                for i, section in enumerate(new_state.story.sections):
                    if section.id == section_id:
                        new_state.story.sections[i].content = response["output"]
                        new_state.story.sections[i].status = "completed"
                        new_state.story.sections[i].updated_at = current_timestamp()
                        break

            # Check if this writer has more sections to write
            more_sections = False

            if isinstance(writer.assigned_sections, list):
                # Find sections that haven't been completed yet
                for assigned_section in writer.assigned_sections:
                    if (
                        assigned_section != section_id
                        and assigned_section not in writer.completed_sections
                    ):
                        next_section_id = assigned_section
                        more_sections = True

                        # Get the section details
                        next_section = None
                        if new_state.story and new_state.story.sections:
                            next_section = next(
                                (s for s in new_state.story.sections if s.id == next_section_id),
                                None,
                            )

                        # Create next task
                        next_task_id = create_task_id("write_section", writer_id)

                        # Get bible sections and outline
                        bible_sections = ""
                        from agents.tools import get_bible_sections

                        bible_result = get_bible_sections(
                            new_state.story_id, BibleSectionType.PLOT_ELEMENTS.value
                        )

                        if bible_result.get("status") != "error" and "sections" in bible_result:
                            plot_sections = bible_result["sections"].get(
                                BibleSectionType.PLOT_ELEMENTS.value, []
                            )
                            for plot_section in plot_sections:
                                bible_sections += (
                                    f"\n## {plot_section['title']}\n{plot_section['content']}\n"
                                )

                        # Get outline for context
                        outline = ""
                        outline_messages = [
                            msg
                            for msg in new_state.messages
                            if msg.get("metadata", {}).get("task_type") == "outline_complete"
                        ]

                        if outline_messages:
                            outline = outline_messages[-1]["content"]

                        section_title = (
                            next_section.title if next_section else f"Section {next_section_id}"
                        )
                        section_act = next_section.act if next_section else ""
                        section_sequence = next_section.sequence if next_section else 0

                        # Format the task
                        next_prompt = WRITING_TASK_PROMPT.format(
                            task_description=f"Write the content for section: {section_title}",
                            story_request=(
                                new_state.user_request.to_prompt_string()
                                if new_state.user_request
                                else "No specific request details provided"
                            ),
                            section=section_title,
                            target_length="Appropriate length for this section - typically 1000-1500 words",
                            tone_style=(
                                new_state.user_request.style
                                if new_state.user_request
                                else "Not specified"
                            ),
                            reference_materials="",
                            bible_entries=bible_sections,
                            outline_elements=f"## Section Context\nAct: {section_act}\nSequence: {section_sequence}\n\n{outline}",
                            previous_feedback="",
                        )

                        # Add task to state
                        new_state.add_task(
                            task_id=next_task_id,
                            agent_id=writer_id,
                            task_type="write_section",
                            description=f"Write section: {section_title}",
                            data={"section_id": next_section_id, "is_joint_llm": is_joint_llm},
                        )

                        # Update writer state for next section
                        writer.current_section = next_section_id
                        writer.update_status("working", f"writing_section_{next_section_id}")

                        # Add message to communication log
                        new_state.add_message(
                            sender="system",
                            recipient=writer_id,
                            content=next_prompt,
                            metadata={
                                "task_id": next_task_id,
                                "task_type": "write_section",
                                "section_id": next_section_id,
                                "is_joint_llm": is_joint_llm,
                            },
                        )

                        # Only assign one section at a time
                        break

            # If no more sections for this writer, check if all writing is complete
            if not more_sections:
                writer.update_status("idle", "all_sections_completed")

                # Check if all writers have completed their sections
                all_complete = True

                for w_id, w_state in new_state.writing_team.items():
                    if w_state.status != "idle" and w_state.status != "error":
                        all_complete = False
                        break

                for w_id, w_state in new_state.joint_writers.items():
                    if w_state.status != "idle" and w_state.status != "error":
                        all_complete = False
                        break

                if all_complete:
                    # All writing is complete - move to editing phase
                    # Combine all content into a complete draft
                    combined_draft = ""

                    # Get all sections in order
                    sections = []
                    if new_state.story and new_state.story.sections:
                        # Sort sections by sequence number
                        sections = sorted(new_state.story.sections, key=lambda s: s.sequence)

                        for section in sections:
                            if section.content:
                                combined_draft += f"\n\n## {section.title}\n\n{section.content}"

                    if not combined_draft and new_state.writing_team:
                        # Fallback - get content from writer states
                        for w_id, w_state in new_state.writing_team.items():
                            for section_id, content in w_state.draft_content.items():
                                combined_draft += f"\n\n## Section: {section_id}\n\n{content}"

                    if combined_draft:
                        new_state.update_story_state(StoryState.EDITING)
                        start_editing_phase(new_state, combined_draft)
                    else:
                        new_state.add_error("system", "No content found after writing phase")
    except Exception as e:
        error_msg = f"Error writing section {section_id}: {str(e)}"
        new_state.add_error("writer", error_msg)

        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id, status="error", result={"error": error_msg}
            )

        writer.update_status("error", f"section_writing_error_{section_id}")

    return new_state


# Build the complete graph with all nodes
def build_graph():
    """Build the workflow graph for the Storybook application."""
    # Create a StateGraph with GraphState
    workflow = StateGraph(GraphState)

    # Add nodes for each phase of the workflow
    workflow.add_node("initialize", initialize_workflow)
    workflow.add_node("process_user_request", process_user_request)

    # Operation mode specific nodes
    workflow.add_node("analyze_imported_content", analyze_imported_content)
    workflow.add_node("create_reverse_outline", create_reverse_outline)
    workflow.add_node("plan_edit_continuation", plan_edit_continuation)
    workflow.add_node("execute_edit_continuation", execute_edit_continuation)
    workflow.add_node("review_edit_continuation", review_edit_continuation)
    workflow.add_node("process_human_edit_approval", process_human_edit_approval)

    # Briefing phase
    workflow.add_node("conduct_briefing", conduct_briefing)
    workflow.add_node("process_human_briefing_input", process_human_briefing_input)

    # Research phase
    workflow.add_node("start_research_phase", start_research_phase)
    workflow.add_node("conduct_research", conduct_research)
    workflow.add_node("review_research", review_research)

    # Planning phase (and bible updates)
    workflow.add_node("update_story_bible", update_story_bible)
    workflow.add_node("start_planning_phase", start_planning_phase)
    workflow.add_node("create_story_outline", create_story_outline)
    workflow.add_node("review_outline", review_outline)
    workflow.add_node("process_human_outline_review", process_human_outline_review)

    # Writing phase
    workflow.add_node("start_writing_phase", start_writing_phase)
    workflow.add_node("write_story_section", write_story_section)

    # Editing phase
    workflow.add_node("start_editing_phase", start_editing_phase)
    workflow.add_node("edit_story", edit_story)
    workflow.add_node("review_final_draft", review_final_draft)
    workflow.add_node("process_human_draft_review", process_human_draft_review)

    # Publishing phase
    workflow.add_node("start_publishing_phase", start_publishing_phase)
    workflow.add_node("prepare_for_publishing", prepare_for_publishing)
    workflow.add_node("review_publishing_package", review_publishing_package)
    workflow.add_node("process_human_publishing_approval", process_human_publishing_approval)
    workflow.add_node("publish_story", publish_story)
    workflow.add_node("execute_publishing", execute_publishing)

    # Define the edges (transitions) between nodes
    # Start with initialization and user request processing
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "process_user_request")

    # Base on operation mode after user request
    workflow.add_conditional_edges(
        "process_user_request",
        lambda state: (
            state.operation_mode.value if state.operation_mode else OperationMode.CREATE.value
        ),
        {
            OperationMode.CREATE.value: "conduct_briefing",
            OperationMode.IMPORT.value: "analyze_imported_content",
            OperationMode.EDIT.value: "plan_edit_continuation",
            OperationMode.CONTINUE.value: "plan_edit_continuation",
        },
    )

    # Import mode flow
    workflow.add_edge("analyze_imported_content", "create_reverse_outline")
    workflow.add_edge("create_reverse_outline", "update_story_bible")

    # Edit/continue mode flow
    workflow.add_edge("plan_edit_continuation", "update_story_bible")
    workflow.add_edge("update_story_bible", "execute_edit_continuation")
    workflow.add_edge("execute_edit_continuation", "review_edit_continuation")
    workflow.add_conditional_edges(
        "review_edit_continuation",
        lambda state: (
            "awaiting_human_input" if state.awaiting_human_input else "execute_edit_continuation"
        ),
        {
            "awaiting_human_input": "process_human_edit_approval",
            "execute_edit_continuation": "execute_edit_continuation",
        },
    )
    workflow.add_conditional_edges(
        "process_human_edit_approval",
        lambda state: (
            "start_publishing_phase"
            if state.current_state == StoryState.READY_FOR_PUBLISHING
            else "execute_edit_continuation"
        ),
        {
            "start_publishing_phase": "start_publishing_phase",
            "execute_edit_continuation": "execute_edit_continuation",
        },
    )

    # Create mode - Briefing phase
    workflow.add_conditional_edges(
        "conduct_briefing",
        lambda state: (
            "awaiting_human_input" if state.awaiting_human_input else "start_research_phase"
        ),
        {
            "awaiting_human_input": "process_human_briefing_input",
            "start_research_phase": "start_research_phase",
        },
    )
    workflow.add_edge("process_human_briefing_input", "conduct_briefing")

    # Research phase
    workflow.add_edge("start_research_phase", "conduct_research")
    workflow.add_edge("conduct_research", "review_research")
    workflow.add_conditional_edges(
        "review_research",
        lambda state: (
            "conduct_research"
            if any(r.status == "revising_research" for r in state.research_team.values())
            else "update_story_bible"
        ),
        {"conduct_research": "conduct_research", "update_story_bible": "update_story_bible"},
    )

    # Planning phase
    workflow.add_edge("update_story_bible", "start_planning_phase")
    workflow.add_edge("start_planning_phase", "create_story_outline")
    workflow.add_edge("create_story_outline", "review_outline")
    workflow.add_conditional_edges(
        "review_outline",
        lambda state: (
            "awaiting_human_input"
            if state.awaiting_human_input
            else (
                "create_story_outline"
                if any(w.status == "revising_outline" for w in state.writing_team.values())
                else "start_writing_phase"
            )
        ),
        {
            "awaiting_human_input": "process_human_outline_review",
            "create_story_outline": "create_story_outline",
            "start_writing_phase": "start_writing_phase",
        },
    )
    workflow.add_conditional_edges(
        "process_human_outline_review",
        lambda state: (
            "create_story_outline"
            if any(w.status == "human_outline_revision" for w in state.writing_team.values())
            else (
                "start_writing_phase"
                if state.current_state == StoryState.WRITING
                else "start_planning_phase"
            )
        ),  # For full restarts
        {
            "create_story_outline": "create_story_outline",
            "start_writing_phase": "start_writing_phase",
            "start_planning_phase": "start_planning_phase",
        },
    )

    # Writing phase
    workflow.add_edge("start_writing_phase", "write_story_section")
    workflow.add_conditional_edges(
        "write_story_section",
        lambda state: (
            "write_story_section"
            if any(
                w.status not in ["idle", "error", "completed"]
                for w in list(state.writing_team.values()) + list(state.joint_writers.values())
            )
            else "start_editing_phase"
        ),
        {
            "write_story_section": "write_story_section",
            "start_editing_phase": "start_editing_phase",
        },
    )

    # Editing phase
    workflow.add_conditional_edges(
        "start_editing_phase",
        lambda state: "edit_story" if isinstance(state, GraphState) else "edit_story",
        {"edit_story": "edit_story"},
    )
    workflow.add_edge("edit_story", "review_final_draft")
    workflow.add_conditional_edges(
        "review_final_draft",
        lambda state: (
            "awaiting_human_input"
            if state.awaiting_human_input
            else (
                "edit_story"
                if any(e.status == "revising" for e in state.editing_team.values())
                else "start_publishing_phase"
            )
        ),
        {
            "awaiting_human_input": "process_human_draft_review",
            "edit_story": "edit_story",
            "start_publishing_phase": "start_publishing_phase",
        },
    )
    workflow.add_conditional_edges(
        "process_human_draft_review",
        lambda state: (
            "edit_story"
            if any(e.status == "human_revisions" for e in state.editing_team.values())
            else "start_publishing_phase"
        ),
        {"edit_story": "edit_story", "start_publishing_phase": "start_publishing_phase"},
    )

    # Publishing phase
    workflow.add_edge("start_publishing_phase", "prepare_for_publishing")
    workflow.add_edge("prepare_for_publishing", "review_publishing_package")
    workflow.add_conditional_edges(
        "review_publishing_package",
        lambda state: (
            "awaiting_human_input"
            if state.awaiting_human_input
            else (
                "prepare_for_publishing"
                if any(p.status == "revising_publishing" for p in state.publishing_team.values())
                else "publish_story"
            )
        ),
        {
            "awaiting_human_input": "process_human_publishing_approval",
            "prepare_for_publishing": "prepare_for_publishing",
            "publish_story": "publish_story",
        },
    )
    workflow.add_conditional_edges(
        "process_human_publishing_approval",
        lambda state: (
            "prepare_for_publishing"
            if any(
                p.status in ["human_publishing_changes", "major_publishing_revision"]
                for p in state.publishing_team.values()
            )
            else "publish_story"
        ),
        {"prepare_for_publishing": "prepare_for_publishing", "publish_story": "publish_story"},
    )
    workflow.add_edge("publish_story", "execute_publishing")

    # Allow multiple tries for error cases
    workflow.add_edge(
        "execute_publishing",
        "execute_publishing",
        condition=lambda state: any(p.status == "error" for p in state.publishing_team.values()),
    )

    return workflow


# Create the graph instance
graph = build_graph().compile()
