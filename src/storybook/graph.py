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
    StoryState, AgentRole, TeamType, BibleSectionType, FeedbackType, StoryStructure, OperationMode,
    UserRequest, ResearchItem, BibleSection, StoryBible, Character, 
    PlotPoint, Setting, StoryOutline, StorySection, Feedback, PublishingMetadata, Story, WriterAssignment,
    STORY_STRUCTURES, USE_GPU, USE_OLLAMA
)

from agents.state import (
    GraphState, AgentState, ResearchAgentState, WritingAgentState, JointWriterAgentState,
    EditingAgentState, PublishingAgentState, SupervisorAgentState,
    AuthorRelationsAgentState, HumanInLoopState, StyleGuideEditorState
)

from agents.prompts import (
    RESEARCHER_SYSTEM_PROMPT, RESEARCH_SUPERVISOR_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT, EDITOR_SYSTEM_PROMPT, WRITING_SUPERVISOR_SYSTEM_PROMPT,
    PUBLISHER_SYSTEM_PROMPT, PUBLISHING_SUPERVISOR_SYSTEM_PROMPT,
    AUTHOR_RELATIONS_SYSTEM_PROMPT, HUMAN_IN_LOOP_SYSTEM_PROMPT,
    STYLE_GUIDE_EDITOR_SYSTEM_PROMPT,
    RESEARCH_TASK_PROMPT, WRITING_TASK_PROMPT, EDITING_TASK_PROMPT,
    PUBLISHING_TASK_PROMPT, REVIEW_TASK_PROMPT, BIBLE_UPDATE_PROMPT,
    BRAINSTORM_SESSION_PROMPT, HUMAN_REVIEW_PROMPT
)

from agents.tools import (
    RESEARCH_TOOLS, WRITING_TOOLS, EDITING_TOOLS, PUBLISHING_TOOLS,
    SUPERVISOR_TOOLS, AUTHOR_RELATIONS_TOOLS, BIBLE_EDITOR_TOOLS,
    HUMAN_IN_LOOP_TOOLS
)

from agents.utils import (
    generate_id, current_timestamp, extract_json_from_text, 
    format_message_history, clean_and_format_text, create_task_id,
    parse_feedback, format_agent_response, validate_story_structure,
    prepare_human_review_prompt, format_brainstorm_session,
    get_story_structure_template, create_model_instance,
    create_section_structure_from_template, distribute_sections_to_writers
)

# Initialize tool executor
tool_executor = ToolExecutor()

# Agent Factory Functions
def create_researcher_agent(agent_id: str) -> Any:
    """Create a researcher agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("research", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=RESEARCH_TOOLS,
        system_message=RESEARCHER_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=RESEARCH_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_research_supervisor_agent(agent_id: str) -> Any:
    """Create a research supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        system_message=RESEARCH_SUPERVISOR_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + RESEARCH_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_writer_agent(agent_id: str) -> Any:
    """Create a writer agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("writing", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=WRITING_TOOLS,
        system_message=WRITER_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_joint_writer_agent(agent_id: str, component_writer_ids: List[str] = None) -> Any:
    """Create a joint writer agent that combines the power of multiple models."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    # This agent uses a more powerful model since it handles complex sections
    llm = create_model_instance("writing", use_local=False)  # Always use API model for joint writing
    
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
        llm=llm,
        tools=WRITING_TOOLS,
        system_message=joint_writer_prompt
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_editor_agent(agent_id: str) -> Any:
    """Create an editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("writing", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=EDITING_TOOLS,
        system_message=EDITOR_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=EDITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_writing_supervisor_agent(agent_id: str) -> Any:
    """Create a writing supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        system_message=WRITING_SUPERVISOR_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + WRITING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_publisher_agent(agent_id: str) -> Any:
    """Create a publisher agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("publishing", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=PUBLISHING_TOOLS,
        system_message=PUBLISHER_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=PUBLISHING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_publishing_supervisor_agent(agent_id: str) -> Any:
    """Create a publishing supervisor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("supervisor", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
        system_message=PUBLISHING_SUPERVISOR_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=SUPERVISOR_TOOLS + PUBLISHING_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_author_relations_agent(agent_id: str) -> Any:
    """Create an author relations agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("author_relations", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=AUTHOR_RELATIONS_TOOLS,
        system_message=AUTHOR_RELATIONS_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=AUTHOR_RELATIONS_TOOLS,
        verbose=True,
        handle_parsing_errors=True
    )

def create_style_guide_editor_agent(agent_id: str) -> Any:
    """Create a style guide editor agent with appropriate tools."""
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    
    llm = create_model_instance("writing", use_local=USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=BIBLE_EDITOR_TOOLS,
        system_message=STYLE_GUIDE_EDITOR_SYSTEM_PROMPT
    )
    
    return AgentExecutor(
        agent=prompt,
        tools=BIBLE_EDITOR_TOOLS,
        verbose=True,
        handle_parsing_errors=True
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
            agent_id=researcher_id,
            agent_role=AgentRole.RESEARCHER,
            status="idle"
        )
    
    # Research supervisor
    if not any(sup.agent_role == AgentRole.RESEARCH_SUPERVISOR for sup in new_state.supervisors.values()):
        supervisor_id = generate_id("rsup")
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.RESEARCH_SUPERVISOR,
            team_type=TeamType.RESEARCH,
            status="idle",
            supervised_agents=list(new_state.research_team.keys())
        )
    
    # Writing team - create multiple writer agents based on num_writers
    if not new_state.writing_team:
        for i in range(new_state.num_writers):
            writer_id = generate_id(f"writer_{i+1}")
            new_state.writing_team[writer_id] = WritingAgentState(
                agent_id=writer_id,
                agent_role=AgentRole.WRITER,
                status="idle"
            )
        
        # Create joint writer if enabled
        if new_state.use_joint_llm:
            joint_writer_id = generate_id("joint_writer")
            new_state.joint_writers[joint_writer_id] = JointWriterAgentState(
                agent_id=joint_writer_id,
                agent_role=AgentRole.JOINT_WRITER,
                status="idle",
                component_writers=list(new_state.writing_team.keys()),
                is_joint_llm=True
            )
    
    # Editing team
    if not new_state.editing_team:
        editor_id = generate_id("editor")
        new_state.editing_team[editor_id] = EditingAgentState(
            agent_id=editor_id,
            agent_role=AgentRole.EDITOR,
            status="idle"
        )
    
    # Writing supervisor
    if not any(sup.agent_role == AgentRole.WRITING_SUPERVISOR for sup in new_state.supervisors.values()):
        supervisor_id = generate_id("wsup")
        writer_ids = list(new_state.writing_team.keys())
        joint_writer_ids = list(new_state.joint_writers.keys())
        editor_ids = list(new_state.editing_team.keys())
        
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.WRITING_SUPERVISOR,
            team_type=TeamType.WRITING,
            status="idle",
            supervised_agents=writer_ids + joint_writer_ids + editor_ids
        )
    
    # Publishing team
    if not new_state.publishing_team:
        publisher_id = generate_id("publisher")
        new_state.publishing_team[publisher_id] = PublishingAgentState(
            agent_id=publisher_id,
            agent_role=AgentRole.PUBLISHER,
            status="idle"
        )
    
    # Publishing supervisor
    if not any(sup.agent_role == AgentRole.PUBLISHING_SUPERVISOR for sup in new_state.supervisors.values()):
        supervisor_id = generate_id("psup")
        new_state.supervisors[supervisor_id] = SupervisorAgentState(
            agent_id=supervisor_id,
            agent_role=AgentRole.PUBLISHING_SUPERVISOR,
            team_type=TeamType.PUBLISHING,
            status="idle",
            supervised_agents=list(new_state.publishing_team.keys())
        )
    
    # Special agents
    # Author relations agent
    if not new_state.author_relations:
        author_relations_id = generate_id("author_relations")
        new_state.author_relations[author_relations_id] = AuthorRelationsAgentState(
            agent_id=author_relations_id,
            agent_role=AgentRole.AUTHOR_RELATIONS,
            status="idle"
        )
    
    # Style guide editor
    if not new_state.style_guide_editor:
        style_guide_id = generate_id("style_guide")
        new_state.style_guide_editor[style_guide_id] = StyleGuideEditorState(
            agent_id=style_guide_id,
            agent_role=AgentRole.STYLE_GUIDE_EDITOR,
            status="idle"
        )
    
    # Human in the loop
    if not new_state.human_in_loop:
        human_id = generate_id("human")
        new_state.human_in_loop[human_id] = HumanInLoopState(
            agent_id=human_id,
            agent_role=AgentRole.HUMAN_IN_LOOP,
            status="ready"
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
                operation_mode=new_state.operation_mode
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
                operation_mode=new_state.operation_mode
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
                operation_mode=new_state.operation_mode
            )
    
    # Initialize empty bible if none exists
    if not new_state.bible:
        new_state.bible = StoryBible(
            story_id=new_state.story_id,
            created_at=current_timestamp(),
            updated_at=current_timestamp()
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
            data={
                "user_request": user_request.dict()
            }
        )
        
        # Update author relations agent state
        author_agent = new_state.author_relations[author_relations_id]
        author_agent.update_status("working", "conducting_briefing")
        
        # Add initial message to communication log
        new_state.add_message(
            sender="system",
            recipient=author_relations_id,
            content=f"New story request received. Please conduct an initial briefing with the client.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nStory Structure: {new_state.story_structure.value}",
            metadata={"task_id": briefing_task_id, "task_type": "briefing"}
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
                "user_request": user_request.dict()
            }
        )
        
        # Update style guide editor state
        style_guide = new_state.style_guide_editor[style_guide_id]
        style_guide.update_status("working", "analyzing_import")
        
        # Add initial message to communication log
        new_state.add_message(
            sender="system",
            recipient=style_guide_id,
            content=f"Imported content received. Please analyze it and create initial bible entries.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nImported Content Preview:\n{user_request.existing_content[:500]}...",
            metadata={"task_id": analysis_task_id, "task_type": "analyze_import"}
        )
    
    elif new_state.operation_mode in [OperationMode.EDIT, OperationMode.CONTINUE]:
        # Start the edit/continue process
        new_state.update_story_state(StoryState.REVISION)
        new_state.imported_content = user_request.existing_content
        new_state.sections_to_edit = user_request.sections_to_edit or []
        
        # Create a task for planning the edits
        writing_supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                                    if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
        
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
                "is_continuation": new_state.operation_mode == OperationMode.CONTINUE
            }
        )
        
        # Update supervisor state
        supervisor = new_state.supervisors[writing_supervisor_id]
        supervisor.update_status("working", "planning_edits")
        
        # Add initial message to communication log
        operation_type = "continuation" if new_state.operation_mode == OperationMode.CONTINUE else "edits"
        new_state.add_message(
            sender="system",
            recipient=writing_supervisor_id,
            content=f"Request for {operation_type} received. Please plan the necessary work.\n\nRequest details:\n{user_request.to_prompt_string()}\n\nExisting Content Preview:\n{user_request.existing_content[:500]}...\n\nSections to Edit: {', '.join(user_request.sections_to_edit or ['All sections' if new_state.operation_mode == OperationMode.EDIT else 'Continue from end'])}",
            metadata={"task_id": planning_task_id, "task_type": f"plan_{operation_type}"}
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
    analysis_messages = [msg for msg in new_state.messages 
                        if msg["recipient"] == style_guide_id 
                        and msg.get("metadata", {}).get("task_type") == "analyze_import"]
    
    if not analysis_messages:
        # No analysis task found
        return new_state
    
    latest_analysis_message = analysis_messages[-1]
    task_id = latest_analysis_message.get("metadata", {}).get("task_id")
    
    # Create the style guide editor agent
    style_guide_agent = create_style_guide_editor_agent(style_guide_id)
    
    try:
        # Execute analysis task
        response = style_guide_agent.invoke({
            "input": latest_analysis_message["content"],
            "agent_id": style_guide_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update style guide editor state
            style_guide.update_status("completed", "import_analyzed")
            
            # Add message to communication log
            new_state.add_message(
                sender=style_guide_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "import_analysis_complete"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"analysis_output": response["output"]}
            )
            
            # Create a reverse outline task for the writing supervisor
            supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                                if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
            
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
                    "analysis": response["output"]
                }
            )
            
            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", "creating_reverse_outline")
            
            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=outline_prompt,
                metadata={"task_id": outline_task_id, "task_type": "reverse_outline"}
            )
    except Exception as e:
        error_msg = f"Error analyzing imported content: {str(e)}"
        new_state.add_error("style_guide_editor", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
            )
        
        style_guide.update_status("error", "import_analysis_error")
        
        # Still try to move to reverse outlining
        supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                            if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
        
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
            data={
                "imported_content": new_state.imported_content,
                "error_recovery": True
            }
        )
        
        # Update supervisor state
        supervisor = new_state.supervisors[supervisor_id]
        supervisor.update_status("working", "creating_reverse_outline")
        
        # Add message to communication log
        new_state.add_message(
            sender="system",
            recipient=supervisor_id,
            content=outline_prompt,
            metadata={"task_id": outline_task_id, "task_type": "reverse_outline"}
        )
    
    return new_state

def create_reverse_outline(state: GraphState) -> GraphState:
    """Create a reverse outline from imported content."""
    new_state = state.copy()
    
    # This node is for the IMPORT operation mode
    if new_state.operation_mode != OperationMode.IMPORT:
        return new_state
    
    # Get supervisor and their task
    supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                        if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
    supervisor = new_state.supervisors[supervisor_id]
    
    # Find the reverse outline task message
    outline_messages = [msg for msg in new_state.messages 
                       if msg["recipient"] == supervisor_id 
                       and msg.get("metadata", {}).get("task_type") == "reverse_outline"]
    
    if not outline_messages:
        # No outline task found
        return new_state
    
    latest_outline_message = outline_messages[-1]
    task_id = latest_outline_message.get("metadata", {}).get("task_id")
    
    # Create the supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)
    
    try:
        # Execute reverse outline task
        response = supervisor_agent.invoke({
            "input": latest_outline_message["content"],
            "agent_id": supervisor_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "reverse_outline_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "reverse_outline_complete"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"outline_output": response["output"]}
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
                related_sections=""
            )
            
            # Add task to state
            new_state.add_task(
                task_id=bible_task_id,
                agent_id=style_guide_id,
                task_type="bible_update",
                description="Create bible sections from reverse outline",
                data={
                    "reverse_outline": response["output"]
                }
            )
            
            # Update style guide editor state
            style_guide = new_state.style_guide_editor[style_guide_id]
            style_guide.update_status("working", "updating_bible")
            
            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=style_guide_id,
                content=update_prompt,
                metadata={"task_id": bible_task_id, "task_type": "bible_update"}
            )
            
            # After reverse outline, set up for continuation or extension
            new_state.update_story_state(StoryState.PLANNING)
    except Exception as e:
        error_msg = f"Error creating reverse outline: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
    supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                        if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
    supervisor = new_state.supervisors[supervisor_id]
    
    # Find the planning task message
    is_continuation = new_state.operation_mode == OperationMode.CONTINUE
    task_type = "plan_continuation" if is_continuation else "plan_edits"
    
    planning_messages = [msg for msg in new_state.messages 
                        if msg["recipient"] == supervisor_id 
                        and msg.get("metadata", {}).get("task_type") in ["plan_edits", "plan_continuation"]]
    
    if not planning_messages:
        # No planning task found
        return new_state
    
    latest_planning_message = planning_messages[-1]
    task_id = latest_planning_message.get("metadata", {}).get("task_id")
    
    # Create the supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)
    
    try:
        # Execute planning task
        response = supervisor_agent.invoke({
            "input": latest_planning_message["content"],
            "agent_id": supervisor_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "edit_plan_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"plan_output": response["output"]}
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
                related_sections=""
            )
            
            # Add task to state
            new_state.add_task(
                task_id=bible_task_id,
                agent_id=style_guide_id,
                task_type="bible_update",
                description=f"Update bible for {'continuation' if is_continuation else 'edits'}",
                data={
                    "plan": response["output"],
                    "is_continuation": is_continuation
                }
            )
            
            # Update style guide editor state
            style_guide = new_state.style_guide_editor[style_guide_id]
            style_guide.update_status("working", "updating_bible")
            
            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=style_guide_id,
                content=update_prompt,
                metadata={"task_id": bible_task_id, "task_type": "bible_update"}
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
                            "continuation_plan": response["output"]
                        }
                    )
                    
                    # Update writer state
                    new_state.writing_team[primary_writer_id].update_status("assigned", "continuation_assigned")
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
                            "edit_plan": response["output"]
                        }
                    )
                    
                    # Update writer state
                    new_state.writing_team[primary_writer_id].update_status("assigned", "edit_assigned")
    except Exception as e:
        error_msg = f"Error planning {'continuation' if is_continuation else 'edits'}: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
    assigned_writers = [w_id for w_id, w in new_state.writing_team.items()
                       if w.status in ["assigned", "working"] and 
                       ("continuation" in w.last_action if is_continuation else "edit" in w.last_action)]
    
    if not assigned_writers:
        # No assigned writers found
        return new_state
    
    writer_id = assigned_writers[0]
    writer = new_state.writing_team[writer_id]
    
    # Find the task message
    edit_messages = [msg for msg in new_state.messages 
                    if msg["recipient"] == writer_id 
                    and msg.get("metadata", {}).get("task_type") == task_type]
    
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
            metadata={"task_id": task_id, "task_type": task_type}
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
        response = writer_agent.invoke({
            "input": latest_task_message["content"],
            "agent_id": writer_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update writer state
            writer.update_status("completed", f"{task_type}_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=writer_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"content": response["output"]}
            )
            
            # Send to supervisor for review
            supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                                if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
            
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
                previous_feedback=""
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
                    "writer_id": writer_id
                }
            )
            
            # Update supervisor state
            supervisor = new_state.supervisors[supervisor_id]
            supervisor.update_status("working", f"reviewing_{task_type}")
            
            # Add message to communication log
            new_state.add_message(
                sender="system",
                recipient=supervisor_id,
                content=review_prompt,
                metadata={"task_id": review_task_id, "task_type": f"review_{task_type}"}
            )
    except Exception as e:
        error_msg = f"Error during {task_type}: {str(e)}"
        new_state.add_error("writer", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
    
    supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                        if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
    supervisor = new_state.supervisors[supervisor_id]
    
    # Find the review task message
    review_messages = [msg for msg in new_state.messages 
                      if msg["recipient"] == supervisor_id 
                      and msg.get("metadata", {}).get("task_type") == task_type]
    
    if not review_messages:
        # No review task found
        return new_state
    
    latest_review_message = review_messages[-1]
    task_id = latest_review_message.get("metadata", {}).get("task_id")
    
    # Create supervisor agent
    supervisor_agent = create_writing_supervisor_agent(supervisor_id)
    
    try:
        # Execute review task
        response = supervisor_agent.invoke({
            "input": latest_review_message["content"],
            "agent_id": supervisor_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", f"{task_type}_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": f"{task_type}_complete"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"review_output": response["output"]}
            )
            
            # Check if edits/continuation are approved
            is_approved = "approved" in response["output"].lower() or "accept" in response["output"].lower()
            
            if is_approved:
                # Update final content
                content_type = "write_continuation_complete" if is_continuation else "edit_content_complete"
                content_messages = [msg for msg in new_state.messages 
                                  if msg.get("metadata", {}).get("task_type") == content_type]
                
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
                        
                        review_type = "continuation_approval" if is_continuation else "edit_approval"
                        
                        # Request human review
                        from agents.tools import request_human_review
                        human_review_result = request_human_review(
                            story_id=new_state.story_id,
                            review_type=review_type,
                            content=updated_content[:2000] + "...",  # Preview
                            options=[
                                {"text": "Approve", "description": f"Approve the {'continuation' if is_continuation else 'edits'}"},
                                {"text": "Request Revisions", "description": "Request specific revisions"},
                                {"text": "Reject", "description": f"Reject the {'continuation' if is_continuation else 'edits'} entirely"}
                            ],
                            context=f"Supervisor's review:\n\n{response['output']}",
                            deadline=None
                        )
                        
                        if human_review_result.get("status") == "success":
                            review_id = human_review_result.get("review_id")
                            
                            # Update state to awaiting human input
                            new_state.request_human_input(
                                input_type=review_type,
                                data={
                                    "review_id": review_id,
                                    "content": updated_content,
                                    "supervisor_feedback": response["output"]
                                }
                            )
            else:
                # Edits/continuation need revision
                writer_id = None
                for msg in new_state.messages:
                    if msg.get("metadata", {}).get("task_type") in ["write_continuation_complete", "edit_content_complete"]:
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
                        data={
                            "feedback": response["output"]
                        }
                    )
                    
                    # Update writer state
                    writer = new_state.writing_team[writer_id]
                    writer.update_status("working", f"revising_{task_type.replace('review_', '')}")
                    
                    # Add message to communication log
                    new_state.add_message(
                        sender=supervisor_id,
                        recipient=writer_id,
                        content=revision_prompt,
                        metadata={"task_id": revision_task_id, "task_type": f"revise_{task_type.replace('review_', '')}"}
                    )
    except Exception as e:
        error_msg = f"Error during {task_type}: {str(e)}"
        new_state.add_error("writing_supervisor", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
            metadata={"review_id": review_id, "decision": "approve"}
        )
        
        # For edit mode, we might want to move to publishing
        if new_state.operation_mode == OperationMode.EDIT:
            start_publishing_phase(new_state)
    
    elif decision.lower() == "request revisions":
        # Human requested revisions
        writer_id = None
        for msg in new_state.messages:
            if msg.get("metadata", {}).get("task_type") in ["write_continuation_complete", "edit_content_complete"]:
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
                data={
                    "feedback": comments
                }
            )
            
            # Update writer state
            writer = new_state.writing_team[writer_id]
            writer.update_status("working", "human_revisions")
            
            # Add message to communication log
            new_state.add_message(
                sender="human",
                recipient=writer_id,
                content=revision_prompt,
                metadata={"task_id": revision_task_id, "task_type": "human_requested_revision"}
            )
            
            # Log feedback
            from agents.tools import add_feedback
            add_feedback(
                story_id=new_state.story_id,
                feedback_type=FeedbackType.CONTENT.value,
                content=comments,
                source="human",
                source_role="client",
                severity=4
            )
    
    elif decision.lower() == "reject":
        # Human rejected the edits/continuation
        writing_supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                                    if sup.agent_role == AgentRole.WRITING_SUPERVISOR)
        
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
            data={
                "feedback": comments
            }
        )
        
        # Update supervisor state
        supervisor = new_state.supervisors[writing_supervisor_id]
        supervisor.update_status("working", f"planning_new_{'continuation' if is_continuation else 'edit'}")
        
        # Add message to communication log
        new_state.add_message(
            sender="human",
            recipient=writing_supervisor_id,
            content=task_prompt,
            metadata={"task_id": task_id, "task_type": f"new_{'continuation' if is_continuation else 'edit'}_plan"}
        )
        
        # Log feedback
        from agents.tools import add_feedback
        add_feedback(
            story_id=new_state.story_id,
            feedback_type=FeedbackType.CONTENT.value,
            content=comments,
            source="human",
            source_role="client",
            severity=5
        )
    
    return new_state

def conduct_briefing(state: GraphState) -> GraphState:
    """Conduct initial briefing with the client using the author relations agent."""
    new_state = state.copy()
    
    # Get author relations agent and their current task
    author_relations_id = next(iter(new_state.author_relations.keys()))
    author_agent = new_state.author_relations[author_relations_id]
    
    # Find the briefing task message
    briefing_messages = [msg for msg in new_state.messages 
                        if msg["recipient"] == author_relations_id 
                        and msg.get("metadata", {}).get("task_type") == "briefing"]
    
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
            structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
            structure_description = new_state.structure_template.get("description", "")
            structure_info = f"\nStory Structure: {structure_name}\nDescription: {structure_description}\n"
        
        prompt = BRAINSTORM_SESSION_PROMPT.format(
            topic="Initial Story Briefing",
            story_request=user_request.to_prompt_string() if user_request else "No specific request details provided",
            current_status=f"Initial briefing phase\n{structure_info}",
            key_questions=(
                "1. What are the most important elements of this story to the client?\n"
                "2. Are there any specific research areas that would benefit the story?\n"
                "3. What tone and style is the client looking for?\n"
                "4. Any specific characters or plot elements that must be included?\n"
                "5. What would make this story particularly successful for the client?"
            ),
            previous_ideas=""
        )
        
        # Execute briefing task
        response = author_relations_agent.invoke({
            "input": prompt,
            "agent_id": author_relations_id,
            "story_id": story_id
        })
        
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
                metadata={
                    "task_id": task_id, 
                    "task_type": "briefing",
                    "session_id": session_id
                }
            )
            
            # Request human input
            new_state.request_human_input(
                input_type="briefing_session",
                data={
                    "session_id": session_id,
                    "agent_id": author_relations_id,
                    "current_message": response["output"],
                    "story_id": story_id
                }
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="awaiting_human_input"
            )
    except Exception as e:
        error_msg = f"Error during briefing: {str(e)}"
        new_state.add_error("author_relations", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
                    metadata={"session_id": session_id}
                )
                
                # Get the session history for context
                session_messages = [msg for msg in author_agent.session_history.get(session_id, [])
                                   if msg.get("role") in ["agent", "human"]]
                
                formatted_history = "\n\n".join([
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in session_messages
                ])
                
                # Include story structure information in the prompt
                structure_info = ""
                if new_state.story_structure and new_state.structure_template:
                    structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
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
                response = author_relations_agent.invoke({
                    "input": prompt,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "story_id": new_state.story_id
                })
                
                if "output" in response:
                    # Record agent response
                    author_agent.add_session_message(session_id, "agent", response["output"])
                    
                    # Add message to communication log
                    new_state.add_message(
                        sender=agent_id,
                        recipient="human",
                        content=response["output"],
                        metadata={"session_id": session_id}
                    )
                    
                    # Request further human input
                    new_state.request_human_input(
                        input_type="briefing_session",
                        data={
                            "session_id": session_id,
                            "agent_id": agent_id,
                            "current_message": response["output"],
                            "story_id": new_state.story_id
                        }
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
                metadata={"session_id": session_id, "session_ended": True}
            )
            
            # Create a summary of the briefing
            try:
                author_relations_agent = create_author_relations_agent(agent_id)
                
                # Get all session messages
                session_messages = author_agent.session_history.get(session_id, [])
                formatted_history = "\n\n".join([
                    f"{msg['role'].upper()}: {msg['content']}"
                    for msg in session_messages
                ])
                
                # Include story structure in the prompt
                structure_info = ""
                if new_state.story_structure and new_state.structure_template:
                    structure_name = new_state.structure_template.get("name", new_state.story_structure.value)
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
                
                summary_response = author_relations_agent.invoke({
                    "input": summary_prompt,
                    "agent_id": agent_id,
                    "story_id": new_state.story_id
                })
                
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
                        tags=["briefing", "requirements"]
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
    supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                        if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR)
    
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
        story_request=new_state.user_request.to_prompt_string() if new_state.user_request else "No specific request details provided",
        research_focus=(
            "1. Background information relevant to the story's setting and time period\n"
            "2. Subject matter expertise needed for authentic details\n"
            "3. Cultural, historical, or scientific context\n"
            "4. Similar works for inspiration and differentiation\n"
            "5. Target audience preferences and expectations\n"
            f"6. Research specific to each act in the {new_state.story_structure.value} structure"
        ),
        bible_entries=f"{briefing_summary}{structure_info}" or "No bible entries available yet.",
        existing_research="No prior research exists for this story."
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
            "story_structure": new_state.story_structure.value
        }
    )
    
    # Update researcher state
    researcher = new_state.research_team[researcher_id]
    researcher.update_status("working", "conducting_research")
    
    # Add message to communication log
    new_state.add_message(
        sender="system",
        recipient=researcher_id,
        content=task_prompt,
        metadata={"task_id": task_id, "task_type": "initial_research"}
    )
    
    return new_state

def conduct_research(state: GraphState) -> GraphState:
    """Conduct research using the researcher agent."""
    new_state = state.copy()
    
    # Get researcher and their task
    researcher_id = next(iter(new_state.research_team.keys()))
    researcher = new_state.research_team[researcher_id]
    
    # Find the research task message
    research_messages = [msg for msg in new_state.messages 
                        if msg["recipient"] == researcher_id 
                        and msg.get("metadata", {}).get("task_type") == "initial_research"]
    
    if not research_messages:
        # No research task found
        return new_state
    
    latest_research_message = research_messages[-1]
    task_id = latest_research_message.get("metadata", {}).get("task_id")
    
    # Create the researcher agent
    researcher_agent = create_researcher_agent(researcher_id)
    
    try:
        # Execute research task
        response = researcher_agent.invoke({
            "input": latest_research_message["content"],
            "agent_id": researcher_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update researcher state
            researcher.update_status("completed", "research_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=researcher_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "research_results"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"research_output": response["output"]}
            )
            
            # Send research to supervisor for review
            supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                                if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR)
            
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
                previous_feedback=""
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
                    "story_structure": new_state.story_structure.value
                }
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
                metadata={"task_id": review_task_id, "task_type": "research_review"}
            )
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        new_state.add_error("researcher", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
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
    supervisor_id = next(sup_id for sup_id, sup in new_state.supervisors.items() 
                        if sup.agent_role == AgentRole.RESEARCH_SUPERVISOR)
    supervisor = new_state.supervisors[supervisor_id]
    
    # Find the review task message
    review_messages = [msg for msg in new_state.messages 
                      if msg["recipient"] == supervisor_id 
                      and msg.get("metadata", {}).get("task_type") == "research_review"]
    
    if not review_messages:
        # No review task found
        return new_state
    
    latest_review_message = review_messages[-1]
    task_id = latest_review_message.get("metadata", {}).get("task_id")
    
    # Create the supervisor agent
    supervisor_agent = create_research_supervisor_agent(supervisor_id)
    
    try:
        # Execute review task
        response = supervisor_agent.invoke({
            "input": latest_review_message["content"],
            "agent_id": supervisor_id,
            "story_id": new_state.story_id
        })
        
        if "output" in response:
            # Update supervisor state
            supervisor.update_status("completed", "review_completed")
            
            # Add message to communication log
            new_state.add_message(
                sender=supervisor_id,
                recipient="system",
                content=response["output"],
                metadata={"task_id": task_id, "task_type": "research_review_results"}
            )
            
            # Update task status
            new_state.update_task_status(
                task_id=task_id,
                status="completed",
                result={"review_output": response["output"]}
            )
            
            # Determine if research is approved or needs revision
            is_approved = "approved" in response["output"].lower() or "satisfactory" in response["output"].lower()
            
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
                        "story_structure": new_state.story_structure.value
                    }
                )
                
                # Update researcher state
                researcher = new_state.research_team[researcher_id]
                researcher.update_status("working", "revising_research")
                
                # Add message to communication log
                new_state.add_message(
                    sender=supervisor_id,
                    recipient=researcher_id,
                    content=revision_prompt,
                    metadata={"task_id": revision_task_id, "task_type": "research_revision"}
                )
                
                # Log feedback
                from agents.tools import add_feedback
                add_feedback(
                    story_id=new_state.story_id,
                    feedback_type=FeedbackType.RESEARCH.value,
                    content=response["output"],
                    source=supervisor_id,
                    source_role=AgentRole.RESEARCH_SUPERVISOR.value,
                    severity=3
                )
    except Exception as e:
        error_msg = f"Error during research review: {str(e)}"
        new_state.add_error("research_supervisor", error_msg)
        
        # Update task status
        if task_id:
            new_state.update_task_status(
                task_id=task_id,
                status="error",
                result={"error": error_msg}
            )
        
        supervisor.update_status("
