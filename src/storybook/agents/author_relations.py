"""Author Relations agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from config import AgentRole, AUTHOR_RELATIONS_MODEL, USE_OLLAMA, StoryStructure, STORY_STRUCTURES
from agents.tools import AUTHOR_RELATIONS_TOOLS
from agents.prompts import AUTHOR_RELATIONS_SYSTEM_PROMPT, BRAINSTORM_SESSION_PROMPT
from agents.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_author_relations_agent(agent_id: str) -> AgentExecutor:
    """Create an author relations agent with appropriate tools."""
    # Author relations agents generally need higher quality, so if USE_OLLAMA is true,
    # we might still want to consider using the API model
    use_local = USE_OLLAMA
    if USE_OLLAMA:
        # This is a client-facing role, so we might want to use the API model regardless
        use_local = False  
        
    llm = create_model_instance("author_relations", use_local)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=AUTHOR_RELATIONS_TOOLS,
        system_message=AUTHOR_RELATIONS_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=AUTHOR_RELATIONS_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8
    )
    
    return agent

def create_brainstorm_prompt(
    topic: str,
    story_request: str,
    current_status: str = "",
    key_questions: str = "",
    previous_ideas: str = "",
    story_structure_options: bool = True
) -> str:
    """Create a formatted brainstorming session prompt."""
    # Include story structure options if requested
    structure_options = ""
    if story_structure_options:
        structure_options = "## Story Structure Options\n\n"
        for structure in StoryStructure:
            if structure.value in STORY_STRUCTURES:
                structure_data = STORY_STRUCTURES[structure.value]
                structure_options += f"### {structure_data['name']}\n{structure_data['description']}\n\n"
    
    return BRAINSTORM_SESSION_PROMPT.format(
        topic=topic,
        story_request=story_request,
        story_structure_options=structure_options,
        current_status=current_status or "Initial brainstorming phase",
        key_questions=key_questions or "What are the key elements the client wants in this story?",
        previous_ideas=previous_ideas or "No previous ideas recorded."
    )

async def execute_briefing_session(
    agent_id: str,
    story_id: str,
    user_request: str,
    session_id: str = None,
    previous_messages: List[Dict[str, Any]] = None,
    include_structure_options: bool = True
) -> Dict[str, Any]:
    """Execute a briefing session with the author relations agent."""
    try:
        # Create the session prompt
        if not previous_messages:
            # Initial briefing prompt
            prompt = create_brainstorm_prompt(
                topic="Initial Story Briefing",
                story_request=user_request,
                key_questions=(
                    "1. What are the most important elements of this story to the client?\n"
                    "2. Are there any specific research areas that would benefit the story?\n"
                    "3. What tone and style is the client looking for?\n"
                    "4. Which story structure would best suit this narrative?\n"
                    "5. Any specific characters or plot elements that must be included?\n"
                    "6. What would make this story particularly successful for the client?"
                ),
                story_structure_options=include_structure_options
            )
        else:
            # Continuation of existing briefing
            formatted_history = "\n\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in previous_messages
            ])
            
            prompt = f"""
            # Continuing Briefing Session
            
            ## Session History
            {formatted_history}
            
            Continue the briefing session based on the client's response above. 
            Focus on gathering all necessary information to create a detailed story plan.
            Be sure to discuss story structure options and help the client make an informed choice.
            """
        
        # Get the agent
        agent = get_author_relations_agent(agent_id)
        
        # Execute the briefing
        response = await agent.ainvoke({
            "input": prompt,
            "agent_id": agent_id,
            "story_id": story_id,
            "session_id": session_id
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "briefing_session",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "session_id": session_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from author relations agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error during briefing session: {str(e)}",
            metadata={"error": True}
        )

async def create_briefing_summary(
    agent_id: str,
    story_id: str,
    session_messages: List[Dict[str, Any]],
    story_structure: str = None
) -> Dict[str, Any]:
    """Create a summary of a completed briefing session."""
    try:
        # Format the session history
        formatted_history = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in session_messages
        ])
        
        # Include story structure in the prompt if available
        structure_info = ""
        if story_structure:
            structure_name = STORY_STRUCTURES.get(story_structure, {}).get("name", story_structure)
            structure_info = f"\n\n## Selected Story Structure\n{structure_name}"
        
        summary_prompt = f"""
        # Briefing Session Summary
        
        {structure_info}
        
        ## Complete Briefing Conversation
        {formatted_history}
        
        Please create a structured summary of this briefing session that:
        1. Identifies the core requirements for the story
        2. Lists key elements that must be included
        3. Notes specific style, tone, and audience preferences
        4. Suggests priority research areas
        5. Captures any special requests or constraints
        6. Notes the selected story structure and its implications
        
        This summary will be used to guide the story creation process.
        """
        
        # Get the agent
        agent = get_author_relations_agent(agent_id)
        
        # Execute the summary creation
        response = await agent.ainvoke({
            "input": summary_prompt,
            "agent_id": agent_id,
            "story_id": story_id
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "briefing_summary",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from author relations agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error creating briefing summary: {str(e)}",
            metadata={"error": True}
        )
considerations=considerations or "Consider the overall quality, alignment with requirements, and potential for success with the target audience.",
        decision_requested=decision_requested or "whether to approve, request revisions, or reject",
        reason_for_human_review=reason_for_human_review or "human judgment is essential for evaluating creative quality and alignment with expectations"
    )

def prepare_review_request(
    review_type: str,
    content: str,
    story_id: str,
    options: List[Dict[str, Any]] = None,
    context: str = None,
    story_structure: str = None
) -> Dict[str, Any]:
    """Prepare a human review request."""
    if options is None:
        options = [
            {"text": "Approve", "description": "Approve and continue to the next phase"},
            {"text": "Request Revisions", "description": "Request specific revisions before proceeding"},
            {"text": "Reject", "description": "Reject and restart this phase"}
        ]
    
    # Create a formatted review prompt
    review_prompt = create_human_review_prompt(
        review_type=review_type,
        item_description=content[:2000] + "..." if len(content) > 2000 else content,
        options=options,
        story_structure=story_structure or "",
        context=context or "",
        considerations=f"This is a {review_type} review. Please evaluate the content carefully.",
        decision_requested=f"the {review_type}",
        reason_for_human_review=f"your judgment is needed to ensure the {review_type} meets expectations and requirements"
    )
    
    # Create the review request
    review_id = f"review_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return {
        "id": review_id,
        "type": review_type,
        "content": review_prompt,
        "options": options,
        "story_id": story_id,
        "created_at": datetime.datetime.now().isoformat(),
        "status": "pending"
    }

def process_human_feedback(
    review_id: str,
    decision: str,
    comments: str,
    original_request: Dict[str, Any]
) -> Dict[str, Any]:
    """Process human feedback from a review."""
    try:
        # Create a response object
        response = {
            "review_id": review_id,
            "decision": decision,
            "comments": comments,
            "original_request": original_request,
            "processed_at": datetime.datetime.now().isoformat()
        }
        
        return format_agent_response(
            content=f"Human feedback processed: {decision}",
            metadata={
                "task_type": "human_feedback",
                "review_id": review_id,
                "decision": decision,
                "has_comments": bool(comments),
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
    except Exception as e:
        return format_agent_response(
            content=f"Error processing human feedback: {str(e)}",
            metadata={"error": True}
        )
