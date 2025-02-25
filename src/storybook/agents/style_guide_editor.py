"""Style Guide Editor agent implementation for the Storybook application."""

from typing import Dict, List, Any, Optional
import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, SystemMessage

from storybook.config import AgentRole, WRITING_MODEL, USE_OLLAMA, OLLAMA_WRITING_MODEL, BibleSectionType
from storybook.tools import BIBLE_EDITOR_TOOLS
from storybook.prompts import STYLE_GUIDE_EDITOR_SYSTEM_PROMPT, BIBLE_UPDATE_PROMPT
from storybook.utils import create_model_instance, extract_json_from_text, format_agent_response

def get_style_guide_editor_agent(agent_id: str) -> AgentExecutor:
    """Create a style guide editor agent with appropriate tools."""
    llm = create_model_instance("writing", USE_OLLAMA)
    
    prompt = create_openai_tools_agent(
        llm=llm,
        tools=BIBLE_EDITOR_TOOLS,
        system_message=STYLE_GUIDE_EDITOR_SYSTEM_PROMPT
    )
    
    agent = AgentExecutor(
        agent=prompt,
        tools=BIBLE_EDITOR_TOOLS,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent

def create_bible_update_task(
    task_description: str,
    section_type: str,
    section_title: str,
    new_information: str,
    story_structure: str = "",
    current_content: str = "",
    related_sections: str = ""
) -> str:
    """Create a formatted bible update task prompt."""
    return BIBLE_UPDATE_PROMPT.format(
        task_description=task_description,
        story_structure=story_structure,
        section_type=section_type,
        section_title=section_title,
        current_content=current_content or "No existing content for this section.",
        new_information=new_information,
        related_sections=related_sections or "No related sections available."
    )

async def execute_bible_update(
    agent_id: str,
    story_id: str,
    section_type: str,
    section_title: str,
    new_information: str,
    task_description: str = "Update or create a section in the story bible based on new information.",
    story_structure: str = "",
    current_content: str = "",
    related_sections: str = ""
) -> Dict[str, Any]:
    """Execute a bible update task with the style guide editor agent."""
    try:
        # Validate the section type
        try:
            section_type_enum = BibleSectionType(section_type)
        except ValueError:
            section_type = BibleSectionType.REFERENCE_MATERIAL.value
        
        # Create the task prompt
        task_prompt = create_bible_update_task(
            task_description=task_description,
            section_type=section_type,
            section_title=section_title,
            new_information=new_information,
            story_structure=story_structure,
            current_content=current_content,
            related_sections=related_sections
        )
        
        # Get the agent
        agent = get_style_guide_editor_agent(agent_id)
        
        # Execute the task
        response = await agent.ainvoke({
            "input": task_prompt,
            "agent_id": agent_id,
            "story_id": story_id,
            "section_type": section_type,
            "section_title": section_title
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "bible_update",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "section_type": section_type,
                    "section_title": section_title,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from style guide editor agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error updating bible: {str(e)}",
            metadata={"error": True}
        )

async def analyze_imported_content(
    agent_id: str,
    story_id: str,
    imported_content: str,
    user_requirements: str = ""
) -> Dict[str, Any]:
    """Analyze imported content to create initial bible entries."""
    try:
        # Create the analysis prompt
        analysis_prompt = f"""
        # Content Import Analysis
        
        You are tasked with analyzing imported content to create the foundation for the story bible.
        
        ## Imported Content Preview
        {imported_content[:5000]}...
        
        ## User Requirements
        {user_requirements}
        
        Please analyze this content and identify:
        1. Main characters and their traits
        2. Settings and world-building elements
        3. Plot structure and key events
        4. Themes and motifs
        5. Style and tone characteristics
        6. Any inconsistencies or areas that need development
        
        Your analysis will be used to create the initial story bible entries.
        """
        
        # Get the agent
        agent = get_style_guide_editor_agent(agent_id)
        
        # Execute the analysis
        response = await agent.ainvoke({
            "input": analysis_prompt,
            "agent_id": agent_id,
            "story_id": story_id
        })
        
        if "output" in response:
            return format_agent_response(
                content=response["output"],
                metadata={
                    "task_type": "import_analysis",
                    "agent_id": agent_id,
                    "story_id": story_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        else:
            return format_agent_response(
                content="Error: No output from style guide editor agent",
                metadata={"error": True}
            )
    except Exception as e:
        return format_agent_response(
            content=f"Error analyzing imported content: {str(e)}",
            metadata={"error": True}
        )
