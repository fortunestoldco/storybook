import datetime
import uuid
import json
import re
import requests
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import storybook.config
from storybook.config import StoryStructure, STORY_STRUCTURES

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    unique_id = uuid.uuid4().hex[:10]
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id

def format_timestamp(dt: Optional[datetime.datetime] = None) -> str:
    """Format a datetime object as ISO string."""
    if dt is None:
        dt = datetime.datetime.now()
    return dt.isoformat()

def current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return format_timestamp()

def parse_timestamp(timestamp_str: str) -> Optional[datetime.datetime]:
    """Parse an ISO format timestamp string into a datetime object."""
    try:
        return datetime.datetime.fromisoformat(timestamp_str)
    except (ValueError, TypeError):
        return None

def calculate_time_difference(start_time: str, end_time: Optional[str] = None) -> int:
    """Calculate time difference in seconds between two timestamps."""
    start_dt = parse_timestamp(start_time)
    if end_time:
        end_dt = parse_timestamp(end_time)
    else:
        end_dt = datetime.datetime.now()
        
    if not start_dt or not end_dt:
        return 0
        
    difference = end_dt - start_dt
    return int(difference.total_seconds())

def is_deadline_passed(deadline: str) -> bool:
    """Check if a deadline has passed."""
    deadline_dt = parse_timestamp(deadline)
    if not deadline_dt:
        return False
    return datetime.datetime.now() > deadline_dt

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from text."""
    try:
        # Look for JSON-like structures in the text
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*?})'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            for group in match:
                if group.strip():
                    parsed = json.loads(group)
                    return parsed
        
        # If no match found, try to parse the entire text
        return json.loads(text)
    except json.JSONDecodeError:
        # Return empty dict if no valid JSON found
        return {}

def format_message_history(messages: List[Dict[str, Any]], limit: int = None) -> str:
    """Format a list of messages into a readable string."""
    if limit:
        messages = messages[-limit:]
    
    formatted = []
    for msg in messages:
        sender = msg.get("sender", "Unknown")
        recipient = msg.get("recipient", "Unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        formatted.append(f"FROM: {sender} TO: {recipient} [{timestamp}]\n{content}\n")
    
    return "\n".join(formatted)

def clean_and_format_text(text: str) -> str:
    """Clean and format text for output."""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove markdown code block syntax
    text = re.sub(r'```(?:markdown|md|json|)\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    
    return text.strip()

def create_task_id(task_type: str, agent_id: str) -> str:
    """Create a unique task ID."""
    return f"{task_type}_{agent_id}_{uuid.uuid4().hex[:6]}"

def parse_feedback(feedback_text: str) -> Dict[str, Any]:
    """Parse feedback text to extract structured information."""
    result = {
        "strengths": [],
        "weaknesses": [],
        "suggestions": [],
        "rating": None
    }
    
    # Try to extract a numerical rating
    rating_match = re.search(r'rating[:\s]+(\d+(?:\.\d+)?)', feedback_text, re.IGNORECASE)
    if rating_match:
        try:
            result["rating"] = float(rating_match.group(1))
        except ValueError:
            pass
    
    # Extract strengths
    strengths_match = re.search(r'strengths?[:\s]+(.*?)(?=weaknesses?|improvements?|suggestions?|$)', 
                               feedback_text, re.IGNORECASE | re.DOTALL)
    if strengths_match:
        strengths_text = strengths_match.group(1).strip()
        # Extract bullet points or numbered lists
        strengths = re.findall(r'(?:^|\n)\s*[-*•]|\d+\.\s*([^\n]+)', strengths_text)
        if strengths:
            result["strengths"] = [s.strip() for s in strengths if s.strip()]
        elif strengths_text:
            # If no bullet points, add the whole text
            result["strengths"] = [strengths_text]
    
    # Extract weaknesses/improvements
    weaknesses_match = re.search(r'(?:weaknesses?|improvements?)[:\s]+(.*?)(?=suggestions?|strengths?|$)', 
                                feedback_text, re.IGNORECASE | re.DOTALL)
    if weaknesses_match:
        weaknesses_text = weaknesses_match.group(1).strip()
        weaknesses = re.findall(r'(?:^|\n)\s*[-*•]|\d+\.\s*([^\n]+)', weaknesses_text)
        if weaknesses:
            result["weaknesses"] = [w.strip() for w in weaknesses if w.strip()]
        elif weaknesses_text:
            result["weaknesses"] = [weaknesses_text]
    
    # Extract suggestions
    suggestions_match = re.search(r'suggestions?[:\s]+(.*?)(?=strengths?|weaknesses?|improvements?|$)', 
                                 feedback_text, re.IGNORECASE | re.DOTALL)
    if suggestions_match:
        suggestions_text = suggestions_match.group(1).strip()
        suggestions = re.findall(r'(?:^|\n)\s*[-*•]|\d+\.\s*([^\n]+)', suggestions_text)
        if suggestions:
            result["suggestions"] = [s.strip() for s in suggestions if s.strip()]
        elif suggestions_text:
            result["suggestions"] = [suggestions_text]
    
    return result

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix

def format_agent_response(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format an agent response with content and metadata."""
    if metadata is None:
        metadata = {}
    
    return {
        "content": content,
        "metadata": metadata,
        "timestamp": current_timestamp()
    }

def validate_story_structure(story: Dict[str, Any]) -> List[str]:
    """Validate story structure and return list of issues if any."""
    issues = []
    
    # Check required fields
    required_fields = ["title", "content"]
    for field in required_fields:
        if field not in story or not story[field]:
            issues.append(f"Missing required field: {field}")
    
    # Check content length
    if "content" in story and len(story["content"]) < 100:
        issues.append("Content is too short (less than 100 characters)")
    
    # Check outline if present
    if "outline" in story and story["outline"]:
        outline = story["outline"]
        if "plot_points" in outline and len(outline["plot_points"]) < 3:
            issues.append("Outline should have at least 3 plot points")
        
        if "characters" in outline and len(outline["characters"]) < 1:
            issues.append("Outline should have at least 1 character")
    
    return issues

def prepare_human_review_prompt(review_type: str, content: str, options: List[Dict[str, Any]], context: Optional[str] = None) -> str:
    """Prepare a prompt for human review."""
    prompt_parts = [
        f"# Human Review Required: {review_type}",
        "",
        "## Content to Review",
        content,
        ""
    ]
    
    if context:
        prompt_parts.extend([
            "## Context",
            context,
            ""
        ])
    
    if options:
        prompt_parts.append("## Options")
        for i, option in enumerate(options, 1):
            option_text = option.get("text", "")
            option_description = option.get("description", "")
            prompt_parts.append(f"{i}. {option_text}")
            if option_description:
                prompt_parts.append(f"   {option_description}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        "## Instructions",
        "Please review the content above and provide your feedback or decision.",
        "Your input is crucial for the quality of the final story."
    ])
    
    return "\n".join(prompt_parts)

def format_brainstorm_session(session: Dict[str, Any]) -> str:
    """Format a brainstorming session into readable text."""
    formatted = [
        f"# Brainstorming Session: {session.get('topic', 'Untitled')}",
        "",
        f"**Description:** {session.get('description', 'No description')}",
        f"**Status:** {session.get('status', 'Unknown')}",
        f"**Created:** {session.get('created_at', 'Unknown')}",
        "",
        "## Messages"
    ]
    
    messages = session.get("messages", [])
    for msg in messages:
        sender = msg.get("sender", "Unknown")
        role = msg.get("role", "Unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        formatted.append(f"**{role.upper()} ({sender})** - {timestamp}")
        formatted.append(content)
        formatted.append("")
    
    if session.get("status") == "completed" and "summary" in session:
        formatted.extend([
            "## Summary",
            session["summary"],
            "",
            f"Session ended at: {session.get('ended_at', 'Unknown')}"
        ])
    
    return "\n".join(formatted)

def get_story_structure_template(structure: StoryStructure) -> Dict[str, Any]:
    """Get the template for a story structure."""
    if structure in STORY_STRUCTURES:
        return STORY_STRUCTURES[structure]
    return STORY_STRUCTURES[StoryStructure.THREE_ACT]  # Default to three-act if not found

def create_model_instance(model_type: str = None, use_local: bool = None):
    """Create an appropriate language model instance based on configuration."""
    from langchain_openai import ChatOpenAI
    
    # Override parameters if provided
    if use_local is None:
        use_local = config.USE_OLLAMA
    
    if not model_type:
        model_type = "default"
    
    # Set temperature based on role
    temperature_map = {
        "default": 0.7,
        "research": 0.3,
        "writing": 0.7,
        "publishing": 0.4,
        "supervisor": 0.2,
        "author_relations": 0.6
    }
    temperature = temperature_map.get(model_type, 0.7)
    
    if use_local:
        # Use Ollama for local models
        from langchain_community.chat_models import ChatOllama
        
        # Select appropriate Ollama model based on role
        model_name_map = {
            "default": config.OLLAMA_DEFAULT_MODEL,
            "research": config.OLLAMA_RESEARCH_MODEL,
            "writing": config.OLLAMA_WRITING_MODEL,
            "publishing": config.OLLAMA_PUBLISHING_MODEL,
            "supervisor": config.OLLAMA_SUPERVISOR_MODEL
        }
        
        model_name = model_name_map.get(model_type, config.OLLAMA_DEFAULT_MODEL)
        
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=config.OLLAMA_BASE_URL
        )
    else:
        # Use OpenAI models
        model_name_map = {
            "default": config.DEFAULT_MODEL,
            "research": config.RESEARCH_MODEL,
            "writing": config.WRITING_MODEL,
            "publishing": config.PUBLISHING_MODEL,
            "supervisor": config.SUPERVISOR_MODEL,
            "author_relations": config.AUTHOR_RELATIONS_MODEL
        }
        
        model_name = model_name_map.get(model_type, config.DEFAULT_MODEL)
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=config.OPENAI_API_KEY
        )

def create_section_structure_from_template(story_structure: StoryStructure) -> List[Dict[str, Any]]:
    """Create a list of sections based on the selected story structure template."""
    template = get_story_structure_template(story_structure)
    sections = []
    
    for act_index, act in enumerate(template.get("acts", [])):
        act_name = act.get("name", f"Act {act_index + 1}")
        
        for comp_index, component in enumerate(act.get("components", [])):
            comp_name = component.get("name", f"Section {act_index}.{comp_index}")
            comp_desc = component.get("description", "")
            
            sections.append({
                "id": f"section_{act_index}_{comp_index}",
                "title": comp_name,
                "description": comp_desc,
                "act": act_name,
                "sequence": (act_index * 100) + comp_index
            })
    
    return sections

def estimate_section_complexity(section_description: str) -> float:
    """Estimate the complexity of a section based on its description."""
    # This is a simplified approach - in a real implementation,
    # you might use more sophisticated analysis
    
    # Count complexity indicators in the description
    complexity_terms = [
        "complex", "intricate", "detailed", "nuanced", "layered",
        "conflict", "multiple", "interweaving", "challenging", 
        "emotional", "deep", "philosophical", "technical"
    ]
    
    count = sum(1 for term in complexity_terms if term.lower() in section_description.lower())
    
    # Calculate a score from 0.0 to 1.0
    base_score = 0.5  # Start at medium complexity
    modifier = count * 0.1  # Each complexity term adds 0.1
    
    # Cap at 1.0
    return min(1.0, base_score + modifier)

def distribute_sections_to_writers(
    sections: List[Dict[str, Any]], 
    writer_ids: List[str],
    use_joint_llm: bool = False,
    joint_llm_threshold: float = 0.7
) -> Dict[str, List[Dict[str, Any]]]:
    """Distribute sections among writers based on complexity and sequence."""
    if not writer_ids:
        return {}
    
    # If only one writer, assign all sections
    if len(writer_ids) == 1:
        return {writer_ids[0]: sections}
        
    # Sort sections by sequence
    sorted_sections = sorted(sections, key=lambda s: s.get("sequence", 0))
    
    # Analyze complexity
    for section in sorted_sections:
        section["complexity"] = estimate_section_complexity(section.get("description", ""))
    
    # Determine which sections should use joint LLM
    joint_sections = []
    regular_sections = []
    
    if use_joint_llm:
        for section in sorted_sections:
            if section.get("complexity", 0) >= joint_llm_threshold:
                joint_sections.append(section)
            else:
                regular_sections.append(section)
    else:
        regular_sections = sorted_sections
    
    # Create a distribution dictionary
    distribution = {writer_id: [] for writer_id in writer_ids}
    
    # Add a special "joint" writer if needed
    if joint_sections and use_joint_llm:
        distribution["joint_writer"] = joint_sections
    
    # Distribute regular sections among writers
    for i, section in enumerate(regular_sections):
        writer_id = writer_ids[i % len(writer_ids)]
        distribution[writer_id].append(section)
    
    return distribution

def word_count(text: str) -> int:
    """Count the number of words in a text."""
    return len(re.findall(r'\b\w+\b', text))

def section_exceeds_complexity_threshold(section_text: str, threshold: int = None) -> bool:
    """Determine if a section exceeds the complexity threshold for joint LLM."""
    if threshold is None:
        threshold = config.JOINT_LLM_THRESHOLD
        
    # Simple approach: check word count
    return word_count(section_text) > threshold
