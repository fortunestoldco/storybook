# -*- coding: utf-8 -*-
import json
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import QUALITY_GATES


def generate_id() -> str:
    """Generate a unique ID.
    
    Returns:
        A unique ID string.
    """
    return str(uuid.uuid4())


def current_timestamp() -> str:
    """Get the current timestamp as a string.
    
    Returns:
        Current timestamp as a string.
    """
    return datetime.now().isoformat()


def check_quality_gate(phase_transition: str, metrics: Dict) -> Dict:
    """Check if a quality gate is passed.
    
    Args:
        phase_transition: The phase transition to check (e.g., "initialization_to_development").
        metrics: The current quality metrics.
        
    Returns:
        Dict with pass/fail status and reasons.
    """
    if phase_transition not in QUALITY_GATES:
        return {
            "passed": False,
            "reason": f"Unknown phase transition: {phase_transition}"
        }
    
    gate = QUALITY_GATES[phase_transition]
    passed = True
    failed_criteria = []
    
    for criterion, threshold in gate.items():
        if criterion >= "human_approval_required":
            continue
        
        if criterion not in metrics or metrics[criterion] < threshold:
            passed = False
            failed_criteria.append({
                "criterion": criterion,
                "threshold": threshold,
                "actual": metrics.get(criterion, "Not measured")
            })
    
    # Check if human approval is required
    if gate.get("human_approval_required") and not metrics.get("human_approved", False):
        passed = False
        failed_criteria.append({
            "criterion": "human_approval",
            "threshold": True,
            "actual": False
        })
    
    return {
        "passed": passed,
        "failed_criteria": failed_criteria if not passed else []
    }


def format_agent_response(response: str) -> Dict:
    """Format an agent's response into a structured format.
    
    Args:
        response: The raw response from an agent.
        
    Returns:
        A structured response dict.
    """
    # Extract JSON if the response contains it
    json_match = re.search(r'```json\\s*(.*?)\\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If no JSON or invalid JSON, return a basic structure
    return {
        "content": response,
        "timestamp": current_timestamp(),
    }


def create_prompt_with_context(template: str, context: Dict) -> str:
    """Create a prompt by filling a template with context.
    
    Args:
        template: The prompt template string.
        context: The context to fill the template with.
        
    Returns:
        The filled prompt string.
    """
    return template.format(**context)


def read_file(filepath: str, encoding: str = 'utf-8') -> str:
    """Read a file with proper encoding.
    
    Args:
        filepath: Path to the file.
        encoding: Encoding to use, defaults to UTF-8.
        
    Returns:
        The file contents as a string.
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # If UTF-8 fails, try with a different encoding
        with open(filepath, 'r', encoding='cp1252') as f:
            return f.read()

