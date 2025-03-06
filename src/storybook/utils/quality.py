from typing import Dict, Any
from datetime import datetime

def check_quality_gate(gate_name: str, metrics: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a project meets quality gate requirements to move to the next phase."""
    # ...existing code from utils.py...