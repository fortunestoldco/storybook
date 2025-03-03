from typing import Dict, Any
from datetime import datetime

def collect_agent_metrics(agent_output: Dict[str, Any]) -> Dict[str, float]:
    """Collect metrics from agent output."""
    base_metrics = {
        "completion": 0.0,
        "quality": 0.0,
        "coherence": 0.0,
        "creativity": 0.0
    }
    
    if "metrics" in agent_output:
        base_metrics.update(agent_output["metrics"])
    
    if "scores" in agent_output:
        base_metrics.update(agent_output["scores"])
        
    return base_metrics

def calculate_phase_metrics(phase_results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate aggregated metrics for a workflow phase."""
    metrics = {}
    agent_metrics = [
        collect_agent_metrics(result) 
        for result in phase_results.get("results", {}).values()
    ]
    
    if agent_metrics:
        # Aggregate metrics across agents
        for key in agent_metrics[0].keys():
            metrics[key] = sum(m.get(key, 0.0) for m in agent_metrics) / len(agent_metrics)
            
    metrics["timestamp"] = datetime.now().isoformat()
    return metrics

def meets_quality_gates(metrics: Dict[str, float], gates: Dict[str, float]) -> bool:
    """Check if metrics meet quality gates requirements."""
    return all(
        metrics.get(metric, 0) >= threshold
        for metric, threshold in gates.items()
        if metric != "human_approval_required"
    )
