from storybook.tools.research.states import DomainResearchState
from storybook.tools.research.graphs import create_research_subgraph
from .base import ResearchAgent

class DomainKnowledgeSpecialist(ResearchAgent):
    """Specialist in domain-specific research."""
    
    def __init__(self, config):
        super().__init__(
            name="Domain Knowledge Specialist",
            research_graph=create_research_subgraph(),
            state_class=DomainResearchState,
            config=config
        )