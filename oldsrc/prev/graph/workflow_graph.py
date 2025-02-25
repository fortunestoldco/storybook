from typing import Dict, Any
from langgraph.graph import StateGraph, Graph
from langgraph.prebuilt.tool_node import ToolNode

class NovelWritingGraph:
    def __init__(self):
        self.graph = StateGraph()
        self.setup_nodes()
        self.setup_edges()

    def setup_nodes(self):
        """Set up all nodes in the graph."""
        # Supervisor Team
        self.graph.add_node("overall_supervisor", self._overall_supervisor_node())
        
        # Author Relations Team
        self.graph.add_node("author_relations", self._author_relations_node())
        
        # Research Team
        self.graph.add_node("research_supervisor", self._research_supervisor_node())
        self.graph.add_node("contextual_researcher", self._contextual_researcher_node())
        self.graph.add_node("market_researcher", self._market_researcher_node())
        self.graph.add_node("consumer_insights", self._consumer_insights_node())
        
        # Creative Writing Team
        self.graph.add_node("creative_supervisor", self._creative_supervisor_node())
        self.graph.add_node("world_builder", self._world_builder_node())
        self.graph.add_node("character_builder", self._character_builder_node())
        self.graph.add_node("story_writer", self._story_writer_node())
        self.graph.add_node("dialogue_writer", self._dialogue_writer_node())
        
        # Publishing Team
        self.graph.add_node("publishing_supervisor", self._publishing_supervisor_node())
        self.graph.add_node("consistency_checker", self._consistency_checker_node())
        self.graph.add_node("continuity_checker", self._continuity_checker_node())
        self.graph.add_node("editor", self._editor_node())
        self.graph.add_node("finalisation", self._finalisation_node())
        self.graph.add_node("conflict_resolution", self._conflict_resolution_node())
        self.graph.add_node("narrative_structure", self._narrative_structure_node())
        self.graph.add_node("thematic_analysis", self._thematic_analysis_node())

        # Importation Agent
        self.graph.add_node("importation_agent", self._importation_agent_node())

        # Multi-Writer Review
        self.graph.add_node("multi_writer_review", self._multi_writer_review_node())

    def setup_edges(self):
        """Set up all edges in the graph."""
        # Overall Supervisor connections
        self.graph.add_edge("overall_supervisor", "author_relations")
        self.graph.add_edge("overall_supervisor", "research_supervisor")
        self.graph.add_edge("overall_supervisor", "creative_supervisor")
        self.graph.add_edge("overall_supervisor", "publishing_supervisor")

        # Research Team connections
        self.graph.add_edge("research_supervisor", "contextual_researcher")
        self.graph.add_edge("research_supervisor", "market_researcher")
        self.graph.add_edge("research_supervisor", "consumer_insights")

        # Creative Team connections
        self.graph.add_edge("creative_supervisor", "world_builder")
        self.graph.add_edge("creative_supervisor", "character_builder")
        self.graph.add_edge("creative_supervisor", "story_writer")
        self.graph.add_edge("creative_supervisor", "dialogue_writer")

        # Publishing Team connections
        self.graph.add_edge("publishing_supervisor", "consistency_checker")
        self.graph.add_edge("publishing_supervisor", "continuity_checker")
        self.graph.add_edge("publishing_supervisor", "editor")
        self.graph.add_edge("publishing_supervisor", "finalisation")
        self.graph.add_edge("editor", "conflict_resolution")
        self.graph.add_edge("editor", "narrative_structure")
        self.graph.add_edge("editor", "thematic_analysis")
        self.graph.add_edge("conflict_resolution", "editor")
        self.graph.add_edge("narrative_structure", "editor")
        self.graph.add_edge("thematic_analysis", "editor")

        # Importation Agent connections
        self.graph.add_edge("author_relations", "importation_agent", condition=self._create_conditional_edge("mode == 'import'"))
        self.graph.add_edge("importation_agent", "publishing_supervisor", condition=self._create_conditional_edge("mode == 'import'"))

        # Multi-Writer Review connections
        self.graph.add_edge("publishing_supervisor", "multi_writer_review", condition=self._create_conditional_edge("multi_writer_mode"))
        self.graph.add_edge("multi_writer_review", "publishing_supervisor")

    def _create_conditional_edge(self, condition: str):
        """Create a conditional edge based on the state."""
        def edge_fn(state):
            return state[condition]
        return edge_fn

    def get_graph(self) -> Graph:
        """Return the configured graph."""
        return self.graph.compile()

    # Node definitions
    def _overall_supervisor_node(self):
        return ToolNode(name="overall_supervisor")

    def _author_relations_node(self):
        return ToolNode(name="author_relations")

    def _research_supervisor_node(self):
        return ToolNode(name="research_supervisor")

    def _contextual_researcher_node(self):
        return ToolNode(name="contextual_researcher")

    def _market_researcher_node(self):
        return ToolNode(name="market_researcher")

    def _consumer_insights_node(self):
        return ToolNode(name="consumer_insights")

    def _creative_supervisor_node(self):
        return ToolNode(name="creative_supervisor")

    def _world_builder_node(self):
        return ToolNode(name="world_builder")

    def _character_builder_node(self):
        return ToolNode(name="character_builder")

    def _story_writer_node(self):
        return ToolNode(name="story_writer")

    def _dialogue_writer_node(self):
        return ToolNode(name="dialogue_writer")

    def _publishing_supervisor_node(self):
        return ToolNode(name="publishing_supervisor")

    def _consistency_checker_node(self):
        return ToolNode(name="consistency_checker")

    def _continuity_checker_node(self):
        return ToolNode(name="continuity_checker")

    def _editor_node(self):
        return ToolNode(name="editor")

    def _finalisation_node(self):
        return ToolNode(name="finalisation")

    def _conflict_resolution_node(self):
        return ToolNode(name="conflict_resolution")

    def _narrative_structure_node(self):
        return ToolNode(name="narrative_structure")

    def _thematic_analysis_node(self):
        return ToolNode(name="thematic_analysis")

    def _importation_agent_node(self):
        return ToolNode(name="importation_agent")

    def _multi_writer_review_node(self):
        return ToolNode(name="multi_writer_review")

    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the graph with an initial state."""
        graph = self.get_graph()
        return graph.run(initial_state)
