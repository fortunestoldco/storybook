# ...existing code...

def create_workflow_graph(self):
    nodes = {
        "research_team": self.research_team.process,
        "supervisor": self.supervisor.process,
        # Ensure node names match exactly
        "research_team_supervisor": self.handle_research_team_supervisor,
    }
    
    edges = {
        "research_team": {"supervisor": self.should_route_to_supervisor},
        "supervisor": {"research_team": self.should_route_to_research},
        # Update edge definitions if needed
    }

    graph = Graph(nodes=nodes)
    
    # Add edges
    for start, destinations in edges.items():
        for end, condition in destinations.items():
            graph.add_edge(start, end, condition)

    # Set correct interrupt node name
    graph.set_entry_point("research_team")
    graph.set_interrupt_handler("research_team_supervisor")  # Fix the node name

    return graph

# ...existing code...
