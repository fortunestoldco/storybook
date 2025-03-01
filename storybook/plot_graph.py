from langchain import graphs
from typing import Dict, List

def create_knowledge_graph(nodes: List[Dict], relationships: List[Dict]):
    """
    Create a knowledge graph from nodes and relationships
    """
    graph = graphs.Graph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node['id'], **node.get('properties', {}))
    
    # Add relationships
    for rel in relationships:
        graph.add_edge(
            rel['source'],
            rel['target'],
            rel.get('type', ''),
            **rel.get('properties', {})
        )
    
    return graph

def visualize_graph(graph: graphs.Graph, output_path: str = 'story_graph.html'):
    """
    Visualize the graph and save to HTML file
    """
    graph.visualize(output_path)
