# storybook/innovations/creativity/emergence.py

"""
Creative Emergence Facilitator for fostering emergent creativity in narrative generation.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import random
import uuid
import networkx as nx
import numpy as np
from collections import defaultdict

class IdeaNetworkGraph:
    """Neural graph maintaining connections between narrative elements."""
    
    def __init__(self):
        """Initialize the idea network graph."""
        self.graph = nx.Graph()
        self.node_types = {}
        self.node_metadata = {}
        self.recently_added = []
        self.connections_strength = {}
        
    def add_element(self, element: str, element_type: str, metadata: Dict[str, Any]) -> str:
        """
        Add narrative element to the graph.
        
        Args:
            element: Element content
            element_type: Type of element
            metadata: Element metadata
            
        Returns:
            Node ID
        """
        # Generate node ID
        node_id = str(uuid.uuid4())
        
        # Add node to graph
        self.graph.add_node(node_id)
        self.node_types[node_id] = element_type
        self.node_metadata[node_id] = {
            "content": element,
            "type": element_type,
            **metadata
        }
        
        # Track recently added elements
        self.recently_added.append(node_id)
        if len(self.recently_added) > 20:
            self.recently_added.pop(0)
            
        # Find and add connections to existing nodes
        self._add_connections(node_id)
        
        return node_id
    
    def find_potential_connections(
        self, 
        node_id: str, 
        strength_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find potential connections for a narrative element.
        
        Args:
            node_id: ID of the node to find connections for
            strength_threshold: Minimum connection strength
            
        Returns:
            List of potential connections
        """
        if node_id not in self.graph:
            return []
            
        connections = []
        
        # Get existing connections
        existing_connections = set(self.graph.neighbors(node_id))
        
        for other_id in self.graph.nodes:
            if other_id == node_id or other_id in existing_connections:
                continue
                
            # Calculate connection strength
            strength = self._calculate_connection_strength(node_id, other_id)
            
            if strength >= strength_threshold:
                connections.append({
                    "node_id": other_id,
                    "content": self.node_metadata[other_id]["content"],
                    "type": self.node_types[other_id],
                    "strength": strength,
                    "potential": self._calculate_connection_potential(node_id, other_id)
                })
                
        # Sort by strength
        connections.sort(key=lambda x: x["strength"], reverse=True)
        
        return connections
    
    def identify_emergent_patterns(self) -> List[Dict[str, Any]]:
        """
        Identify emergent patterns in the narrative graph.
        
        Returns:
            List of emergent patterns
        """
        patterns = []
        
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        # Find clusters using community detection
        communities = nx.community.greedy_modularity_communities(self.graph)
        
        # Analyze each community
        for i, community in enumerate(communities):
            community_nodes = list(community)
            
            if len(community_nodes) < 3:
                continue
                
            # Get node types in this community
            community_types = [self.node_types[node] for node in community_nodes]
            
            # Check for thematic patterns
            theme_pattern = self._check_thematic_pattern(community_nodes)
            if theme_pattern:
                patterns.append(theme_pattern)
                
            # Check for character relationship patterns
            character_pattern = self._check_character_relationship_pattern(community_nodes)
            if character_pattern:
                patterns.append(character_pattern)
                
            # Check for plot twist potential
            twist_pattern = self._check_plot_twist_pattern(community_nodes)
            if twist_pattern:
                patterns.append(twist_pattern)
                
        # Find motif patterns
        motif_patterns = self._find_motif_patterns()
        patterns.extend(motif_patterns)
        
        return patterns
    
    def _add_connections(self, node_id: str) -> None:
        """
        Add connections between the new node and existing nodes.
        
        Args:
            node_id: ID of the new node
        """
        for other_id in self.graph.nodes:
            if other_id == node_id:
                continue
                
            # Calculate connection strength
            strength = self._calculate_connection_strength(node_id, other_id)
            
            # Add edge if strength is above threshold
            if strength >= 0.5:
                self.graph.add_edge(node_id, other_id, weight=strength)
                self.connections_strength[(node_id, other_id)] = strength
                self.connections_strength[(other_id, node_id)] = strength
    
    def _calculate_connection_strength(self, node1: str, node2: str) -> float:
        """
        Calculate connection strength between two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Connection strength (0.0 to 1.0)
        """
        # Get node metadata
        meta1 = self.node_metadata[node1]
        meta2 = self.node_metadata[node2]
        
        # Calculate keyword overlap
        content1 = meta1["content"].lower()
        content2 = meta2["content"].lower()
        
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        overlap = len(words1.intersection(words2)) / max(1, min(len(words1), len(words2)))
        
        # Check for type-specific connections
        type_connection = 0.0
        if self.node_types[node1] == self.node_types[node2]:
            type_connection = 0.3
        elif (self.node_types[node1] == "character" and self.node_types[node2] == "goal") or \
             (self.node_types[node2] == "character" and self.node_types[node1] == "goal"):
            type_connection = 0.5
        elif (self.node_types[node1] == "setting" and self.node_types[node2] == "event") or \
             (self.node_types[node2] == "setting" and self.node_types[node1] == "event"):
            type_connection = 0.4
            
        # Check for metadata-based connections
        metadata_connection = 0.0
        common_keys = set(meta1.keys()).intersection(set(meta2.keys())) - {"content", "type"}
        
        for key in common_keys:
            if meta1[key] == meta2[key]:
                metadata_connection += 0.1
                
        metadata_connection = min(0.5, metadata_connection)
        
        # Combine scores
        strength = 0.4 * overlap + 0.3 * type_connection + 0.3 * metadata_connection
        
        return strength
    
    def _calculate_connection_potential(self, node1: str, node2: str) -> Dict[str, Any]:
        """
        Calculate creative potential of connecting two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Connection potential information
        """
        # Get node metadata
        meta1 = self.node_metadata[node1]
        meta2 = self.node_metadata[node2]
        
        # Get node types
        type1 = self.node_types[node1]
        type2 = self.node_types[node2]
        
        # Check for specific potentials
        potentials = []
        
        # Character-theme potential
        if (type1 == "character" and type2 == "theme") or (type2 == "character" and type1 == "theme"):
            potentials.append({
                "type": "character_theme",
                "description": "Character embodiment of theme",
                "strength": 0.8
            })
            
        # Conflict potential
        if (type1 == "character" and type2 == "character"):
            potentials.append({
                "type": "conflict",
                "description": "Character conflict",
                "strength": 0.7
            })
            
        # Setting-atmosphere potential
        if (type1 == "setting" and type2 == "theme") or (type2 == "setting" and type1 == "theme"):
            potentials.append({
                "type": "setting_mood",
                "description": "Setting reflecting theme",
                "strength": 0.75
            })
            
        # Add other potentials based on content analysis
        # (In a real implementation, this would be more sophisticated)
        
        if not potentials:
            potentials.append({
                "type": "general",
                "description": "General narrative connection",
                "strength": 0.5
            })
            
        # Sort by strength
        potentials.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "potentials": potentials,
            "primary_potential": potentials[0] if potentials else None
        }
    
    def _check_thematic_pattern(self, nodes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check for thematic patterns in a set of nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Thematic pattern if found, None otherwise
        """
        # Check if there are theme nodes
        theme_nodes = [n for n in nodes if self.node_types[n] == "theme"]
        
        if not theme_nodes:
            return None
            
        # Check if there are related elements (characters, settings, events)
        related_nodes = [n for n in nodes if self.node_types[n] in ["character", "setting", "event"]]
        
        if len(related_nodes) < 2:
            return None
            
        # Get theme content
        themes = [self.node_metadata[n]["content"] for n in theme_nodes]
        related_elements = [self.node_metadata[n] for n in related_nodes]
        
        return {
            "type": "thematic_pattern",
            "elements": [{
                "id": n,
                "content": self.node_metadata[n]["content"],
                "type": self.node_types[n]
            } for n in nodes],
            "themes": themes,
            "related_elements": related_elements,
            "potential": random.uniform(0.7, 0.9)  # In real implementation, calculate based on factors
        }
    
    def _check_character_relationship_pattern(self, nodes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check for character relationship patterns in a set of nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Character relationship pattern if found, None otherwise
        """
        # Check if there are multiple character nodes
        character_nodes = [n for n in nodes if self.node_types[n] == "character"]
        
        if len(character_nodes) < 2:
            return None
            
        # Check for connection strength between characters
        relationships = []
        for i, char1 in enumerate(character_nodes):
            for char2 in character_nodes[i+1:]:
                if self.graph.has_edge(char1, char2):
                    strength = self.graph.get_edge_data(char1, char2).get("weight", 0.5)
                else:
                    strength = self._calculate_connection_strength(char1, char2)
                    
                if strength >= 0.4:
                    relationships.append({
                        "characters": [
                            self.node_metadata[char1]["content"],
                            self.node_metadata[char2]["content"]
                        ],
                        "strength": strength,
                        "potential_type": "conflict" if strength > 0.7 else "parallel"
                    })
        
        if not relationships:
            return None
            
        return {
            "type": "character_relationship_pattern",
            "elements": [{
                "id": n,
                "content": self.node_metadata[n]["content"],
                "type": self.node_types[n]
            } for n in character_nodes],
            "relationships": relationships,
            "potential": 0.8 if len(relationships) > 2 else 0.6
        }
    
    def _check_plot_twist_pattern(self, nodes: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check for plot twist potential in a set of nodes.
        
        Args:
            nodes: List of node IDs
            
        Returns:
            Plot twist pattern if found, None otherwise
        """
        # Check for elements that could create a twist
        event_nodes = [n for n in nodes if self.node_types[n] == "event"]
        secret_nodes = [n for n in nodes if self.node_types[n] == "secret"]
        character_nodes = [n for n in nodes if self.node_types[n] == "character"]
        
        # Need at least one event or secret and one character
        if (not event_nodes and not secret_nodes) or not character_nodes:
            return None
            
        twist_elements = []
        
        # Add secrets as twist elements
        for n in secret_nodes:
            twist_elements.append({
                "id": n,
                "content": self.node_metadata[n]["content"],
                "type": "secret",
                "potential": 0.9
            })
            
        # Add events with potential for surprise
        for n in event_nodes:
            # Check if this event has surprise potential
            meta = self.node_metadata[n]
            if "unexpected" in meta.get("tags", []) or "surprising" in meta.get("tags", []):
                twist_elements.append({
                    "id": n,
                    "content": meta["content"],
                    "type": "event",
                    "potential": 0.8
                })
        
        if not twist_elements:
            return None
            
        return {
            "type": "plot_twist_pattern",
            "elements": twist_elements,
            "characters": [{
                "id": n,
                "content": self.node_metadata[n]["content"]
            } for n in character_nodes],
            "potential": max(e["potential"] for e in twist_elements)
        }
    
    def _find_motif_patterns(self) -> List[Dict[str, Any]]:
        """
        Find recurring motif patterns in the graph.
        
        Returns:
            List of motif patterns
        """
        # Count word frequencies across all nodes
        word_counts = defaultdict(int)
        node_types = defaultdict(set)
        
        for node, meta in self.node_metadata.items():
            content = meta["content"].lower()
            words = content.split()
            
            for word in words:
                if len(word) > 4:  # Only consider words of sufficient length
                    word_counts[word] += 1
                    node_types[word].add(self.node_types[node])
        
        # Find recurring motifs (words that appear in multiple contexts)
        motifs = []
        for word, count in word_counts.items():
            if count >= 3 and len(node_types[word]) >= 2:
                # Find nodes containing this word
                related_nodes = []
                for node, meta in self.node_metadata.items():
                    if word in meta["content"].lower():
                        related_nodes.append(node)
                
                motifs.append({
                    "type": "motif_pattern",
                    "motif": word,
                    "occurrence_count": count,
                    "element_types": list(node_types[word]),
                    "elements": [{
                        "id": n,
                        "content": self.node_metadata[n]["content"],
                        "type": self.node_types[n]
                    } for n in related_nodes],
                    "potential": 0.6 + (0.1 * min(count / 5, 0.3))
                })
        
        # Sort by potential
        motifs.sort(key=lambda x: x["potential"], reverse=True)
        
        return motifs[:5]  # Return top 5 motifs


class PatternDetector:
    """Detects patterns that could lead to creative insights."""
    
    def detect_thematic_resonance(
        self, 
        narrative_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect thematic resonance between narrative elements.
        
        Args:
            narrative_elements: List of narrative elements
            
        Returns:
            List of thematic resonances
        """
        # Extract themes
        themes = [e for e in narrative_elements if e.get("type") == "theme"]
        
        if not themes:
            return []
            
        # Group other elements by type
        elements_by_type = defaultdict(list)
        for element in narrative_elements:
            if element.get("type") != "theme":
                elements_by_type[element.get("type", "unknown")].append(element)
                
        # Find resonances between themes and other elements
        resonances = []
        
        for theme in themes:
            theme_content = theme.get("content", "").lower()
            theme_keywords = self._extract_keywords(theme_content)
            
            for element_type, elements in elements_by_type.items():
                for element in elements:
                    element_content = element.get("content", "").lower()
                    element_keywords = self._extract_keywords(element_content)
                    
                    # Calculate keyword overlap
                    overlap = len(set(theme_keywords).intersection(set(element_keywords)))
                    
                    if overlap > 0:
                        resonance_strength = min(1.0, overlap / 5.0)
                        
                        if resonance_strength >= 0.3:
                            resonances.append({
                                "type": "thematic_resonance",
                                "theme": theme,
                                "element": element,
                                "strength": resonance_strength,
                                "shared_keywords": list(set(theme_keywords).intersection(set(element_keywords)))
                            })
        
        # Sort by strength
        resonances.sort(key=lambda x: x["strength"], reverse=True)
        
        return resonances
    
    def identify_subtext_opportunities(
        self, 
        narrative_elements: List[Dict[str, Any]], 
        character_models: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify opportunities for meaningful subtext.
        
        Args:
            narrative_elements: List of narrative elements
            character_models: Dictionary of character models
            
        Returns:
            List of subtext opportunities
        """
        # Look for elements related to secrets, motivations, or conflicts
        opportunities = []
        
        # Find elements with secret potential
        secrets = [e for e in narrative_elements if 
                  e.get("type") == "secret" or 
                  "secret" in e.get("tags", [])]
        
        # Find character motivations
        motivations = []
        for char_id, char_model in character_models.items():
            if "motivations" in char_model:
                for motivation in char_model["motivations"]:
                    motivations.append({
                        "character_id": char_id,
                        "character_name": char_model.get("name", ""),
                        "motivation": motivation
                    })
        
        # Find conflicts
        conflicts = [e for e in narrative_elements if 
                    e.get("type") == "conflict" or 
                    "conflict" in e.get("tags", [])]
        
        # Convert secrets to subtext opportunities
        for secret in secrets:
            for char_id, char_model in character_models.items():
                # Check if this secret relates to the character
                if char_id in secret.get("related_characters", []):
                    opportunities.append({
                        "type": "secret_subtext",
                        "character": {
                            "id": char_id,
                            "name": char_model.get("name", "")
                        },
                        "element": secret,
                        "potential": 0.8,
                        "suggestions": [
                            "Hint at the secret through behavior inconsistencies",
                            "Create dialogue with double meaning",
                            "Use symbolic objects or settings to suggest the hidden truth"
                        ]
                    })
        
        # Convert motivations to subtext opportunities
        for motivation in motivations:
            opportunities.append({
                "type": "motivation_subtext",
                "character": {
                    "id": motivation["character_id"],
                    "name": motivation["character_name"]
                },
                "element": {
                    "type": "motivation",
                    "content": motivation["motivation"]
                },
                "potential": 0.7,
                "suggestions": [
                    "Show behavior that hints at deeper motivation",
                    "Create situations where motivation conflicts with stated goals",
                    "Use physical reactions to suggest deeper concerns"
                ]
            })
        
        # Convert conflicts to subtext opportunities
        for conflict in conflicts:
            opportunities.append({
                "type": "conflict_subtext",
                "element": conflict,
                "potential": 0.75,
                "suggestions": [
                    "Show unspoken tension in interactions",
                    "Use setting and atmosphere to emphasize underlying conflict",
                    "Create parallel situations that mirror the central conflict"
                ]
            })
        
        # Sort by potential
        opportunities.sort(key=lambda x: x["potential"], reverse=True)
        
        return opportunities[:10]  # Return top 10 opportunities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction - in a real implementation would be more sophisticated
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "was", "were"}
        words = [w for w in text.lower().split() if w not in stop_words and len(w) > 3]
        
        # Return unique words
        return list(set(words))


class SerendipityInjector:
    """Injects controlled serendipity to spark creative connections."""
    
    def __init__(self):
        """Initialize the serendipity injector."""
        # Lists of potential serendipitous elements
        self.objects = [
            "locket", "old photograph", "strange key", "torn letter", "antique watch",
            "mysterious map", "rare book", "unusual plant", "family heirloom", "foreign coin"
        ]
        
        self.locations = [
            "hidden cellar", "abandoned lighthouse", "forgotten garden", "strange shop",
            "old attic", "secluded beach", "ancient library", "unexpected detour",
            "mysterious island", "underground tunnel"
        ]
        
        self.weather = [
            "unexpected storm", "unusual fog", "rare solar event", "sudden snowfall",
            "strange winds", "unusually clear night", "ominous clouds", "rainbow after rain"
        ]
        
        self.encounters = [
            "mysterious stranger", "childhood friend", "unusual animal", "eccentric local",
            "similar-looking person", "foreigner with knowledge", "person with unique skills",
            "someone who knows a secret"
        ]
        
        self.coincidences = [
            "unlikely timing", "shared birthday", "same unusual name", "parallel experiences",
            "identical possessions", "mutual acquaintance", "similar distinctive features"
        ]
    
    def generate_serendipitous_connection(
        self, 
        context: Dict[str, Any], 
        surprise_level: float
    ) -> Dict[str, Any]:
        """
        Generate a surprising but meaningful narrative connection.
        
        Args:
            context: Context information
            surprise_level: Level of surprise desired (0.0 to 1.0)
            
        Returns:
            Serendipitous connection
        """
        # Determine the type of serendipity based on context
        element_type = self._determine_serendipity_type(context)
        
        # Select elements based on context
        if element_type == "object":
            elements = self.objects
            integration_strategies = [
                "Introduce as an unexpected discovery",
                "Have character randomly receive it",
                "Place it prominently in a new setting",
                "Make it part of another character's possessions"
            ]
        elif element_type == "location":
            elements = self.locations
            integration_strategies = [
                "Force an unexpected detour to this location",
                "Create a reason for characters to seek it out",
                "Incorporate as a chance discovery",
                "Add as a recommendation from a minor character"
            ]
        elif element_type == "weather":
            elements = self.weather
            integration_strategies = [
                "Use to complicate an existing scene",
                "Make it force characters together",
                "Use it to reveal something previously hidden",
                "Have it symbolically mirror a character's emotional state"
            ]
        elif element_type == "encounter":
            elements = self.encounters
            integration_strategies = [
                "Create a chance meeting in an unusual place",
                "Make it an unexpected reunion",
                "Use it to reveal new information",
                "Have the encounter change a character's perspective"
            ]
        else:  # coincidence
            elements = self.coincidences
            integration_strategies = [
                "Reveal gradually through conversation",
                "Make it a surprising discovery",
                "Use it to create an immediate bond or conflict",
                "Have it relate to a central mystery"
            ]
        
        # Select a random element
        element = random.choice(elements)
        
        # Generate meaning based on context
        potential_meanings = self._generate_potential_meanings(element_type, element, context)
        narrative_impacts = self._generate_narrative_impacts(element_type, surprise_level)
        
        return {
            "type": "serendipitous_connection",
            "element_type": element_type,
            "element": element,
            "surprise_level": surprise_level,
            "potential_meanings": potential_meanings,
            "narrative_impacts": narrative_impacts,
            "integration_strategies": integration_strategies
        }
    
    def _determine_serendipity_type(self, context: Dict[str, Any]) -> str:
        """
        Determine the type of serendipity based on context.
        
        Args:
            context: Context information
            
        Returns:
            Type of serendipity
        """
        current_setting = context.get("setting", "")
        current_plot_point = context.get("plot_point", "")
        
        # Use context to determine what type of serendipity would be most effective
        if "outdoor" in current_setting.lower() or "travel" in current_plot_point.lower():
            candidates = ["weather", "location", "encounter"]
        elif "indoor" in current_setting.lower() or "exploration" in current_plot_point.lower():
            candidates = ["object", "encounter", "coincidence"]
        elif "social" in current_plot_point.lower() or "interaction" in current_plot_point.lower():
            candidates = ["encounter", "coincidence", "object"]
        else:
            candidates = ["object", "location", "weather", "encounter", "coincidence"]
            
        return random.choice(candidates)
    
    def _generate_potential_meanings(
        self, 
        element_type: str, 
        element: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Generate potential meanings for a serendipitous element.
        
        Args:
            element_type: Type of serendipitous element
            element: The serendipitous element
            context: Context information
            
        Returns:
            List of potential meanings
        """
        themes = context.get("themes", [])
        
        # Generate meanings tied to themes if available
        theme_meanings = []
        for theme in themes:
            theme_meanings.append(f"Symbol of {theme}")
            theme_meanings.append(f"Represents the conflict between {theme} and opposing forces")
            theme_meanings.append(f"Catalyst for exploring {theme}")
            
        # Generic meanings based on element type
        generic_meanings = {
            "object": [
                "Link to a character's past",
                "Key to a mystery",
                "Symbol of desire or fear",
                "Connection between characters"
            ],
            "location": [
                "Place of revelation",
                "Setting for confrontation",
                "Symbolic landscape",
                "Container of secrets"
            ],
            "weather": [
                "Mirror of emotional state",
                "Force driving plot forward",
                "Symbol of changing circumstances",
                "Backdrop highlighting thematic elements"
            ],
            "encounter": [
                "Catalyst for character change",
                "Source of important information",
                "Testing ground for character values",
                "Representation of what character needs"
            ],
            "coincidence": [
                "Revelation of underlying connections",
                "Illumination of fate or design",
                "Trigger for character realization",
                "Setup for dramatic irony"
            ]
        }
        
        # Combine theme-specific and generic meanings
        all_meanings = theme_meanings + generic_meanings.get(element_type, [])
        
        # Select a subset of meanings
        num_meanings = min(3, len(all_meanings))
        return random.sample(all_meanings, num_meanings)
    
    def _generate_narrative_impacts(
        self, 
        element_type: str, 
        surprise_level: float
    ) -> List[str]:
        """
        Generate potential narrative impacts.
        
        Args:
            element_type: Type of serendipitous element
            surprise_level: Level of surprise desired
            
        Returns:
            List of potential narrative impacts
        """
        # Generate impacts based on element type and surprise level
        if surprise_level > 0.7:
            # High surprise level - major impacts
            impacts = {
                "object": [
                    "Reveals a major secret",
                    "Completely changes character understanding",
                    "Provides key to central conflict"
                ],
                "location": [
                    "Becomes crucial setting for climax",
                    "Reveals hidden history essential to plot",
                    "Forces confrontation with antagonist"
                ],
                "weather": [
                    "Creates life-or-death situation",
                    "Dramatically alters planned course of action",
                    "Reveals something fundamental about setting or characters"
                ],
                "encounter": [
                    "Introduces key ally or antagonist",
                    "Provides critical information changing course of story",
                    "Forces character to confront core fears or flaws"
                ],
                "coincidence": [
                    "Reveals fundamental connection between plot elements",
                    "Forces complete reassessment of situation",
                    "Creates dramatic turning point in story"
                ]
            }
        elif surprise_level > 0.4:
            # Medium surprise level - moderate impacts
            impacts = {
                "object": [
                    "Provides new insight into character or situation",
                    "Creates meaningful subplot",
                    "Becomes symbolic of character journey"
                ],
                "location": [
                    "Becomes setting for important character development",
                    "Reveals new dimension to story world",
                    "Creates meaningful obstacle or opportunity"
                ],
                "weather": [
                    "Forces characters into unexpected interactions",
                    "Changes mood and atmosphere in meaningful way",
                    "Creates compelling scene backdrop"
                ],
                "encounter": [
                    "Provides new perspective on situation",
                    "Creates interesting relationship dynamic",
                    "Challenges character assumptions"
                ],
                "coincidence": [
                    "Creates intriguing connection between characters",
                    "Adds layer of meaning to existing events",
                    "Leads to useful discovery"
                ]
            }
        else:
            # Low surprise level - subtle impacts
            impacts = {
                "object": [
                    "Adds interesting detail to scene",
                    "Creates subtle foreshadowing",
                    "Enriches character interaction"
                ],
                "location": [
                    "Provides interesting backdrop for scene",
                    "Adds texture to story world",
                    "Creates minor challenge or opportunity"
                ],
                "weather": [
                    "Sets mood for scene",
                    "Creates minor complication",
                    "Adds sensory richness to setting"
                ],
                "encounter": [
                    "Adds interesting minor character",
                    "Creates moment of character insight",
                    "Provides small piece of information"
                ],
                "coincidence": [
                    "Creates subtle connection between elements",
                    "Adds touch of serendipity to story",
                    "Provides small surprise for reader"
                ]
            }
            
        return impacts.get(element_type, ["Creates interesting story development"])


class CreativeEmergenceFacilitator:
    """System to facilitate emergent creativity across agent collaborations."""
    
    def __init__(self):
        """Initialize the creative emergence facilitator."""
        self.idea_network = IdeaNetworkGraph()
        self.pattern_detector = PatternDetector()
        self.serendipity_injector = SerendipityInjector()
    
    def facilitate_creative_connections(
        self, 
        manuscript_state: Dict[str, Any], 
        agent_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Facilitate creative connections between agent outputs.
        
        Args:
            manuscript_state: Dictionary with manuscript state information
            agent_outputs: Dictionary of agent outputs
            
        Returns:
            Creative connections and suggestions
        """
        # Extract narrative elements from agent outputs
        elements = self._extract_narrative_elements(agent_outputs)
        
        # Add elements to idea network
        for element in elements:
            self.idea_network.add_element(
                element["content"],
                element["type"],
                element["metadata"]
            )
            
        # Detect patterns and potential connections
        emergent_patterns = self.idea_network.identify_emergent_patterns()
        
        # Extract recent elements from manuscript state
        recent_elements = self._get_recent_elements(manuscript_state)
        
        # Detect thematic resonances
        thematic_resonances = self.pattern_detector.detect_thematic_resonance(recent_elements)
        
        # Generate subtext opportunities
        character_models = manuscript_state.get("characters", {})
        subtext_opportunities = self.pattern_detector.identify_subtext_opportunities(
            recent_elements, character_models)
            
        # Generate creative connection suggestions
        connection_suggestions = []
        for pattern in emergent_patterns:
            connection_suggestions.append({
                "elements": pattern.get("elements", []),
                "connection_type": pattern.get("type", ""),
                "narrative_potential": pattern.get("potential", 0.5),
                "integration_guidance": self._generate_integration_guidance(pattern)
            })
            
        # Add serendipitous elements
        current_context = manuscript_state.get("current_context", {})
        surprise_level = manuscript_state.get("creativity_parameters", {}).get("surprise_level", 0.5)
        serendipity = self.serendipity_injector.generate_serendipitous_connection(
            current_context, surprise_level)
            
        return {
            "connection_suggestions": connection_suggestions,
            "thematic_resonances": thematic_resonances,
            "subtext_opportunities": subtext_opportunities,
            "serendipitous_elements": serendipity,
            "creative_guidance": self._generate_creative_guidance(
                manuscript_state, 
                connection_suggestions, 
                thematic_resonances,
                subtext_opportunities,
                serendipity
            )
        }
    
    def _extract_narrative_elements(
        self, 
        agent_outputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract narrative elements from agent outputs.
        
        Args:
            agent_outputs: Dictionary of agent outputs
            
        Returns:
            List of narrative elements
        """
        elements = []
        
        # Process each agent's output
        for agent_id, output in agent_outputs.items():
            if "output" not in output:
                continue
                
            agent_output = output["output"]
            
            # Extract characters
            if "characters" in agent_output:
                for character in agent_output["characters"]:
                    elements.append({
                        "content": character.get("name", "") + " - " + character.get("description", ""),
                        "type": "character",
                        "metadata": {
                            "agent_id": agent_id,
                            "character_id": character.get("id", str(uuid.uuid4())),
                            "traits": character.get("traits", []),
                            "goals": character.get("goals", [])
                        }
                    })
                    
                    # Add character goals as separate elements
                    for goal in character.get("goals", []):
                        elements.append({
                            "content": goal,
                            "type": "goal",
                            "metadata": {
                                "agent_id": agent_id,
                                "character_id": character.get("id"),
                                "related_characters": [character.get("id")]
                            }
                        })
            
            # Extract settings
            if "settings" in agent_output:
                for setting in agent_output["settings"]:
                    elements.append({
                        "content": setting.get("name", "") + " - " + setting.get("description", ""),
                        "type": "setting",
                        "metadata": {
                            "agent_id": agent_id,
                            "setting_id": setting.get("id", str(uuid.uuid4())),
                            "atmosphere": setting.get("atmosphere", "")
                        }
                    })
            
            # Extract themes
            if "themes" in agent_output:
                for theme in agent_output["themes"]:
                    if isinstance(theme, dict):
                        elements.append({
                            "content": theme.get("name", "") + " - " + theme.get("description", ""),
                            "type": "theme",
                            "metadata": {
                                "agent_id": agent_id,
                                "theme_id": theme.get("id", str(uuid.uuid4()))
                            }
                        })
                    else:
                        elements.append({
                            "content": theme,
                            "type": "theme",
                            "metadata": {
                                "agent_id": agent_id,
                                "theme_id": str(uuid.uuid4())
                            }
                        })
            
            # Extract plot points
            if "plot_points" in agent_output:
                for plot_point in agent_output["plot_points"]:
                    elements.append({
                        "content": plot_point.get("description", ""),
                        "type": "event",
                        "metadata": {
                            "agent_id": agent_id,
                            "event_id": plot_point.get("id", str(uuid.uuid4())),
                            "related_characters": plot_point.get("characters", [])
                        }
                    })
            
            # Extract secrets (if available)
            if "secrets" in agent_output:
                for secret in agent_output["secrets"]:
                    elements.append({
                        "content": secret.get("description", ""),
                        "type": "secret",
                        "metadata": {
                            "agent_id": agent_id,
                            "secret_id": secret.get("id", str(uuid.uuid4())),
                            "related_characters": secret.get("related_characters", [])
                        }
                    })
            
            # Extract conflicts (if available)
            if "conflicts" in agent_output:
                for conflict in agent_output["conflicts"]:
                    elements.append({
                        "content": conflict.get("description", ""),
                        "type": "conflict",
                        "metadata": {
                            "agent_id": agent_id,
                            "conflict_id": conflict.get("id", str(uuid.uuid4())),
                            "related_characters": conflict.get("related_characters", [])
                        }
                    })
        
        return elements
    
    def _get_recent_elements(self, manuscript_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract recent elements from manuscript state.
        
        Args:
            manuscript_state: Manuscript state
            
        Returns:
            List of recent elements
        """
        elements = []
        
        # Extract characters
        for char_id, character in manuscript_state.get("characters", {}).items():
            elements.append({
                "id": char_id,
                "content": character.get("name", "") + " - " + character.get("description", ""),
                "type": "character",
                "name": character.get("name", ""),
                "tags": character.get("traits", [])
            })
        
        # Extract settings
        for setting_id, setting in manuscript_state.get("settings", {}).items():
            elements.append({
                "id": setting_id,
                "content": setting.get("name", "") + " - " + setting.get("description", ""),
                "type": "setting",
                "name": setting.get("name", ""),
                "tags": [setting.get("atmosphere", "")]
            })
        
        # Extract themes
        for theme in manuscript_state.get("themes", []):
            if isinstance(theme, dict):
                elements.append({
                    "id": theme.get("id", str(uuid.uuid4())),
                    "content": theme.get("name", "") + " - " + theme.get("description", ""),
                    "type": "theme",
                    "name": theme.get("name", "")
                })
            else:
                elements.append({
                    "id": str(uuid.uuid4()),
                    "content": theme,
                    "type": "theme",
                    "name": theme
                })
        
        # Extract recent plot points
        plot_points = manuscript_state.get("plot_points", {})
        if isinstance(plot_points, dict):
            for pp_id, plot_point in plot_points.items():
                elements.append({
                    "id": pp_id,
                    "content": plot_point.get("description", ""),
                    "type": "event",
                    "name": plot_point.get("title", ""),
                    "related_characters": plot_point.get("characters", []),
                    "tags": plot_point.get("tags", [])
                })
        elif isinstance(plot_points, list):
            for plot_point in plot_points:
                if isinstance(plot_point, dict):
                    elements.append({
                        "id": plot_point.get("id", str(uuid.uuid4())),
                        "content": plot_point.get("description", ""),
                        "type": "event",
                        "name": plot_point.get("title", ""),
                        "related_characters": plot_point.get("characters", []),
                        "tags": plot_point.get("tags", [])
                    })
        
        return elements
    
    def _generate_integration_guidance(self, pattern: Dict[str, Any]) -> List[str]:
        """
        Generate guidance for integrating a pattern into the narrative.
        
        Args:
            pattern: Pattern information
            
        Returns:
            List of integration guidance
        """
        guidance = []
        
        pattern_type = pattern.get("type", "")
        
        if pattern_type == "thematic_pattern":
            guidance = [
                "Develop scenes where these elements interact to highlight the theme",
                "Use imagery and symbolism consistent with the theme across these elements",
                "Create dialogue that subtly references the thematic connections"
            ]
        elif pattern_type == "character_relationship_pattern":
            guidance = [
                "Develop scenes that explore these character connections",
                "Create situations where their relationships are tested or revealed",
                "Use their interactions to highlight character growth"
            ]
        elif pattern_type == "plot_twist_pattern":
            guidance = [
                "Carefully foreshadow the twist without revealing it",
                "Ensure the twist relates to character motivations and growth",
                "Create a moment of revelation with maximum emotional impact"
            ]
        elif pattern_type == "motif_pattern":
            motif = pattern.get("motif", "")
            guidance = [
                f"Weave the '{motif}' motif into descriptions and settings",
                f"Use the '{motif}' motif in metaphors and similes",
                f"Connect the '{motif}' motif to thematic elements for deeper meaning"
            ]
        else:
            guidance = [
                "Find natural ways to connect these elements in the narrative",
                "Consider how these connections might reveal deeper meaning",
                "Use these connections to create more cohesive storytelling"
            ]
            
        return guidance
    
    def _generate_creative_guidance(
        self,
        manuscript_state: Dict[str, Any],
        connection_suggestions: List[Dict[str, Any]],
        thematic_resonances: List[Dict[str, Any]],
        subtext_opportunities: List[Dict[str, Any]],
        serendipity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate overall creative guidance based on all findings.
        
        Args:
            manuscript_state: Manuscript state
            connection_suggestions: Connection suggestions
            thematic_resonances: Thematic resonances
            subtext_opportunities: Subtext opportunities
            serendipity: Serendipitous elements
            
        Returns:
            Creative guidance
        """
        # Get current context
        current_context = manuscript_state.get("current_context", {})
        stage = current_context.get("stage", "unknown")
        
        # Generate stage-specific guidance
        if stage == "planning":
            guidance_type = "structural_guidance"
            specific_guidance = [
                "Focus on developing the most promising thematic patterns",
                "Ensure character relationships support the main themes",
                "Build structure that allows for the identified subtext opportunities"
            ]
        elif stage == "drafting":
            guidance_type = "drafting_guidance"
            specific_guidance = [
                "Incorporate the serendipitous elements to add freshness to the narrative",
                "Develop scenes that exploit the subtext opportunities",
                "Use the thematic resonances to deepen meaning in key scenes"
            ]
        elif stage == "revision":
            guidance_type = "revision_guidance"
            specific_guidance = [
                "Strengthen connections between the identified thematic elements",
                "Ensure subtext is clear enough to be felt but subtle enough to remain subtext",
                "Verify that serendipitous elements serve narrative purpose rather than seeming random"
            ]
        else:
            guidance_type = "general_guidance"
            specific_guidance = [
                "Look for opportunities to create meaningful connections between elements",
                "Balance planned structure with serendipitous discoveries",
                "Develop multilayered meaning through thematic connections"
            ]
            
        # Add high-potential opportunities
        high_potential_opportunities = []
        
        # Add connection suggestions with high potential
        for suggestion in connection_suggestions:
            if suggestion.get("narrative_potential", 0) > 0.7:
                elements = suggestion.get("elements", [])
                element_names = [e.get("content", "").split(" - ")[0] for e in elements]
                high_potential_opportunities.append({
                    "type": suggestion.get("connection_type", "connection"),
                    "elements": element_names,
                    "potential": suggestion.get("narrative_potential", 0)
                })
                
        # Add subtext opportunities with high potential
        for opportunity in subtext_opportunities:
            if opportunity.get("potential", 0) > 0.7:
                character_name = opportunity.get("character", {}).get("name", "Character")
                element_content = opportunity.get("element", {}).get("content", "")
                high_potential_opportunities.append({
                    "type": opportunity.get("type", "subtext"),
                    "description": f"{character_name}: {element_content}",
                    "potential": opportunity.get("potential", 0)
                })
                
        # Sort by potential
        high_potential_opportunities.sort(key=lambda x: x["potential"], reverse=True)
                
        return {
            "guidance_type": guidance_type,
            "specific_guidance": specific_guidance,
            "high_potential_opportunities": high_potential_opportunities[:5],
            "recommended_serendipitous_element": {
                "element": serendipity.get("element", ""),
                "element_type": serendipity.get("element_type", ""),
                "integration_strategies": serendipity.get("integration_strategies", [])[:2]
            }
        }
