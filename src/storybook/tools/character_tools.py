from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
from datetime import datetime
import uuid

from db_config import get_collection, COLLECTIONS

@tool
def voice_pattern_template(character_id: str, project_id: str) -> Dict[str, Any]:
    """Create speech fingerprints for characters.
    
    Args:
        character_id: ID of the character.
        project_id: The project ID.
        
    Returns:
        Character voice pattern template.
    """
    characters_collection = get_collection(COLLECTIONS["characters"])
    voice_collection = get_collection("character_voices")
    
    # Get character information
    character = characters_collection.find_one({"character_id": character_id, "project_id": project_id})
    
    if not character:
        return {"error": f"Character {character_id} not found"}
    
    # This would be based on character attributes in a real implementation
    # Here we're using placeholder values
    character_name = character.get("name", "Unnamed Character")
    character_background = character.get("background", {})
    character_traits = character.get("traits", [])
    
    # Generate voice patterns based on character information
    patterns = [
        "Uses frequent rhetorical questions",
        "Tends to speak in short, direct sentences",
        "Often references personal experiences",
        "Interrupts own thoughts with clarifications"
    ]
    
    # Generate vocabulary preferences
    vocabulary = {
        "preferred_terms": ["absolutely", "fundamentally", "precisely"],
        "avoided_terms": ["maybe", "perhaps", "possibly"],
        "expletive_usage": "minimal",
        "formality_level": "moderate"
    }
    
    # Generate speech quirks
    speech_quirks = [
        "Clears throat when nervous",
        "Trails off at end of sentences when unsure",
        "Uses hands expressively while speaking"
    ]
    
    # Store the voice template in MongoDB
    voice_template = {
        "template_id": str(uuid.uuid4()),
        "character_id": character_id,
        "character_name": character_name,
        "project_id": project_id,
        "patterns": patterns,
        "vocabulary": vocabulary,
        "speech_quirks": speech_quirks,
        "examples": [
            "\"Well, I've *absolutely* never seen that before. Have you? No, I thought not. It's *precisely* what I was concerned about.\"",
            "\"This is... this is fundamentally wrong. I can't... [clears throat] I won't accept it.\""
        ],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    voice_collection.insert_one(voice_template)
    
    return {
        "character_name": character_name,
        "patterns": patterns,
        "vocabulary": vocabulary,
        "speech_quirks": speech_quirks,
        "template_id": voice_template["template_id"]
    }

@tool
def relationship_matrix_generator(project_id: str, characters: List[str] = None) -> Dict[str, Any]:
    """Map connections between all characters.
    
    Args:
        project_id: The project ID.
        characters: Optional list of character IDs.
        
    Returns:
        Matrix of character relationships.
    """
    characters_collection = get_collection(COLLECTIONS["characters"])
    relationships_collection = get_collection(COLLECTIONS["character_relationships"])
    
    # Get characters if not provided
    if not characters:
        character_docs = list(characters_collection.find({"project_id": project_id}, {"_id": 0, "character_id": 1}))
        characters = [c.get("character_id") for c in character_docs]
    
    # Generate matrix of relationships
    relationships = {}
    dynamics = []
    conflicts = []
    
    for i, char1_id in enumerate(characters):
        char1 = characters_collection.find_one({"character_id": char1_id, "project_id": project_id})
        if not char1:
            continue
            
        char1_name = char1.get("name", f"Character {i+1}")
        relationships[char1_id] = {}
        
        for j, char2_id in enumerate(characters):
            if char1_id == char2_id:
                continue
                
            char2 = characters_collection.find_one({"character_id": char2_id, "project_id": project_id})
            if not char2:
                continue
                
            char2_name = char2.get("name", f"Character {j+1}")
            
            # Check if relationship already exists in database
            existing_relationship = relationships_collection.find_one({
                "project_id": project_id,
                "$or": [
                    {"character1_id": char1_id, "character2_id": char2_id},
                    {"character1_id": char2_id, "character2_id": char1_id}
                ]
            })
            
            if existing_relationship:
                relationship_type = existing_relationship.get("relationship_type", "acquaintance")
                sentiment = existing_relationship.get("sentiment", "neutral")
                power_balance = existing_relationship.get("power_balance", "equal")
                relationship_id = existing_relationship.get("relationship_id")
            else:
                # Create placeholder relationship
                relationship_types = ["friend", "enemy", "family", "colleague", "romantic", "mentor", "rival"]
                sentiments = ["positive", "negative", "complex", "neutral"]
                power_balances = ["char1_dominant", "char2_dominant", "equal", "fluctuating"]
                
                relationship_type = relationship_types[(hash(char1_id + char2_id) % len(relationship_types))]
                sentiment = sentiments[(hash(char1_id + char2_id + "s") % len(sentiments))]
                power_balance = power_balances[(hash(char1_id + char2_id + "p") % len(power_balances))]
                
                # Create new relationship in database
                relationship_id = str(uuid.uuid4())
                new_relationship = {
                    "relationship_id": relationship_id,
                    "project_id": project_id,
                    "character1_id": char1_id,
                    "character1_name": char1_name,
                    "character2_id": char2_id,
                    "character2_name": char2_name,
                    "relationship_type": relationship_type,
                    "sentiment": sentiment,
                    "power_balance": power_balance,
                    "tension_level": (hash(char1_id + char2_id + "t") % 10) / 10.0,  # 0.0-0.9
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                relationships_collection.insert_one(new_relationship)
            
            # Add to relationships matrix
            relationships[char1_id][char2_id] = {
                "relationship_id": relationship_id,
                "type": relationship_type,
                "sentiment": sentiment,
                "power_balance": power_balance
            }
            
            # Add to dynamics list
            dynamics.append({
                "character1_id": char1_id,
                "character1_name": char1_name,
                "character2_id": char2_id,
                "character2_name": char2_name,
                "relationship_type": relationship_type,
                "dynamic": f"{sentiment.capitalize()} {relationship_type} relationship with {power_balance} power balance"
            })
            
            # Add to conflicts list if negative
            if sentiment in ["negative", "complex"]:
                conflicts.append({
                    "character1_id": char1_id,
                    "character1_name": char1_name,
                    "character2_id": char2_id,
                    "character2_name": char2_name,
                    "conflict_type": "rivalry" if relationship_type == "rival" else "antagonism" if relationship_type == "enemy" else "tension",
                    "description": f"Ongoing {sentiment} dynamic between {char1_name} and {char2_name}"
                })
    
    # Store the matrix in MongoDB
    matrix_document = {
        "matrix_id": str(uuid.uuid4()),
        "project_id": project_id,
        "relationships": relationships,
        "dynamics": dynamics,
        "conflicts": conflicts,
        "created_at": datetime.utcnow()
    }
    
    matrix_collection = get_collection("relationship_matrices")
    matrix_collection.insert_one(matrix_document)
    
    return {
        "relationships": relationships,
        "dynamics": dynamics,
        "conflicts": conflicts,
        "matrix_id": matrix_document["matrix_id"]
    }

@tool
def dynamic_evolution_planner(project_id: str, relationship_id: str) -> Dict[str, Any]:
    """Track relationship changes over time.
    
    Args:
        project_id: The project ID.
        relationship_id: ID of the relationship to track.
        
    Returns:
        Evolution plan for the relationship.
    """
    relationships_collection = get_collection(COLLECTIONS["character_relationships"])
    evolution_collection = get_collection("relationship_evolutions")
    
    # Get relationship information
    relationship = relationships_collection.find_one({"relationship_id": relationship_id, "project_id": project_id})
    
    if not relationship:
        return {"error": f"Relationship {relationship_id} not found"}
    
    # Get characters
    characters_collection = get_collection(COLLECTIONS["characters"])
    char1 = characters_collection.find_one({"character_id": relationship["character1_id"], "project_id": project_id})
    char2 = characters_collection.find_one({"character_id": relationship["character2_id"], "project_id": project_id})
    
    if not char1 or not char2:
        return {"error": "One or both characters not found"}
    
    char1_name = char1.get("name", "Character 1")
    char2_name = char2.get("name", "Character 2")
    
    # Generate evolution points - this would be based on plot in a real implementation
    # Here we're using placeholder values based on relationship type
    relationship_type = relationship.get("relationship_type", "acquaintance")
    initial_sentiment = relationship.get("sentiment", "neutral")
    
    evolution_points = []
    catalysts = []
    
    if relationship_type == "rival" and initial_sentiment in ["negative", "complex"]:
        evolution_points = [
            {
                "stage": 1,
                "description": f"Initial rivalry: {char1_name} and {char2_name} compete for the same goal",
                "sentiment": "negative",
                "chapter_estimate": 2
            },
            {
                "stage": 2,
                "description": f"Grudging respect: {char1_name} recognizes {char2_name}'s skills",
                "sentiment": "complex",
                "chapter_estimate": 6
            },
            {
                "stage": 3,
                "description": f"Forced cooperation: External threat forces them to work together",
                "sentiment": "complex",
                "chapter_estimate": 10
            },
            {
                "stage": 4,
                "description": f"Mutual respect: {char1_name} and {char2_name} develop genuine respect",
                "sentiment": "positive",
                "chapter_estimate": 15
            }
        ]
        
        catalysts = [
            {
                "event": f"{char1_name} sees {char2_name} succeed where they failed",
                "impact": "Forces reassessment of rival's abilities",
                "placement": "After stage 1"
            },
            {
                "event": "External threat targets both characters",
                "impact": "Creates need for cooperation despite rivalry",
                "placement": "Beginning of stage 3"
            },
            {
                "event": f"{char2_name} saves {char1_name} from danger",
                "impact": "Demonstrates value of relationship despite differences",
                "placement": "During stage 3"
            }
        ]
    elif relationship_type == "romantic":
        evolution_points = [
            {
                "stage": 1,
                "description": f"Initial attraction: {char1_name} and {char2_name} meet and feel connection",
                "sentiment": "positive",
                "chapter_estimate": 3
            },
            {
                "stage": 2,
                "description": "Growing closeness: Characters learn more about each other",
                "sentiment": "positive",
                "chapter_estimate": 7
            },
            {
                "stage": 3,
                "description": "Conflict: Key difference or external obstacle threatens relationship",
                "sentiment": "complex",
                "chapter_estimate": 12
            },
            {
                "stage": 4,
                "description": "Resolution: Characters overcome obstacle through growth",
                "sentiment": "positive",
                "chapter_estimate": 18
            }
        ]
        
        catalysts = [
            {
                "event": "Characters bond over shared experience",
                "impact": "Deepens emotional connection",
                "placement": "During stage 2"
            },
            {
                "event": "Secret from past is revealed",
                "impact": "Tests trust in the relationship",
                "placement": "Beginning of stage 3"
            },
            {
                "event": "Character makes personal sacrifice for the other",
                "impact": "Demonstrates commitment and love",
                "placement": "End of stage 3"
            }
        ]
    else:
        # Generic evolution for other relationship types
        evolution_points = [
            {
                "stage": 1,
                "description": f"Initial {relationship_type} relationship established",
                "sentiment": initial_sentiment,
                "chapter_estimate": 3
            },
            {
                "stage": 2,
                "description": "Relationship tested by challenge",
                "sentiment": "complex",
                "chapter_estimate": 9
            },
            {
                "stage": 3,
                "description": "Relationship evolves based on response to challenge",
                "sentiment": "positive" if initial_sentiment != "positive" else "complex",
                "chapter_estimate": 15
            }
        ]
        
        catalysts = [
            {
                "event": "Shared goal brings characters together",
                "impact": "Establishes common ground",
                "placement": "During stage 1"
            },
            {
                "event": "Conflicting interests create tension",
                "impact": "Forces characters to reassess relationship",
                "placement": "Beginning of stage 2"
            },
            {
                "event": "Resolution of conflict through compromise",
                "impact": "Strengthens relationship foundation",
                "placement": "End of stage 2"
            }
        ]
    
    # Store the evolution plan in MongoDB
    evolution_plan = {
        "plan_id": str(uuid.uuid4()),
        "project_id": project_id,
        "relationship_id": relationship_id,
        "character1_id": relationship["character1_id"],
        "character1_name": char1_name,
        "character2_id": relationship["character2_id"],
        "character2_name": char2_name,
        "relationship_type": relationship_type,
        "initial_sentiment": initial_sentiment,
        "evolution_points": evolution_points,
        "catalysts": catalysts,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    evolution_collection.insert_one(evolution_plan)
    
    return {
        "relationship_type": relationship_type,
        "character1_name": char1_name,
        "character2_name": char2_name,
        "evolution_points": evolution_points,
        "catalysts": catalysts,
        "plan_id": evolution_plan["plan_id"]
    }

@tool
def interaction_pattern_analyzer(project_id: str, character_id: str = None) -> Dict[str, Any]:
    """Identify relationship consistency issues.
    
    Args:
        project_id: The project ID.
        character_id: Optional character ID to focus on.
        
    Returns:
        Analysis of character interaction patterns.
    """
    interactions_collection = get_collection("character_interactions")
    
    # This would analyze dialogue and scenes in a real implementation
    # Here we're using placeholder values
    characters_collection = get_collection(COLLECTIONS["characters"])
    relationships_collection = get_collection(COLLECTIONS["character_relationships"])
    
    # Get character information if specified
    character = None
    if character_id:
        character = characters_collection.find_one({"character_id": character_id, "project_id": project_id})
        if not character:
            return {"error": f"Character {character_id} not found"}
    
    # Get relevant relationships
    relationship_query = {"project_id": project_id}
    if character_id:
        relationship_query["$or"] = [
            {"character1_id": character_id},
            {"character2_id": character_id}
        ]
    
    relationships = list(relationships_collection.find(relationship_query, {"_id": 0}))
    
    # Generate interaction patterns
    patterns = []
    
    for relationship in relationships:
        char1_id = relationship["character1_id"]
        char2_id = relationship["character2_id"]
        
        # Skip if not focused character
        if character_id and character_id != char1_id and character_id != char2_id:
            continue
            
        char1 = characters_collection.find_one({"character_id": char1_id}) or {"name": "Character 1"}
        char2 = characters_collection.find_one({"character_id": char2_id}) or {"name": "Character 2"}
        
        patterns.append({
            "character1_id": char1_id,
            "character1_name": char1.get("name"),
            "character2_id": char2_id,
            "character2_name": char2.get("name"),
            "relationship_type": relationship.get("relationship_type", "acquaintance"),
            "interaction_style": "confrontational" if relationship.get("sentiment") == "negative" else 
                                "supportive" if relationship.get("sentiment") == "positive" else
                                "cautious",
            "frequency": "high" if (hash(char1_id + char2_id) % 3 == 0) else
                         "medium" if (hash(char1_id + char2_id) % 3 == 1) else
                         "low",
            "power_dynamic": relationship.get("power_balance", "equal")
        })
    
    # Identify anomalies
    anomalies = []
    
    # In a real implementation, this would find actual inconsistencies
    # Here we're adding a placeholder anomaly
    if patterns and len(patterns) > 1:
        random_pattern = patterns[hash(str(patterns)) % len(patterns)]
        
        anomalies.append({
            "character1_name": random_pattern["character1_name"],
            "character2_name": random_pattern["character2_name"],
            "description": f"Inconsistent {random_pattern['interaction_style']} interactions in Chapter 7",
            "severity": "medium",
            "location": "Chapter 7, Scene 3"
        })
    
    # Generate suggestions
    suggestions = []
    
    for anomaly in anomalies:
        suggestions.append(f"Revise {anomaly['location']} to maintain consistent interaction style between {anomaly['character1_name']} and {anomaly['character2_name']}")
    
    if patterns:
        suggestions.append(f"Consider adding more varied interaction contexts between characters")
    
    # Store the analysis in MongoDB
    interaction_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "project_id": project_id,
        "character_id": character_id,
        "patterns": patterns,
        "anomalies": anomalies,
        "suggestions": suggestions,
        "created_at": datetime.utcnow()
    }
    
    interactions_collection.insert_one(interaction_analysis)
    
    return {
        "patterns": patterns,
        "anomalies": anomalies,
        "suggestions": suggestions,
        "analysis_id": interaction_analysis["analysis_id"]
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [voice_pattern_template, relationship_matrix_generator,
                 dynamic_evolution_planner, interaction_pattern_analyzer]:
    tool_registry.register_tool(tool_func, "character")