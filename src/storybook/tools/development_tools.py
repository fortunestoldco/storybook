from typing import Dict, Any, List
from langchain_core.tools import BaseTool, tool
import uuid
from datetime import datetime

from ..db_config import get_collection, COLLECTIONS
from ..models.story_structure import StoryStructure, PlotElement
from ..models.character import Character
from ..models.world_building import WorldBuildingEntry

@tool
def story_structure_template(structure_type: str, project_id: str) -> Dict[str, Any]:
    """Access proven narrative frameworks.

    Args:
        structure_type: Type of structure template to retrieve.
        project_id: The project ID.

    Returns:
        Template structure definition.
    """
    structures_collection = get_collection(COLLECTIONS["story_structures"])
    
    # Basic structure templates
    templates = {
        "three_act": {
            "name": "Three-Act Structure",
            "sections": [
                {"name": "Act 1: Setup", "description": "Introduce the world and characters, establish the status quo, and present the inciting incident."},
                {"name": "Act 2: Confrontation", "description": "The protagonist faces obstacles and conflicts while pursuing their goal."},
                {"name": "Act 3: Resolution", "description": "The climax occurs, followed by falling action and resolution."}
            ],
            "points": [
                {"name": "Inciting Incident", "description": "The event that sets the story in motion."},
                {"name": "First Plot Point", "description": "The protagonist crosses the threshold into the main conflict."},
                {"name": "Midpoint", "description": "A major turning point that raises the stakes."},
                {"name": "Second Plot Point", "description": "The final obstacle before the climax."},
                {"name": "Climax", "description": "The peak of tension and conflict resolution."}
            ]
        },
        "hero_journey": {
            "name": "Hero's Journey",
            "sections": [
                {"name": "Departure", "description": "The hero receives a call to adventure and crosses into the unknown."},
                {"name": "Initiation", "description": "The hero faces trials, meets allies and enemies, and approaches the central ordeal."},
                {"name": "Return", "description": "The hero returns transformed with the power to bestow boons on their community."}
            ],
            "points": [
                {"name": "Ordinary World", "description": "The hero's normal life before the adventure."},
                {"name": "Call to Adventure", "description": "The hero is presented with a challenge or quest."},
                {"name": "Refusal of the Call", "description": "Initial hesitation to accept the challenge."},
                {"name": "Meeting the Mentor", "description": "Encountering a guide who provides advice and tools."},
                {"name": "Crossing the Threshold", "description": "Committing to the adventure and entering the special world."},
                {"name": "Tests, Allies, Enemies", "description": "Facing challenges and forming relationships."},
                {"name": "Approach to the Inmost Cave", "description": "Preparing for the major challenge."},
                {"name": "Ordeal", "description": "The central crisis where the hero faces their greatest fear."},
                {"name": "Reward", "description": "Seizing the treasure or reward from the ordeal."},
                {"name": "The Road Back", "description": "Beginning the return journey."},
                {"name": "Resurrection", "description": "A final test where the hero must use all they've learned."},
                {"name": "Return with the Elixir", "description": "Using the reward to improve the ordinary world."}
            ]
        },
        "save_the_cat": {
            "name": "Save the Cat Beat Sheet",
            "sections": [
                {"name": "Act 1", "description": "Setup and establishment of the protagonist's world."},
                {"name": "Act 2A", "description": "The protagonist adapts to the new situation and begins pursuing their goal."},
                {"name": "Act 2B", "description": "Complications arise and the protagonist hits their lowest point."},
                {"name": "Act 3", "description": "The protagonist makes a final push to achieve their goal."}
            ],
            "points": [
                {"name": "Opening Image", "description": "A snapshot of the protagonist's life before the story begins."},
                {"name": "Theme Stated", "description": "A statement that hints at the story's theme."},
                {"name": "Setup", "description": "Introduction to the protagonist's world and problems."},
                {"name": "Catalyst", "description": "The inciting incident that disrupts the status quo."},
                {"name": "Debate", "description": "The protagonist wrestles with whether to pursue the journey."},
                {"name": "Break into Two", "description": "The protagonist decides to pursue the journey."},
                {"name": "B Story", "description": "Introduction of a secondary storyline, often a love story."},
                {"name": "Fun and Games", "description": "The protagonist explores the new world and situation."},
                {"name": "Midpoint", "description": "A major turning point that raises the stakes."},
                {"name": "Bad Guys Close In", "description": "Enemies regroup and obstacles intensify."},
                {"name": "All Is Lost", "description": "The protagonist hits their lowest point."},
                {"name": "Dark Night of the Soul", "description": "The protagonist's moment of despair."},
                {"name": "Break into Three", "description": "The protagonist finds a solution and begins the final push."},
                {"name": "Finale", "description": "The protagonist proves they've changed and achieves their goal."},
                {"name": "Final Image", "description": "A snapshot of the protagonist's new life."}
            ]
        }
    }
    
    # Check if the requested structure type exists
    if structure_type not in templates:
        return {"error": f"Structure type '{structure_type}' not found. Available types: {', '.join(templates.keys())}"}
    
    # Create and save the structure
    structure_id = str(uuid.uuid4())
    structure = StoryStructure(
        structure_id=structure_id,
        project_id=project_id,
        name=templates[structure_type]["name"],
        structure_type=structure_type,
        sections=templates[structure_type]["sections"],
        points=templates[structure_type]["points"]
    )
    
    structures_collection.insert_one(structure.to_dict())
    
    return structure.to_dict()

@tool
def plot_conflict_generator(project_id: str, conflict_type: str = "external", 
                          characters: List[str] = None) -> Dict[str, Any]:
    """Create compelling story tensions.
    
    Args:
        project_id: The project ID.
        conflict_type: Type of conflict (external, internal, interpersonal).
        characters: List of character IDs involved.
        
    Returns:
        Generated conflict information.
    """
    plot_collection = get_collection(COLLECTIONS["plot_elements"])
    
    # Conflict type templates
    conflict_templates = {
        "external": {
            "description": "Character vs. External Force (nature, society, technology, etc.)",
            "examples": [
                "Character battles against a natural disaster",
                "Character faces discrimination from society",
                "Character struggles against a corrupt system"
            ],
            "stakes_examples": [
                "Physical survival",
                "Community welfare",
                "Justice and truth"
            ]
        },
        "internal": {
            "description": "Character vs. Self (inner demons, flaws, desires)",
            "examples": [
                "Character struggles with addiction",
                "Character must overcome their fear",
                "Character confronts their own moral failings"
            ],
            "stakes_examples": [
                "Mental health",
                "Self-acceptance",
                "Moral integrity"
            ]
        },
        "interpersonal": {
            "description": "Character vs. Character (relationships, opposing goals)",
            "examples": [
                "Rivalry between colleagues",
                "Family members with opposing values",
                "Friends competing for the same goal"
            ],
            "stakes_examples": [
                "Valued relationship",
                "Status or position",
                "Trust and loyalty"
            ]
        }
    }
    
    if conflict_type not in conflict_templates:
        return {"error": f"Conflict type '{conflict_type}' not found. Available types: {', '.join(conflict_templates.keys())}"}
    
    # Generate a conflict based on the template
    template = conflict_templates[conflict_type]
    
    conflict = {
        "element_id": str(uuid.uuid4()),
        "project_id": project_id,
        "element_type": "conflict",
        "conflict_type": conflict_type,
        "description": template["description"],
        "examples": template["examples"],
        "stakes_examples": template["stakes_examples"],
        "characters_involved": characters or [],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    plot_collection.insert_one(conflict)
    
    return conflict

@tool
def world_encyclopedia(action: str, project_id: str, entry: Dict[str, Any] = None) -> Dict[str, Any]:
    """Document world elements systematically.
    
    Args:
        action: Action to perform (create, read, update, list, delete).
        project_id: The project ID.
        entry: Entry data for create/update actions.
        
    Returns:
        Action result.
    """
    world_collection = get_collection(COLLECTIONS["world_building"])
    
    if action == "create":
        if not entry:
            return {"error": "Entry data is required for create action"}
        
        # Create new world-building entry
        entry_id = str(uuid.uuid4())
        world_entry = WorldBuildingEntry(
            entry_id=entry_id,
            project_id=project_id,
            name=entry.get("name", "Unnamed Entry"),
            category=entry.get("category", "location"),
            description=entry.get("description", ""),
            properties=entry.get("properties", {}),
            relationships=entry.get("relationships", []),
            references=entry.get("references", []),
            notes=entry.get("notes")
        )
        
        world_collection.insert_one(world_entry.to_dict())
        
        return {"action": action, "status": "success", "entry_id": entry_id}
    
    elif action == "read":
        if not entry or "entry_id" not in entry:
            return {"error": "Entry ID is required for read action"}
        
        entry_id = entry["entry_id"]
        world_entry = world_collection.find_one({"entry_id": entry_id, "project_id": project_id})
        
        if not world_entry:
            return {"error": f"Entry {entry_id} not found"}
        
        return {"action": action, "status": "success", "entry": world_entry}
    
    elif action == "update":
        if not entry or "entry_id" not in entry:
            return {"error": "Entry ID is required for update action"}
        
        entry_id = entry["entry_id"]
        
        # Remove entry_id from update data
        update_data = entry.copy()
        update_data.pop("entry_id", None)
        update_data["updated_at"] = datetime.utcnow()
        
        result = world_collection.update_one(
            {"entry_id": entry_id, "project_id": project_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            return {"error": f"Entry {entry_id} not found"}
        
        return {"action": action, "status": "success", "entry_id": entry_id}
    
    elif action == "list":
        category = entry.get("category") if entry else None
        
        query = {"project_id": project_id}
        if category:
            query["category"] = category
        
        entries = list(world_collection.find(query, {"_id": 0}))
        
        return {"action": action, "status": "success", "entries": entries}
    
    elif action == "delete":
        if not entry or "entry_id" not in entry:
            return {"error": "Entry ID is required for delete action"}
        
        entry_id = entry["entry_id"]
        
        result = world_collection.delete_one({"entry_id": entry_id, "project_id": project_id})
        
        if result.deleted_count == 0:
            return {"error": f"Entry {entry_id} not found"}
        
        return {"action": action, "status": "success", "entry_id": entry_id}
    
    else:
        return {"error": f"Unknown action: {action}"}

@tool
def psychological_profile_builder(character_id: str, project_id: str, profile_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create detailed character psychologies.
    
    Args:
        character_id: The character ID.
        project_id: The project ID.
        profile_data: Psychological profile data.
        
    Returns:
        Character profile information.
    """
    characters_collection = get_collection(COLLECTIONS["characters"])
    
    # Check if character exists
    character = characters_collection.find_one({"character_id": character_id, "project_id": project_id})
    
    if not character:
        return {"error": f"Character {character_id} not found"}
    
    if profile_data:
        # Update character with psychological profile
        psychological_profile = {
            "personality_traits": profile_data.get("personality_traits", []),
            "core_values": profile_data.get("core_values", []),
            "fears": profile_data.get("fears", []),
            "desires": profile_data.get("desires", []),
            "contradictions": profile_data.get("contradictions", []),
            "coping_mechanisms": profile_data.get("coping_mechanisms", []),
            "blind_spots": profile_data.get("blind_spots", []),
            "growth_potential": profile_data.get("growth_potential", {})
        }
        
        characters_collection.update_one(
            {"character_id": character_id, "project_id": project_id},
            {"$set": {
                "psychological_profile": psychological_profile,
                "updated_at": datetime.utcnow()
            }}
        )
        
        character["psychological_profile"] = psychological_profile
    
    # Return the character info with psychological profile
    return {
        "character_id": character["character_id"],
        "name": character.get("name", "Unnamed Character"),
        "psychological_profile": character.get("psychological_profile", {})
    }

# Register tools
from storybook.tools import tool_registry
for tool_func in [
    story_structure_template,
    plot_conflict_generator,
    world_encyclopedia,
    psychological_profile_builder,
]:
    tool_registry.register_tool(tool_func, "development")