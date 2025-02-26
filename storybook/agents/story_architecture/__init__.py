from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from storybook.utils.state import NovelState, PlotPoint
from storybook.config import Config


class StoryStructure(BaseModel):
    """Story structure definition."""

    structure_type: str = Field(description="The type of story structure being used")
    act_breakdown: Dict[str, str] = Field(
        description="Breakdown of acts/sections with descriptions"
    )
    major_plot_points: List[Dict[str, str]] = Field(
        description="Major plot points with descriptions"
    )
    estimated_word_counts: Dict[str, int] = Field(
        description="Estimated word counts for each section"
    )
    pacing_notes: Dict[str, str] = Field(description="Pacing notes for each section")


class ChapterOutline(BaseModel):
    """Outline for a single chapter."""

    chapter_number: int = Field(description="Chapter number")
    title: str = Field(description="Chapter title")
    pov_character: str = Field(description="POV character for this chapter")
    setting: str = Field(description="Primary setting for this chapter")
    summary: str = Field(description="Brief summary of chapter events")
    opening_hook: str = Field(description="Opening hook description")
    closing_hook: str = Field(description="Closing hook or cliffhanger")
    character_development: Dict[str, str] = Field(
        description="Character development points"
    )
    plot_advancement: List[str] = Field(
        description="How the plot advances in this chapter"
    )
    themes_explored: List[str] = Field(description="Themes explored in this chapter")
    estimated_word_count: int = Field(
        description="Estimated word count for this chapter"
    )


class StructureSpecialistAgent:
    """Structure Specialist Agent that designs the novel's structure."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "StructureSpecialist"

    def recommend_structure(self, state: NovelState) -> Dict[str, Any]:
        """Recommend an appropriate story structure based on genre and premise."""
        prompt = PromptTemplate(
            template="""You are a story structure specialist recommending the optimal narrative structure for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Subgenres: {subgenres}
Premise: {premise}
Themes: {themes}

Based on these details, recommend 3 potential story structures that would work well for this novel.
For each structure, provide:
1. Name and brief description of the structure
2. Why it's appropriate for this genre and premise
3. How it would support the themes
4. Examples of successful novels with similar premises that used this structure
5. Potential advantages and disadvantages

After providing the options, recommend the single best structure and explain why it's the optimal choice.
""",
            input_variables=["project_name", "genre", "subgenres", "premise", "themes"],
        )

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                subgenres=", ".join(state.subgenres),
                premise=state.premise,
                themes=", ".join(state.themes),
            )
        )

        return {
            "structure_recommendations": response.content,
            "recommended_structure": self._extract_recommended_structure(
                response.content
            ),
        }

    def _extract_recommended_structure(self, text: str) -> str:
        """Extract the recommended structure from the response."""
        # This is a simplified method - in a real implementation, we would use more robust extraction
        if "recommend" in text.lower() and "structure" in text.lower():
            lines = text.split("\n")
            for line in lines:
                if "recommend" in line.lower() and "structure" in line.lower():
                    return line
        return "Three-Act Structure"  # Default fallback

    def design_structure(
        self, state: NovelState, structure_type: str
    ) -> StoryStructure:
        """Design a detailed story structure based on the recommended structure type."""
        parser = PydanticOutputParser(pydantic_object=StoryStructure)

        prompt = PromptTemplate(
            template="""You are a story structure specialist designing a detailed {structure_type} for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Themes: {themes}
Target Word Count: {target_word_count}

Create a detailed {structure_type} for this novel that includes:
1. Breakdown of acts/sections with descriptions of what should happen in each
2. All major plot points with descriptions
3. Estimated word counts for each section
4. Pacing notes for each section (where to speed up/slow down)

The structure should support the premise and themes while adhering to genre expectations.

{format_instructions}
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=[
                "structure_type",
                "project_name",
                "genre",
                "premise",
                "themes",
                "target_word_count",
            ],
        )

        response = self.llm.invoke(
            prompt.format(
                structure_type=structure_type,
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
                target_word_count=state.target_word_count,
            )
        )

        try:
            return parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing story structure: {e}")
            # Return a basic fallback structure
            return StoryStructure(
                structure_type=structure_type,
                act_breakdown={
                    "Act 1": "Setup and introduction",
                    "Act 2": "Confrontation and development",
                    "Act 3": "Resolution and conclusion",
                },
                major_plot_points=[
                    {
                        "name": "Inciting Incident",
                        "description": "Event that sets story in motion",
                    },
                    {"name": "Midpoint", "description": "Major shift in the middle"},
                    {"name": "Climax", "description": "Final confrontation"},
                ],
                estimated_word_counts={
                    "Act 1": state.target_word_count // 4,
                    "Act 2": state.target_word_count // 2,
                    "Act 3": state.target_word_count // 4,
                },
                pacing_notes={
                    "Act 1": "Start with hook, gradually build",
                    "Act 2": "Intensify conflicts, escalate stakes",
                    "Act 3": "Accelerate to climax, then denouement",
                },
            )


class PlotDevelopmentAgent:
    """Plot Development Agent that creates detailed plot outlines."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "PlotDevelopment"

    def develop_plot_points(
        self, state: NovelState, structure: StoryStructure
    ) -> List[PlotPoint]:
        """Develop detailed plot points based on the story structure."""
        prompt = PromptTemplate(
            template="""You are a plot developer creating detailed plot points for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Structure Type: {structure_type}

Structure Breakdown:
{act_breakdown}

Major Plot Points:
{major_plot_points}

Develop 15-20 detailed plot points that trace the complete story arc from beginning to end. 
Each plot point should include:
1. A specific title for the event
2. A detailed description of what happens
3. Which characters are involved
4. Where in the story structure it falls (which act/section)
5. The tension level (1-10)

Make sure the plot points collectively:
- Create a cohesive story that fulfills the premise
- Follow the provided structure
- Include all necessary components of a satisfying narrative
- Have a logical progression of cause and effect
- Build to a compelling climax
- Include both external plot events and character development moments

Format each plot point as:
TITLE:
DESCRIPTION:
CHARACTERS:
STORY POSITION:
TENSION:
""",
            input_variables=[
                "project_name",
                "genre",
                "premise",
                "structure_type",
                "act_breakdown",
                "major_plot_points",
            ],
        )

        # Format the structure information for the prompt
        act_breakdown_text = ""
        for act, description in structure.act_breakdown.items():
            act_breakdown_text += f"{act}: {description}\n"

        major_plot_points_text = ""
        for point in structure.major_plot_points:
            major_plot_points_text += f"{point['name']}: {point['description']}\n"

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                structure_type=structure.structure_type,
                act_breakdown=act_breakdown_text,
                major_plot_points=major_plot_points_text,
            )
        )

        # Parse the response into plot points
        plot_points = []
        current_point = {}
        current_field = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if ":" in line and line.split(":")[0] in [
                "TITLE",
                "DESCRIPTION",
                "CHARACTERS",
                "STORY POSITION",
                "TENSION",
            ]:
                current_field = line.split(":")[0]
                value = line.split(":", 1)[1].strip()
                current_point[current_field] = value

                # If we've completed a plot point, add it to the list
                if all(
                    field in current_point
                    for field in ["TITLE", "DESCRIPTION", "CHARACTERS", "TENSION"]
                ):
                    try:
                        tension = float(
                            current_point.get("TENSION", "5").replace("/10", "").strip()
                        )
                    except:
                        tension = 5.0

                    plot_point = PlotPoint(
                        title=current_point["TITLE"],
                        description=current_point["DESCRIPTION"],
                        characters_involved=current_point["CHARACTERS"].split(","),
                        tension_level=tension / 10.0,  # Normalize to 0-1
                        resolution_status=False,
                    )
                    plot_points.append(plot_point)
                    current_point = {}
            elif current_field and line:
                # Continue previous field
                current_point[current_field] += " " + line

        return plot_points

    def create_chapter_outlines(
        self, state: NovelState, plot_points: List[PlotPoint]
    ) -> List[ChapterOutline]:
        """Create chapter outlines based on the plot points."""
        parser = PydanticOutputParser(pydantic_object=ChapterOutline)

        prompt = PromptTemplate(
            template="""You are a novel outliner creating detailed chapter outlines based on plot points.

Novel Details:
Title: {project_name}
Genre: {genre}
Target Word Count: {target_word_count}

Main Characters:
{characters}

Plot Points:
{plot_points}

Create a detailed outline for {chapter_count} chapters that collectively tell this story.
Each chapter should:
- Advance the plot through one or more key events
- Develop characters
- Build toward the next major plot point
- Have a compelling opening and closing hook

For each chapter, provide:
{format_instructions}

Ensure the chapters collectively form a cohesive novel with proper pacing, rising action, and a satisfying arc.
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=[
                "project_name",
                "genre",
                "target_word_count",
                "characters",
                "plot_points",
                "chapter_count",
            ],
        )

        # Estimate appropriate chapter count based on target word count
        # Standard novel has ~2500-3000 words per chapter
        chapter_count = max(12, min(40, state.target_word_count // 2500))

        # Format character information
        characters_text = ""
        for name, character in state.characters.items():
            characters_text += (
                f"{name}: {character.role} - {character.background[:100]}...\n"
            )

        # If no characters yet, provide placeholder
        if not characters_text:
            characters_text = "Main character: Protagonist\nSupporting character: Ally\nAntagonist: Opposing force"

        # Format plot points
        plot_points_text = ""
        for i, point in enumerate(plot_points):
            plot_points_text += f"{i+1}. {point.title}: {point.description[:150]}...\n"

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                target_word_count=state.target_word_count,
                characters=characters_text,
                plot_points=plot_points_text,
                chapter_count=chapter_count,
            )
        )

        # In a real implementation, we would properly parse multiple chapter outlines
        # This is simplified for demonstration
        try:
            chapter_outline = parser.parse(response.content)
            return [chapter_outline]
        except Exception as e:
            print(f"Error parsing chapter outline: {e}")
            # Return a placeholder chapter outline
            return [
                ChapterOutline(
                    chapter_number=1,
                    title="Chapter 1",
                    pov_character="Protagonist",
                    setting="Main setting",
                    summary="Introduction to the main character and setting",
                    opening_hook="Intriguing situation that draws reader in",
                    closing_hook="Question or situation that makes reader want to continue",
                    character_development={
                        "Protagonist": "Initial character state established"
                    },
                    plot_advancement=["Introduction of main conflict"],
                    themes_explored=["Main theme introduction"],
                    estimated_word_count=2500,
                )
            ]


class WorldBuildingAgent:
    """World Building Agent that develops setting rules, maps, and systems."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "WorldBuilding"

    def develop_setting(self, state: NovelState) -> Dict[str, Any]:
        """Develop a detailed setting for the novel."""
        prompt = PromptTemplate(
            template="""You are a world-building specialist developing a detailed setting for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Themes: {themes}

Create a comprehensive setting for this novel that includes:

1. PRIMARY LOCATIONS
   Develop 3-5 key locations where the story takes place. For each location, provide:
   - Name and description
   - Notable features and atmosphere
   - Cultural/historical significance
   - How it relates to the story themes

2. SETTING RULES AND SYSTEMS
   Detail the rules that govern this world:
   - If fantasy/sci-fi: magic systems, technology, or alternate physics
   - If contemporary/historical: social hierarchies, economic systems, cultural norms
   - Any constraints or limitations that impact characters

3. SENSORY LANDSCAPE
   Create a sensory palette for the world:
   - Visual elements (colors, architecture, natural features)
   - Sounds and ambient noise
   - Smells and tastes unique to this world
   - Tactile elements (textures, climate, physical sensations)

4. CULTURAL ELEMENTS
   Develop cultural details that enrich the setting:
   - Customs, traditions, and rituals
   - Languages or communication styles
   - Art, music, and entertainment
   - Food and dining practices

The setting should feel cohesive, support the story themes, and provide rich opportunities for conflict and character development.
""",
            input_variables=["project_name", "genre", "premise", "themes"],
        )

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
            )
        )

        # Extract the sections from the response
        # This is a simplified extraction - a real implementation would be more robust
        sections = {
            "locations": "",
            "rules_systems": "",
            "sensory_landscape": "",
            "cultural_elements": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "PRIMARY LOCATIONS" in line:
                current_section = "locations"
                continue
            elif "SETTING RULES AND SYSTEMS" in line:
                current_section = "rules_systems"
                continue
            elif "SENSORY LANDSCAPE" in line:
                current_section = "sensory_landscape"
                continue
            elif "CULTURAL ELEMENTS" in line:
                current_section = "cultural_elements"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        return {
            "world_building_document": response.content,
            "locations": sections["locations"].strip(),
            "rules_systems": sections["rules_systems"].strip(),
            "sensory_landscape": sections["sensory_landscape"].strip(),
            "cultural_elements": sections["cultural_elements"].strip(),
        }

    def create_setting_bible(
        self, state: NovelState, world_building: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create a comprehensive setting bible based on the world building."""
        prompt = PromptTemplate(
            template="""You are a world-building specialist creating a comprehensive setting bible for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}

Existing World Building Elements:
LOCATIONS:
{locations}

RULES AND SYSTEMS:
{rules_systems}

SENSORY LANDSCAPE:
{sensory_landscape}

CULTURAL ELEMENTS:
{cultural_elements}

Expand these elements into a complete setting bible that provides all the information a writer would need to create a consistent, immersive world. The bible should include:

1. EXPANDED LOCATIONS
   - Detailed maps and spatial relationships between locations
   - History and development of key sites
   - How different characters/groups perceive these locations

2. SYSTEMS IN DEPTH
   - Complete explanation of how systems (magic, technology, social, etc.) function
   - Limitations, costs, and consequences of these systems
   - How these systems affect daily life and society

3. TIMELINE
   - Historical timeline of important events that shaped this world
   - How historical events influence the present situation
   - Reference points that characters might mention

4. SOCIAL STRUCTURE
   - Class or group divisions
   - Power dynamics and hierarchies
   - Economic systems and their impact on characters

5. FLORA, FAUNA, AND RESOURCES
   - Distinctive plants, animals, or resources
   - How these elements are used or interact with society
   - Any unique natural phenomena

Format this as a comprehensive reference document that could be provided to multiple writers working in this world.
""",
            input_variables=[
                "project_name",
                "genre",
                "locations",
                "rules_systems",
                "sensory_landscape",
                "cultural_elements",
            ],
        )

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                locations=world_building["locations"],
                rules_systems=world_building["rules_systems"],
                sensory_landscape=world_building["sensory_landscape"],
                cultural_elements=world_building["cultural_elements"],
            )
        )

        # For storage in the state, extract the sections
        sections = {
            "expanded_locations": "",
            "systems_in_depth": "",
            "timeline": "",
            "social_structure": "",
            "flora_fauna_resources": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "EXPANDED LOCATIONS" in line:
                current_section = "expanded_locations"
                continue
            elif "SYSTEMS IN DEPTH" in line:
                current_section = "systems_in_depth"
                continue
            elif "TIMELINE" in line:
                current_section = "timeline"
                continue
            elif "SOCIAL STRUCTURE" in line:
                current_section = "social_structure"
                continue
            elif "FLORA, FAUNA, AND RESOURCES" in line:
                current_section = "flora_fauna_resources"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        return {"setting_bible": response.content, "sections": sections}
