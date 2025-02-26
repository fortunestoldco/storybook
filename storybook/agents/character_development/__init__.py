# agents/character_development/__init__.py
import json
from typing import Any, Dict, List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from storybook.config import Config
from storybook.utils.state import Character, NovelState


class CharacterPersonality(BaseModel):
    """Detailed personality model for a character."""

    traits: Dict[str, float] = Field(
        description="Personality traits with 0-1 scale ratings"
    )
    cognitive_style: str = Field(
        description="How the character thinks and processes information"
    )
    emotional_patterns: Dict[str, str] = Field(
        description="Emotional response patterns"
    )
    motivation_drivers: List[str] = Field(description="Primary motivation drivers")
    values: List[Dict[str, str]] = Field(description="Core values with descriptions")
    shadow_traits: List[str] = Field(description="Hidden or repressed traits")
    stress_responses: Dict[str, str] = Field(
        description="How the character responds to different stressors"
    )
    growth_potential: Dict[str, str] = Field(
        description="Areas where character could grow or change"
    )


class CharacterArc(BaseModel):
    """Character arc model."""

    starting_state: Dict[str, Any] = Field(
        description="Character's state at the beginning"
    )
    key_moments: List[Dict[str, str]] = Field(
        description="Key transformational moments"
    )
    ending_state: Dict[str, Any] = Field(description="Character's state at the end")
    internal_journey: str = Field(description="Description of internal transformation")
    external_manifestation: str = Field(
        description="How internal changes manifest externally"
    )
    theme_connection: str = Field(description="How this arc connects to novel themes")


class DialoguePattern(BaseModel):
    """Character-specific dialogue patterns."""

    speech_style: str = Field(description="Overall speech style description")
    vocabulary_level: str = Field(description="Level and type of vocabulary used")
    sentence_structure: str = Field(description="Typical sentence structures")
    verbal_tics: List[str] = Field(description="Repeated phrases or verbal tics")
    topics: List[str] = Field(description="Topics this character gravitates toward")
    avoidances: List[str] = Field(
        description="Topics or speech patterns this character avoids"
    )
    emotion_indicators: Dict[str, List[str]] = Field(
        description="How different emotions manifest in speech"
    )


class CharacterResearchAgent:
    """Character Research Agent that creates psychological profiles for characters."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "CharacterResearch"

    def create_character_profile(
        self, name: str, role: str, genre: str, themes: List[str]
    ) -> Character:
        """Create a comprehensive character profile."""
        # First, create the personality profile
        personality = self._create_personality_profile(name, role, genre, themes)

        # Then create the character background
        background = self._create_character_background(name, role, personality, genre)

        # Create character motivations
        motivations = self._extract_motivations(personality)

        # Initial relationship map (will be expanded by the relationship mapper)
        relationships = {"Other characters": "To be developed"}

        # Create initial character arc
        arc = self._create_initial_arc(name, role, personality, themes)

        # Create dialogue patterns
        dialogue_patterns = self._create_dialogue_patterns(
            name, personality, background
        )

        # Identify and subvert tropes
        tropes, subversions = self._identify_and_subvert_tropes(
            role, genre, personality
        )

        return Character(
            name=name,
            role=role,
            background=background,
            personality=personality.model_dump(),
            motivations=motivations,
            arc=arc.model_dump(),
            relationships=relationships,
            dialogue_patterns=dialogue_patterns.model_dump(),
            tropes=tropes,
            trope_subversions=subversions,
        )

    def _create_personality_profile(
        self, name: str, role: str, genre: str, themes: List[str]
    ) -> CharacterPersonality:
        """Create a detailed psychological profile for a character."""
        parser = PydanticOutputParser(pydantic_object=CharacterPersonality)

        prompt = PromptTemplate(
            template="""You are a character psychologist creating a detailed personality profile for a fictional character.

Character Information:
Name: {name}
Role in Story: {role}
Genre: {genre}
Themes the character will explore: {themes}

Create a psychologically complex and realistic personality profile for this character.
The profile should:
- Have depth and nuance
- Avoid one-dimensional or stereotypical traits
- Contain productive internal conflicts
- Support the themes of the story
- Be appropriate for the character's role and genre

{format_instructions}
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=["name", "role", "genre", "themes"],
        )

        response = self.llm.invoke(
            prompt.format(name=name, role=role, genre=genre, themes=", ".join(themes))
        )

        try:
            return parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing character personality: {e}")
            # Return a basic personality profile as fallback
            return CharacterPersonality(
                traits={"determination": 0.8, "empathy": 0.7},
                cognitive_style="Analytical with creative problem-solving tendencies",
                emotional_patterns={
                    "stress": "Becomes withdrawn",
                    "joy": "Expressive and generous",
                },
                motivation_drivers=["Achievement", "Connection"],
                values=[
                    {"name": "Loyalty", "description": "Values commitment to others"}
                ],
                shadow_traits=["Self-doubt", "Fear of abandonment"],
                stress_responses={
                    "conflict": "Becomes defensive",
                    "failure": "Self-critical",
                },
                growth_potential={
                    "empathy": "Could develop greater understanding of others"
                },
            )

    def _create_character_background(
        self, name: str, role: str, personality: CharacterPersonality, genre: str
    ) -> str:
        """Create a character background that explains their personality development."""
        prompt = PromptTemplate(
            template="""You are creating a detailed background for a fictional character that explains how they developed their current personality.

Character Information:
Name: {name}
Role in Story: {role}
Genre: {genre}

Personality Profile:
{personality}

Create a compelling background for this character that:
1. Explains the formative experiences that shaped their personality traits
2. Includes key moments from childhood, adolescence, and early adulthood
3. Establishes relationships with family, friends, and mentors
4. Explains any significant traumas or triumphs that affected their development
5. Aligns with their current personality and motivations
6. Would be believable and appropriate for the genre

Write this as a detailed narrative that gives a clear understanding of how this character became who they are.
""",
            input_variables=["name", "role", "genre", "personality"],
        )

        response = self.llm.invoke(
            prompt.format(
                name=name,
                role=role,
                genre=genre,
                personality=json.dumps(personality.model_dump(), indent=2),
            )
        )

        return response.content

    def _extract_motivations(self, personality: CharacterPersonality) -> List[str]:
        """Extract and expand upon character motivations."""
        prompt = PromptTemplate(
            template="""Based on this character personality profile:

{personality}

Extract and expand upon this character's core motivations. What truly drives them? What do they want consciously and unconsciously?

List 5-7 specific motivations, from most to least important, with a brief explanation of each.
""",
            input_variables=["personality"],
        )

        response = self.llm.invoke(
            prompt.format(personality=json.dumps(personality.model_dump(), indent=2))
        )

        # Extract motivations from the response
        motivations = []
        for line in response.content.split("\n"):
            line = line.strip()
            if (
                line
                and (":" in line or "." in line)
                and any(char.isdigit() for char in line[:2])
            ):
                # Clean up the line to remove numbers/bullets and get just the motivation
                cleaned = line.split(":", 1)[-1].split(".", 1)[-1].strip()
                if cleaned:
                    motivations.append(cleaned)

        # If extraction failed, use personality's motivation drivers
        if not motivations and personality.motivation_drivers:
            return personality.motivation_drivers

        return motivations[:7]  # Limit to 7 max

    def _create_initial_arc(
        self, name: str, role: str, personality: CharacterPersonality, themes: List[str]
    ) -> CharacterArc:
        """Create an initial character arc."""
        parser = PydanticOutputParser(pydantic_object=CharacterArc)

        prompt = PromptTemplate(
            template="""You are designing a character arc for a fictional character.

Character Information:
Name: {name}
Role in Story: {role}
Themes to explore: {themes}

Personality Profile:
{personality}

Design a compelling character arc that:
1. Has a clear starting state based on their current personality
2. Includes key transformational moments that would challenge and change them
3. Defines a meaningful ending state that represents growth but stays true to their core
4. Connects directly to the story's themes
5. Includes both internal transformation and external manifestation of that change

{format_instructions}
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=["name", "role", "themes", "personality"],
        )

        response = self.llm.invoke(
            prompt.format(
                name=name,
                role=role,
                themes=", ".join(themes),
                personality=json.dumps(personality.model_dump(), indent=2),
            )
        )

        try:
            return parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing character arc: {e}")
            # Return a basic arc as fallback
            return CharacterArc(
                starting_state={"state": "Initial character state"},
                key_moments=[
                    {
                        "moment": "Challenge",
                        "description": "Character faces a significant challenge",
                    }
                ],
                ending_state={"state": "Character growth"},
                internal_journey="Character overcomes internal obstacles",
                external_manifestation="Changes are shown through actions",
                theme_connection="Arc connects to main themes",
            )

    def _create_dialogue_patterns(
        self, name: str, personality: CharacterPersonality, background: str
    ) -> DialoguePattern:
        """Create distinctive dialogue patterns for a character."""
        parser = PydanticOutputParser(pydantic_object=DialoguePattern)

        prompt = PromptTemplate(
            template="""You are a dialogue specialist creating distinctive speech patterns for a fictional character.

Character Information:
Name: {name}

Personality Profile:
{personality}

Background Excerpt:
{background}

Create distinctive dialogue patterns for this character that:
1. Reflect their personality, background, and education level
2. Feel authentic and consistent
3. Help distinguish them from other characters
4. Include specific verbal tics, favorite phrases, or speech habits
5. Show how their speech changes under different emotional states

{format_instructions}
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=["name", "personality", "background"],
        )

        response = self.llm.invoke(
            prompt.format(
                name=name,
                personality=json.dumps(personality.model_dump(), indent=2),
                background=background[
                    :1000
                ],  # Truncate background to avoid token limits
            )
        )

        try:
            return parser.parse(response.content)
        except Exception as e:
            print(f"Error parsing dialogue patterns: {e}")
            # Return basic dialogue patterns as fallback
            return DialoguePattern(
                speech_style="Straightforward and direct",
                vocabulary_level="Average with occasional specialized terms",
                sentence_structure="Typically uses complete sentences of medium length",
                verbal_tics=["Well...", "Look,"],
                topics=["Practical matters", "Immediate concerns"],
                avoidances=["Overly emotional topics"],
                emotion_indicators={
                    "angry": ["Short sentences", "Increased volume"],
                    "happy": ["More animated speech", "Laughs frequently"],
                },
            )

    def _identify_and_subvert_tropes(
        self, role: str, genre: str, personality: CharacterPersonality
    ) -> tuple[List[str], List[str]]:
        """Identify character tropes and suggest subversions."""
        prompt = PromptTemplate(
            template="""You are a literary analyst specializing in character tropes and their subversion.

Character Role: {role}
Genre: {genre}
Personality Traits:
{personality_traits}

First, identify 3-5 common character tropes that this type of character often falls into within this genre.
Then, for each identified trope, suggest a specific subversion that would make the character more unique and three-dimensional.

Format your response as:
TROPES:
1. [Trope Name]: [Brief description]
2. [Trope Name]: [Brief description]
...

SUBVERSIONS:
1. [Subversion of first trope]: [How to implement this subversion]
2. [Subversion of second trope]: [How to implement this subversion]
...
""",
            input_variables=["role", "genre", "personality_traits"],
        )

        # Extract just the traits for simplicity
        traits_text = ""
        for trait, value in personality.traits.items():
            traits_text += f"- {trait}: {value}\n"

        response = self.llm.invoke(
            prompt.format(role=role, genre=genre, personality_traits=traits_text)
        )

        # Parse the response to extract tropes and subversions
        tropes = []
        subversions = []
        current_section = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if "TROPES:" in line:
                current_section = "tropes"
                continue
            elif "SUBVERSIONS:" in line:
                current_section = "subversions"
                continue

            if current_section == "tropes" and any(char.isdigit() for char in line[:2]):
                # Extract trope name from line like "1. [Trope Name]: [Description]"
                if ":" in line:
                    trope = line.split(":", 1)[0].split(".", 1)[-1].strip()
                    if "[" in trope and "]" in trope:
                        trope = trope.split("[", 1)[1].split("]", 1)[0].strip()
                    tropes.append(trope)

            elif current_section == "subversions" and any(
                char.isdigit() for char in line[:2]
            ):
                # Extract subversion from line like "1. [Subversion]: [Implementation]"
                if ":" in line:
                    subversion = line.split(":", 1)[0].split(".", 1)[-1].strip()
                    if "[" in subversion and "]" in subversion:
                        subversion = (
                            subversion.split("[", 1)[1].split("]", 1)[0].strip()
                        )
                    subversions.append(subversion)

        return tropes, subversions


class CharacterArcDesignerAgent:
    """Character Arc Designer Agent that plots character growth trajectories."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "CharacterArcDesigner"

    def refine_character_arc(
        self, character: Character, plot_points: List[Any], themes: List[str]
    ) -> Dict[str, Any]:
        """Refine a character's arc based on plot points and themes."""
        prompt = PromptTemplate(
            template="""You are a character arc designer refining a character's transformational journey.

Character Information:
Name: {name}
Role: {role}
Background: {background}
Initial Arc: {initial_arc}

Plot Points:
{plot_points}

Themes:
{themes}

Refine this character's arc to integrate seamlessly with the plot points and explore the story's themes.
Your refined arc should:

1. MAP ARC TO PLOT
   Identify specific plot points where the character experiences growth or change
   Explain how each plot point affects their internal journey

2. THEMATIC EXPLORATION
   Show how this character specifically explores each theme
   Create internal conflicts that reflect thematic tensions

3. EMOTIONAL TRANSFORMATION
   Define clear emotional states at beginning, middle, and end
   Create a nuanced emotional journey with setbacks and progress

4. EXTERNAL MANIFESTATION
   Show how internal changes manifest in behavior and decisions
   Create "showing" moments that demonstrate character growth

5. SYMBOLIC ELEMENTS
   Suggest symbolic objects, settings, or recurring motifs tied to this character's journey

6. ARC COMPLETION
   Define what "success" looks like for this character
   Explain how their final state relates to but transcends their starting state

Format your response as a complete character arc development document with clear sections.
""",
            input_variables=[
                "name",
                "role",
                "background",
                "initial_arc",
                "plot_points",
                "themes",
            ],
        )

        # Format plot points for the prompt
        plot_points_text = ""
        for i, point in enumerate(
            plot_points[:10]
        ):  # Limit to first 10 to avoid token limits
            if hasattr(point, "model_dump"):
                point_data = point.model_dump()
            else:
                point_data = point

            if isinstance(point_data, dict):
                title = point_data.get("title", f"Point {i+1}")
                desc = point_data.get("description", "")
                plot_points_text += f"{i+1}. {title}: {desc[:100]}...\n"
            else:
                plot_points_text += f"{i+1}. Plot Point {i+1}\n"

        response = self.llm.invoke(
            prompt.format(
                name=character.name,
                role=character.role,
                background=character.background[:500]
                + "...",  # Truncate to avoid token limits
                initial_arc=json.dumps(character.arc, indent=2),
                plot_points=plot_points_text,
                themes=", ".join(themes),
            )
        )

        # Extract sections from the response
        sections = {
            "arc_plot_mapping": "",
            "thematic_exploration": "",
            "emotional_transformation": "",
            "external_manifestation": "",
            "symbolic_elements": "",
            "arc_completion": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "MAP ARC TO PLOT" in line or "1. MAP ARC TO PLOT" in line:
                current_section = "arc_plot_mapping"
                continue
            elif "THEMATIC EXPLORATION" in line or "2. THEMATIC EXPLORATION" in line:
                current_section = "thematic_exploration"
                continue
            elif (
                "EMOTIONAL TRANSFORMATION" in line
                or "3. EMOTIONAL TRANSFORMATION" in line
            ):
                current_section = "emotional_transformation"
                continue
            elif (
                "EXTERNAL MANIFESTATION" in line or "4. EXTERNAL MANIFESTATION" in line
            ):
                current_section = "external_manifestation"
                continue
            elif "SYMBOLIC ELEMENTS" in line or "5. SYMBOLIC ELEMENTS" in line:
                current_section = "symbolic_elements"
                continue
            elif "ARC COMPLETION" in line or "6. ARC COMPLETION" in line:
                current_section = "arc_completion"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        # Update the character arc with the refined information
        refined_arc = character.arc.copy()

        # Extract key moments from the arc_plot_mapping section
        key_moments = []
        for line in sections["arc_plot_mapping"].split("\n"):
            if ":" in line and any(char.isdigit() for char in line[:2]):
                moment = {
                    "moment": line.split(":", 1)[0].strip(),
                    "description": line.split(":", 1)[1].strip(),
                }
                key_moments.append(moment)

        if key_moments:
            refined_arc["key_moments"] = key_moments

        refined_arc["thematic_exploration"] = sections["thematic_exploration"].strip()
        refined_arc["emotional_journey"] = sections["emotional_transformation"].strip()
        refined_arc["external_manifestation"] = sections[
            "external_manifestation"
        ].strip()
        refined_arc["symbolic_elements"] = sections["symbolic_elements"].strip()
        refined_arc["arc_completion"] = sections["arc_completion"].strip()

        return {"refined_arc": refined_arc, "arc_document": response.content}

    def create_transformation_scenes(
        self, character: Character, refined_arc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create key transformation scenes for character development."""
        prompt = PromptTemplate(
            template="""You are a scene designer specializing in character development moments.

Character Information:
Name: {name}
Role: {role}
Personality: {personality_summary}

Character Arc:
{refined_arc}

Create 3-5 pivotal scenes that show this character's transformation. For each scene:

1. Identify where in the story it should occur (beginning, middle, end)
2. Describe the setting and situation
3. Explain what challenge or decision the character faces
4. Show how the character's response reveals their development
5. Describe specific dialogue, actions, and internal thoughts that demonstrate growth

Each scene should be a powerful "showing not telling" moment that clearly demonstrates character development.
Focus on emotional impact and authenticity to the character's personality.

Format each scene as:
SCENE TITLE:
STORY POSITION:
SETTING:
SITUATION:
CHARACTER RESPONSE:
KEY DIALOGUE/ACTIONS:
INTERNAL TRANSFORMATION:
""",
            input_variables=["name", "role", "personality_summary", "refined_arc"],
        )

        # Create a summary of personality for the prompt
        personality_traits = character.personality.get("traits", {})
        personality_summary = "Key traits: " + ", ".join(
            [f"{trait} ({value})" for trait, value in personality_traits.items()]
        )

        response = self.llm.invoke(
            prompt.format(
                name=character.name,
                role=character.role,
                personality_summary=personality_summary,
                refined_arc=json.dumps(refined_arc, indent=2),
            )
        )

        # Parse the response into scenes
        scenes = []
        current_scene = {}
        current_field = None

        for line in response.content.split("\n"):
            line = line.strip()
            if not line:
                continue

            if ":" in line and line.split(":")[0] in [
                "SCENE TITLE",
                "STORY POSITION",
                "SETTING",
                "SITUATION",
                "CHARACTER RESPONSE",
                "KEY DIALOGUE/ACTIONS",
                "INTERNAL TRANSFORMATION",
            ]:
                field = line.split(":")[0]
                value = line.split(":", 1)[1].strip()

                if field == "SCENE TITLE" and current_scene:
                    scenes.append(current_scene)
                    current_scene = {}

                current_scene[field] = value
            elif current_field and line:
                current_scene[current_field] += " " + line

        if current_scene:
            scenes.append(current_scene)

        return scenes


class CharacterRelationshipMapperAgent:
    """Character Relationship Mapper Agent that defines complex relationship dynamics."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "CharacterRelationshipMapper"

    def map_relationships(
        self, characters: Dict[str, Character]
    ) -> Dict[str, Dict[str, Any]]:
        """Map relationships between all characters."""
        if not characters:
            return {}

        character_summaries = {
            name: self._summarize_character(char) for name, char in characters.items()
        }

        prompt = PromptTemplate(
            template="""You are a character relationship designer creating complex, realistic relationships between characters.

Characters:
{character_summaries}

Create a relationship map between all these characters. For each pair of characters:

1. Define the nature of their relationship (family, friends, rivals, etc.)
2. Describe the history of their relationship
3. Identify power dynamics and dependency patterns
4. Note areas of conflict and harmony
5. Describe how they influence each other
6. Suggest how their relationship might evolve through the story

Focus on creating realistic, nuanced relationships with depth and complexity. Include both positive and negative aspects in each relationship.

Format your response as:
CHARACTER 1 -> CHARACTER 2:
[Detailed relationship description]

CHARACTER 2 -> CHARACTER 1:
[How CHARACTER 2 perceives and relates to CHARACTER 1, which may differ]
""",
            input_variables=["character_summaries"],
        )

        # Format character summaries for the prompt
        summaries_text = ""
        for name, summary in character_summaries.items():
            summaries_text += f"{name}:\n{summary}\n\n"

        response = self.llm.invoke(prompt.format(character_summaries=summaries_text))

        # Parse the response into a relationship map
        relationship_map = {}
        current_relationship = None
        relationship_text = ""

        for line in response.content.split("\n"):
            if "->" in line:
                # Save previous relationship if there was one
                if current_relationship and relationship_text:
                    char1, char2 = current_relationship
                    if char1 not in relationship_map:
                        relationship_map[char1] = {}
                    relationship_map[char1][char2] = relationship_text.strip()

                # Parse new relationship
                parts = line.split("->")
                char1 = parts[0].strip()
                char2 = parts[1].split(":")[0].strip()
                current_relationship = (char1, char2)
                relationship_text = line.split(":", 1)[1].strip() if ":" in line else ""
            elif current_relationship and line.strip():
                relationship_text += " " + line

        # Save the last relationship
        if current_relationship and relationship_text:
            char1, char2 = current_relationship
            if char1 not in relationship_map:
                relationship_map[char1] = {}
            relationship_map[char1][char2] = relationship_text.strip()

        return relationship_map

    def _summarize_character(self, character: Character) -> str:
        """Create a brief summary of a character for relationship mapping."""
        summary = f"Role: {character.role}\n"
        summary += f"Background: {character.background[:200]}...\n"

        if character.personality and "traits" in character.personality:
            traits = character.personality["traits"]
            summary += (
                "Key traits: " + ", ".join([f"{t}" for t in traits.keys()]) + "\n"
            )

        if character.motivations:
            summary += "Motivations: " + ", ".join(character.motivations[:3]) + "\n"

        return summary

    def update_character_relationships(
        self,
        characters: Dict[str, Character],
        relationship_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Character]:
        """Update all characters with their relationship information."""
        updated_characters = characters.copy()

        for char_name, relationships in relationship_map.items():
            if char_name in updated_characters:
                char = updated_characters[char_name]
                char.relationships = {
                    other: desc for other, desc in relationships.items()
                }

        return updated_characters


class DialogueSpecialistAgent:
    """Dialogue Specialist Agent that develops distinctive speech patterns for characters."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "DialogueSpecialist"

    def refine_dialogue_patterns(self, character: Character) -> Dict[str, Any]:
        """Refine a character's dialogue patterns with more depth and examples."""
        prompt = PromptTemplate(
            template="""You are a dialogue specialist refining speech patterns for a fictional character.

Character Information:
Name: {name}
Role: {role}
Background: {background}
Personality Traits: {personality_traits}

Current Dialogue Patterns:
{current_patterns}

Refine and expand these dialogue patterns to create truly distinctive speech for this character.
Include:

1. EXPANDED SPEECH STYLE
   Provide a more detailed analysis of their overall communication style
   
2. VOCABULARY CHOICES
   Specific word choices, complexity level, and terminology they would use
   
3. SENTENCE STRUCTURES
   Examples of sentence structures, lengths, and patterns this character employs
   
4. VERBAL QUIRKS
   Unique verbal tics, repeated phrases, or habitual expressions
   
5. DIALOGUE SAMPLES
   Create 5-7 sample dialogue lines showing how this character would speak in different situations:
   - When introducing themselves
   - When angry or upset
   - When being persuasive
   - When discussing something they're passionate about
   - When in a formal/professional setting
   - When speaking with someone they're close to
   
6. EMOTIONAL PATTERNS
   How their speech changes based on different emotional states
   
Format your response with clear sections for each category.
""",
            input_variables=[
                "name",
                "role",
                "background",
                "personality_traits",
                "current_patterns",
            ],
        )

        # Extract personality traits
        personality_traits = ""
        if character.personality and "traits" in character.personality:
            traits = character.personality["traits"]
            personality_traits = ", ".join([f"{t}: {v}" for t, v in traits.items()])

        # Format current dialogue patterns
        current_patterns = (
            json.dumps(character.dialogue_patterns, indent=2)
            if character.dialogue_patterns
            else "Not yet developed"
        )

        response = self.llm.invoke(
            prompt.format(
                name=character.name,
                role=character.role,
                background=character.background[:300] + "...",  # Truncate
                personality_traits=personality_traits,
                current_patterns=current_patterns,
            )
        )

        # Extract sections from the response
        sections = {
            "speech_style": "",
            "vocabulary": "",
            "sentence_structures": "",
            "verbal_quirks": "",
            "dialogue_samples": "",
            "emotional_patterns": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "EXPANDED SPEECH STYLE" in line or "1. EXPANDED SPEECH STYLE" in line:
                current_section = "speech_style"
                continue
            elif "VOCABULARY CHOICES" in line or "2. VOCABULARY CHOICES" in line:
                current_section = "vocabulary"
                continue
            elif "SENTENCE STRUCTURES" in line or "3. SENTENCE STRUCTURES" in line:
                current_section = "sentence_structures"
                continue
            elif "VERBAL QUIRKS" in line or "4. VERBAL QUIRKS" in line:
                current_section = "verbal_quirks"
                continue
            elif "DIALOGUE SAMPLES" in line or "5. DIALOGUE SAMPLES" in line:
                current_section = "dialogue_samples"
                continue
            elif "EMOTIONAL PATTERNS" in line or "6. EMOTIONAL PATTERNS" in line:
                current_section = "emotional_patterns"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        # Extract dialogue samples
        dialogue_samples = []
        for line in sections["dialogue_samples"].split("\n"):
            if line.strip() and ":" in line and line.count('"') >= 2:
                dialogue_samples.append(line.strip())

        # Create refined dialogue patterns
        refined_patterns = (
            character.dialogue_patterns.copy() if character.dialogue_patterns else {}
        )

        refined_patterns["expanded_speech_style"] = sections["speech_style"].strip()
        refined_patterns["vocabulary_details"] = sections["vocabulary"].strip()
        refined_patterns["sentence_structures"] = sections[
            "sentence_structures"
        ].strip()

        # Extract verbal quirks as a list
        verbal_quirks = []
        for line in sections["verbal_quirks"].split("\n"):
            if line.strip() and (
                "- " in line or "• " in line or any(char.isdigit() for char in line[:2])
            ):
                quirk = line.split("- ", 1)[-1].split("• ", 1)[-1]
                if any(char.isdigit() for char in line[:2]):
                    quirk = line.split(".", 1)[-1].strip()
                verbal_quirks.append(quirk)

        refined_patterns["verbal_quirks"] = (
            verbal_quirks if verbal_quirks else sections["verbal_quirks"].strip()
        )
        refined_patterns["dialogue_samples"] = dialogue_samples
        refined_patterns["emotional_speech_patterns"] = sections[
            "emotional_patterns"
        ].strip()

        return {
            "refined_dialogue_patterns": refined_patterns,
            "dialogue_document": response.content,
        }

    def generate_dialogue_examples(
        self, character: Character, situations: List[str]
    ) -> Dict[str, str]:
        """Generate dialogue examples for a character in different situations."""
        prompt = PromptTemplate(
            template="""You are an expert dialogue writer who perfectly captures character voice.

Character Information:
Name: {name}
Role: {role}
Background: {background}
Dialogue Patterns: {dialogue_patterns}

Generate authentic dialogue for this character in each of the following situations:
{situations}

For each situation:
1. Write 3-5 lines of dialogue that this character would realistically say
2. Include brief context notes about tone, delivery, or body language
3. Ensure the dialogue is consistent with the character's established patterns

Format each situation as:
SITUATION: [Situation description]
DIALOGUE:
"[Character's dialogue]" [Context note]
"[Character's dialogue]" [Context note]
...
""",
            input_variables=[
                "name",
                "role",
                "background",
                "dialogue_patterns",
                "situations",
            ],
        )

        # Format situations for the prompt
        situations_text = "\n".join([f"- {situation}" for situation in situations])

        # Format dialogue patterns for the prompt
        dialogue_patterns = (
            json.dumps(character.dialogue_patterns, indent=2)
            if character.dialogue_patterns
            else "Not yet developed"
        )

        response = self.llm.invoke(
            prompt.format(
                name=character.name,
                role=character.role,
                background=character.background[:300] + "...",  # Truncate
                dialogue_patterns=dialogue_patterns,
                situations=situations_text,
            )
        )

        # Parse the response into dialogue examples
        dialogue_examples = {}
        current_situation = None
        current_dialogue = []

        for line in response.content.split("\n"):
            if line.startswith("SITUATION:"):
                # Save previous situation if there was one
                if current_situation and current_dialogue:
                    dialogue_examples[current_situation] = "\n".join(current_dialogue)

                # Start new situation
                current_situation = line.split("SITUATION:")[1].strip()
                current_dialogue = []
            elif line.startswith("DIALOGUE:"):
                continue  # Skip the DIALOGUE: header
            elif current_situation and line.strip() and '"' in line:
                current_dialogue.append(line.strip())

        # Save the last situation
        if current_situation and current_dialogue:
            dialogue_examples[current_situation] = "\n".join(current_dialogue)

        return dialogue_examples
