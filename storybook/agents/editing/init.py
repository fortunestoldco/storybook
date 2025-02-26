# agents/editing/__init__.py
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import re

from storybook.utils.state import NovelState, Chapter
from storybook.config import storybookConfig

class DevelopmentalEditorAgent:
    """Developmental Editor Agent that addresses structural and thematic issues."""
    
    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "DevelopmentalEditor"
    
    def evaluate_structure(self, state: NovelState) -> Dict[str, Any]:
        """Evaluate the overall narrative structure of the novel."""
        prompt = PromptTemplate(
            template="""You are a developmental editor evaluating the structure of a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Themes: {themes}

Chapter Structure:
{chapter_structure}

Evaluate the novel's overall structure for:

1. NARRATIVE ARC
   Assess the shape of the overall narrative arc
   Evaluate whether the structure effectively supports the premise
   
2. PACING ANALYSIS
   Identify pacing issues (too slow, too fast, uneven)
   Analyze tension distribution across the novel
   
3. STRUCTURAL BALANCE
   Evaluate the balance between different narrative elements (action, dialogue, description, etc.)
   Assess whether subplots are given appropriate weight
   
4. THEMATIC DEVELOPMENT
   Analyze how effectively themes are introduced, developed, and resolved
   Identify missed opportunities for thematic resonance
   
5. CHARACTER ARCS
   Evaluate how character development is structured through the narrative
   Assess whether character arcs align with and support the overall structure

6. RECOMMENDED STRUCTURAL CHANGES
   Suggest specific structural changes to address identified issues
   Provide a revised structural outline if significant changes are needed

Provide a comprehensive structural analysis with specific examples and actionable recommendations.
""",
            input_variables=["project_name", "genre", "premise", "themes", "chapter_structure"]
        )
        
        # Create chapter structure summary
        chapter_structure = ""
        for num, chapter in sorted(state.chapters.items()):
            chapter_structure += f"Chapter {num}: {chapter.title}\n"
            chapter_structure += f"Summary: {chapter.summary}\n"
            chapter_structure += f"POV: {chapter.pov_character or 'Not specified'}\n"
            chapter_structure += f"Word count: {chapter.word_count}\n\n"
        
        if not chapter_structure:
            chapter_structure = "Chapters not yet developed."
        
        response = self.llm.invoke(prompt.format(
            project_name=state.project_name,
            genre=state.genre,
            premise=state.premise,
            themes=", ".join(state.themes),
            chapter_structure=chapter_structure
        ))
        
        # Extract sections from the response
        sections = {
            "narrative_arc": "",
            "pacing_analysis": "",
            "structural_balance": "",
            "thematic_development": "",
            "character_arcs": "",
            "recommended_changes": ""
        }
        
        current_section = None
        
        for line in response.content.split("\n"):
            if "NARRATIVE ARC" in line or "1. NARRATIVE ARC" in line:
                current_section = "narrative_arc"
                continue
            elif "PACING ANALYSIS" in line or "2. PACING ANALYSIS" in line:
                current_section = "pacing_analysis"
                continue
            elif "STRUCTURAL BALANCE" in line or "3. STRUCTURAL BALANCE" in line:
                current_section = "structural_balance"
                continue
            elif "THEMATIC DEVELOPMENT" in line or "4. THEMATIC DEVELOPMENT" in line:
                current_section = "thematic_development"
                continue
            elif "CHARACTER ARCS" in line or "5. CHARACTER ARCS" in line:
                current_section = "character_arcs"
                continue
            elif "RECOMMENDED STRUCTURAL CHANGES" in line or "6. RECOMMENDED STRUCTURAL CHANGES" in line:
                current_section = "recommended_changes"
                continue
                
            if current_section and line.strip():
                sections[current_section] += line + "\n"
        
        return {
            "structural_analysis": response.content,
            "sections": sections
        }
    
    def revise_chapter_structure(self, chapter: Chapter, structural_notes: str) -> Dict[str, Any]:
        """Provide structural revision guidance for a chapter."""
        prompt = PromptTemplate(
            template="""You are a developmental editor providing structural revision guidance for a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}
Summary: {summary}
POV Character: {pov_character}

Content:
{content}

Structural Issues to Address:
{structural_notes}

Provide detailed guidance on how to restructure this chapter to address the identified issues. Include:

1. STRUCTURAL DIAGNOSIS
   Analyze the current structure of the chapter
   Identify specific structural weaknesses
   
2. SCENE RESTRUCTURING
   Suggest how to reorder, add, remove, expand, or contract scenes
   Provide a scene-by-scene outline for the revised structure
   
3. NARRATIVE FLOW
   Recommend improvements to transitions between scenes
   Suggest how to better manage the flow of information
   
4. BEGINNING AND ENDING
   Advise on strengthening the chapter's opening and closing
   Ensure proper setup and payoff within the chapter

5. INTEGRATION POINTS
   Identify where to better integrate this chapter with the overall novel structure
   Suggest connective elements to reinforce with other chapters

Your guidance should be specific and actionable, providing clear direction for revision while respecting the chapter's core purpose and content.
""",
            input_variables=["title", "chapter_number", "summary", "pov_character", "content", "structural_notes"]
        )
        
        response = self.llm.invoke(prompt.format(
            title=chapter.title,
            chapter_number=chapter.number,
            summary=chapter.summary,
            pov_character=chapter.pov_character or "Not specified",
            content=chapter.content[:2000] + "..." if len(chapter.content) > 2000 else chapter.content,
            structural_notes=structural_notes
        ))
        
        # Extract sections from the response
        sections = {
            "structural_diagnosis": "",
            "scene_restructuring": "",
            "narrative_flow": "",
            "beginning_and_ending": "",
            "integration_points": ""
        }
        
        current_section = None
        
        for line in response.content.split("\n"):
            if "STRUCTURAL DIAGNOSIS" in line or "1. STRUCTURAL DIAGNOSIS" in line:
                current_section = "structural_diagnosis"
                continue
            elif "SCENE RESTRUCTURING" in line or "2. SCENE RESTRUCTURING" in line:
                current_section = "scene_restructuring"
                continue
            elif "NARRATIVE FLOW" in line or "3. NARRATIVE FLOW" in line:
                current_section = "narrative_flow"
                continue
            elif "BEGINNING AND ENDING" in line or "4. BEGINNING AND ENDING" in line:
                current_section = "beginning_and_ending"
                continue
            elif "INTEGRATION POINTS" in line or "5. INTEGRATION POINTS" in line:
                current_section = "integration_points"
                continue
                
            if current_section and line.strip():
                sections[current_section] += line + "\n"
        
        return {
            "revision_guidance": response.content,
            "sections": sections
        }

class LineEditorAgent:
    """Line Editor Agent that improves sentence flow and language."""
    
    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "LineEditor"
    
    def edit_chapter(self, chapter: Chapter) -> Chapter:
        """Perform line editing on a chapter."""
        prompt = PromptTemplate(
            template="""You are a professional line editor improving the prose quality of a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}

Content:
{content}

Perform comprehensive line editing on this chapter to improve its prose quality. Focus on:

1. SENTENCE FLOW AND VARIETY
   Improve sentence rhythm, length variation, and flow
   Fix awkward constructions
   
2. WORD CHOICE AND PRECISION
   Replace weak, vague, or repetitive words with stronger alternatives
   Ensure precise and evocative language
   
3. REDUNDANCY ELIMINATION
   Remove unnecessary repetition of words, phrases, or ideas
   Tighten prose without losing meaning
   
4. ACTIVE VOICE AND SHOWING
   Convert passive voice to active where appropriate
   Transform "telling" statements into "showing" through sensory details and concrete language
   
5. CLARITY AND CONCISION
   Clarify any confusing or ambiguous passages
   Make language more concise while maintaining style and impact

Your edits should preserve the author's voice and style while elevating the quality of the prose.
Do not change plot elements or character decisions - focus only on the expression of the existing content.

Provide the complete edited chapter.
""",
            input_variables=["title", "chapter_number", "content"]
        )
        
        response = self.llm.invoke(prompt.format(
            title=chapter.title,
            chapter_number=chapter.number,
            content=chapter.content
        ))
        
        # Update the chapter
        edited_chapter = chapter.model_copy()
        edited_chapter.content = response.content
        edited_chapter.word_count = len(response.content.split())
        edited_chapter.revision_count += 1
        
        return edited_chapter
    
    def analyze_prose_quality(self, text: str) -> Dict[str, Any]:
        """Analyze the prose quality of a text."""
        prompt = PromptTemplate(
            template="""You are a professional editor analyzing the prose quality of a novel excerpt.

Text:
{text}

Analyze this text for:

1. SENTENCE STRUCTURE
   Evaluate sentence variety, rhythm, and flow
   Identify patterns, strengths, and weaknesses
   
2. LANGUAGE EFFICIENCY
   Assess word choice precision and economy
   Evaluate for redundancy and wordiness
   
3. VOICE AND STYLE
   Analyze the distinctive qualities of the author's voice
   Evaluate consistency of style
   
4. SHOWING VS. TELLING
   Assess the balance between showing and telling
   Identify opportunities to convert telling to showing
   
5. DIALOGUE EFFECTIVENESS
   Evaluate natural flow and distinctiveness of dialogue
   Assess dialogue mechanics and attribution

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific examples from the text
- Suggestions for improvement

Also provide an overall prose quality score and summary of the author's stylistic fingerprint.
""",
            input_variables=["text"]
        )
        
        response = self.llm.invoke(prompt.format(text=text[:3000]))  # Limit text length for token constraints
        
        # Extract scores from the response
        scores = {}
        aspects = ["sentence structure", "language efficiency", "voice and style", 
                  "showing vs. telling", "dialogue effectiveness"]
        
        for aspect in aspects:
            pattern = rf"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    scores[aspect.replace(" ", "_").replace(".", "")] = float(match.group(1))
                except:
                    scores[aspect.replace(" ", "_").replace(".", "")] = 0.5  # Default if parsing fails
            else:
                scores[aspect.replace(" ", "_").replace(".", "")] = 0.5  # Default if not found
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5
        
        return {
            "analysis": response.content,
            "scores": scores,
            "overall_score": overall_score
        }

class DialogueEnhancementAgent:
    """Dialogue Enhancement Agent that refines character voices and subtext."""
    
    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "DialogueEnhancement"
    
    def enhance_dialogue(self, chapter: Chapter, characters: Dict[str, Any]) -> Chapter:
        """Enhance the dialogue in a chapter."""
        # Prepare character dialogue information
        character_voices = ""
        chapter_characters = self._extract_chapter_characters(chapter.content)
        
        for name in chapter_characters:
            if name in characters:
                char = characters[name]
                character_voices += f"CHARACTER: {name}\n"
                character_voices += f"Role: {char.role}\n"
                
                if hasattr(char, 'dialogue_patterns') and char.dialogue_patterns:
                    if isinstance(char.dialogue_patterns, dict):
                        if 'speech_style' in char.dialogue_patterns:
                            character_voices += f"Speech style: {char.dialogue_patterns['speech_style']}\n"
                        if 'verbal_tics' in char.dialogue_patterns:
                            tics = char.dialogue_patterns['verbal_tics']
                            if isinstance(tics, list):
                                character_voices += f"Verbal tics: {', '.join(tics)}\n"
                            else:
                                character_voices += f"Verbal tics: {tics}\n"
                
                character_voices += "\n"
        
        prompt = PromptTemplate(
            template="""You are a dialogue specialist enhancing dialogue in a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}

Character Voice Guidelines:
{character_voices}

Original Content:
{content}

Enhance the dialogue in this chapter to make it more authentic, distinctive, and effective. Focus on:

1. CHARACTER VOICE CONSISTENCY
   Ensure each character's dialogue matches their established speech patterns
   Maintain consistency with their verbal tics and vocabulary choices
   
2. SUBTEXT DEVELOPMENT
   Add layers of meaning beneath the surface of conversations
   Create dialogue that communicates more than just literal meaning
   
3. DIALOGUE DYNAMICS
   Improve the rhythm and flow of conversations
   Create more natural exchanges with appropriate interruptions, hesitations, etc.
   
4. DIALOGUE ATTRIBUTION
   Refine dialogue tags and action beats
   Ensure attribution enhances rather than distracts from the dialogue
   
5. EMOTIONAL AUTHENTICITY
   Make emotional states more convincing through dialogue
   Show rather than tell emotions through speech patterns

Your enhancements should maintain the existing plot and character decisions while making the dialogue more compelling and authentic.

Provide the complete enhanced chapter.
""",
            input_variables=["title", "chapter_number", "character_voices", "content"]
        )
        
        response = self.llm.invoke(prompt.format(
            title=chapter.title,
            chapter_number=chapter.number,
            character_voices=character_voices,
            content=chapter.content
        ))
        
        # Update the chapter
        enhanced_chapter = chapter.model_copy()
        enhanced_chapter.content = response.content
        enhanced_chapter.word_count = len(response.content.split())
        
        return enhanced_chapter
    
    def _extract_chapter_characters(self, content: str) -> List[str]:
        """Extract character names mentioned in the chapter."""
        # This is a simplified implementation
        # In a real system, we would use NER or other advanced techniques
        character_names = set()
        
        # Simple pattern matching for dialogue attribution
        dialogue_pattern = r'"[^"]+"\s*,?\s*([A-Z][a-z]+)(?:\s+[a-z]+)?'
        matches = re.finditer(dialogue_pattern, content)
        
        for match in matches:
            name = match.group(1)
            if name not in ["I", "He", "She", "They", "It", "Then", "But", "And", "The"]:
                character_names.add(name)
        
        return list(character_names)
    
    def analyze_dialogue_quality(self, text: str, characters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the quality of dialogue in a text."""
        # Prepare character dialogue information
        character_voices = ""
        for name, char in characters.items():
            character_voices += f"CHARACTER: {name}\n"
            character_voices += f"Role: {char.role}\n"
            
            if hasattr(char, 'dialogue_patterns') and char.dialogue_patterns:
                if isinstance(char.dialogue_patterns, dict):
                    if 'speech_style' in char.dialogue_patterns:
                        character_voices += f"Speech style: {char.dialogue_patterns['speech_style']}\n"
            
            character_voices += "\n"
        
        prompt = PromptTemplate(
            template="""You are a dialogue specialist analyzing dialogue quality in a novel excerpt.

Character Information:
{character_voices}

Text:
{text}

Analyze the dialogue in this text for:

1. CHARACTER VOICE DISTINCTIVENESS
   Evaluate how distinctive each character's voice is
   Assess consistency with character personalities
   
2. DIALOGUE AUTHENTICITY
   Analyze how natural and believable the dialogue sounds
   Evaluate for stilted or unrealistic exchanges
   
3. SUBTEXT EFFECTIVENESS
   Assess the layers of meaning beneath the surface dialogue
   Evaluate how well dialogue communicates unspoken thoughts/feelings
   
4. DIALOGUE MECHANICS
   Analyze effectiveness of attribution and action beats
   Evaluate formatting and punctuation
   
5. DIALOGUE PURPOSE
   Assess how dialogue advances plot, reveals character, or provides information
   Evaluate balance between dialogue functions

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific examples from the text
- Suggestions for improvement

Also provide an overall dialogue quality score and summary.
""",
            input_variables=["character_voices", "text"]
        )
        
        response = self.llm.invoke(prompt.format(
            character_voices=character_voices,
            text=text[:3000]  # Limit text length for token constraints
        ))
        
        # Extract scores from the response
        scores = {}
        aspects = ["character voice distinctiveness", "dialogue authenticity", "subtext effectiveness", 
                  "dialogue mechanics", "dialogue purpose"]
        
        for aspect in aspects:
            pattern = rf"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    scores[aspect.replace(" ", "_")] = float(match.group(1))
                except:
                    scores[aspect.replace(" ", "_")] = 0.5  # Default if parsing fails
            else:
                scores[aspect.replace(" ", "_")] = 0.5  # Default if not found
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5
        
        return {
            "analysis": response.content,
            "scores": scores,
            "overall_score": overall_score
        }

class TensionOptimizationAgent:
    """Tension Optimization Agent that adjusts pacing and dramatic moments."""
    
    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "TensionOptimization"
    
    def optimize_tension(self, chapter: Chapter, target_tension: float) -> Chapter:
        """Optimize the tension and pacing in a chapter to match a target tension level."""
        prompt = PromptTemplate(
            template="""You are a narrative tension specialist optimizing the pacing and dramatic tension in a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}
Current Estimated Tension Level: {current_tension}
Target Tension Level: {target_tension} (scale 0.0-1.0, where 1.0 is maximum tension)

Original Content:
{content}

Revise this chapter to achieve the target tension level. Based on whether you need to increase or decrease tension:

IF INCREASING TENSION:
- Add stakes or consequences to decisions
- Introduce complications or obstacles
- Create time pressure or urgency
- Heighten emotional conflicts
- Add foreshadowing of future threats
- Increase sensory details during tense moments
- Create more clipped, urgent pacing through sentence structure

IF DECREASING TENSION:
- Add reflective or contemplative moments
- Develop deeper character interactions
- Expand descriptive passages
- Introduce moments of relief or success
- Use longer, more flowing sentence structures
- Focus more on internal thoughts than external threats
- Add context or background information

The revision should maintain all plot points and character decisions while adjusting only the tension and pacing.

Provide the complete revised chapter.
""",
            input_variables=["title", "chapter_number", "current_tension", "target_tension", "content"]
        )
        
        # Estimate current tension
        current_tension = self._estimate_tension(chapter.content)
        
        response = self.llm.invoke(prompt.format(
            title=chapter.title,
            chapter_number=chapter.number,
            current_tension=current_tension,
            target_tension=target_tension,
            content=chapter.content
        ))
        
        # Update the chapter
        optimized_chapter = chapter.model_copy()
        optimized_chapter.content = response.content
        optimized_chapter.word_count = len(response.content.split())
        
        return optimized_chapter
    
    def _estimate_tension(self, text: str) -> float:
        """Estimate the current tension level in a text."""
        prompt = PromptTemplate(
            template="""You are a narrative tension analyzer. Estimate the overall tension level in this text on a scale from 0.0 to 1.0, where:

0.0 = Completely calm, reflective, no tension
0.5 = Moderate tension, balanced with calmer moments
1.0 = Extreme tension, urgent, high-stakes, fast-paced

Text:
{text}

Consider factors like:
- Pacing (sentence and paragraph length)
- Stakes and consequences
- Character emotional states
- Presence of threats or obstacles
- Time pressure elements
- Language intensity and sensory detail in action
- Cliffhangers or unresolved questions

Provide only a single number between 0.0 and 1.0 representing the estimated tension level.
""",
            input_variables=["text"]
        )
        
        response = self.llm.invoke(prompt.format(text=text[:3000]))  # Limit text length for token constraints
        
        # Extract the tension value
        try:
            # Look for a decimal number in the response
            match = re.search(r'\d+\.\d+', response.content)
            if match:
                tension = float(match.group(0))
                return min(max(tension, 0.0), 1.0)  # Ensure it's between 0 and 1
        except:
            pass
        
        return 0.5  # Default if extraction fails
    
    def create_tension_map(self, state: NovelState) -> Dict[str, Any]:
        """Create a tension map for the entire novel, recommending target tension for each chapter."""
        prompt = PromptTemplate(
            template="""You are a narrative tension designer creating a tension map for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Total Chapters: {chapter_count}

Chapter Summaries:
{chapter_summaries}

Create a comprehensive tension map for this novel that will create an optimal reading experience. For each chapter:

1. Assess its position in the narrative arc
2. Recommend a specific tension level (0.0-1.0)
3. Explain why this tension level is appropriate at this point in the story
4. Suggest specific tension techniques appropriate for this chapter

Your tension map should:
- Create a compelling narrative rhythm with appropriate peaks and valleys
- Align with genre expectations for {genre}
- Build toward major climactic moments
- Provide necessary relief after high-tension sequences
- Create an overall ascending pattern with appropriate modulation

Also provide a visual representation of the tension curve (using ASCII art if necessary) for the complete novel.
""",
            input_variables=["project_name", "genre", "chapter_count", "chapter_summaries"]
        )
        
        # Create chapter summaries
        chapter_summaries = ""
        for num, chapter in sorted(state.chapters.items()):
            chapter_summaries += f"Chapter {num}: {chapter.title}\n"
            chapter_summaries += f"Summary: {chapter.summary}\n\n"
        
        if not chapter_summaries:
            chapter_summaries = "Chapter details not yet available."
        
        response = self.llm.invoke(prompt.format(
            project_name=state.project_name,
            genre=state.genre,
            chapter_count=len(state.chapters),
            chapter_summaries=chapter_summaries
        ))
        
        # Extract tension recommendations by chapter
        chapter_tensions = {}
        current_chapter = None
        current_text = ""
        in_chapter_section = False
        
        for line in response.content.split("\n"):
            # Check for chapter headers
            chapter_match = re.match(r"Chapter\s+(\d+)[:\s]", line)
            if chapter_match:
                # Save previous chapter if we have one
                if current_chapter and current_text:
                    chapter_tensions[current_chapter] = current_text
                
                current_chapter = int(chapter_match.group(1))
                current_text = line + "\n"
                in_chapter_section = True
            elif in_chapter_section and line.strip():
                current_text += line + "\n"
            elif in_chapter_section and not line.strip():
                # Blank line might end a section
                if current_chapter and current_text:
                    chapter_tensions[current_chapter] = current_text
                    in_chapter_section = False
                    current_chapter = None
                    current_text = ""
        
        # Save the last chapter if we have one
        if current_chapter and current_text:
            chapter_tensions[current_chapter] = current_text
        
        # Extract tension values
        tension_values = {}
        for chapter, text in chapter_tensions.items():
            # Look for tension level specification
            match = re.search(r"tension level.*?(\d+\.\d+)", text, re.IGNORECASE)
            if match:
                try:
                    tension_values[chapter] = float(match.group(1))
                except:
                    tension_values[chapter] = 0.5  # Default if parsing fails
            else:
                tension_values[chapter] = 0.5  # Default if not found
        
        return {
            "tension_map": response.content,
            "chapter_tensions": chapter_tensions,
            "tension_values": tension_values
        }
