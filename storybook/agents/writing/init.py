# agents/writing/__init__.py
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import re

from storybook.utils.state import NovelState, Chapter
from storybook.config import Config


class WritingSupervisorAgent:
    """Writing Supervisor Agent that coordinates the writing process."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "WritingSupervisor"

    def create_writing_plan(self, state: NovelState) -> Dict[str, Any]:
        """Create a detailed writing plan for the novel."""
        prompt = PromptTemplate(
            template="""You are a writing supervisor creating a detailed novel writing plan.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Target Word Count: {target_word_count}

Character Information:
{character_summary}

Plot Information:
{plot_summary}

Create a comprehensive writing plan that includes:

1. CHAPTER SEQUENCING
   List all chapters in order with target word counts and POV characters
   
2. WRITING PRIORITIES
   Identify which chapters should be written first and why
   Highlight any dependencies between chapters
   
3. STYLISTIC GUIDELINES
   Establish consistent style guidelines for the novel
   Define narrative voice, tense, and point of view approach
   
4. CONSISTENCY REQUIREMENTS
   Specify what elements must remain consistent across chapters
   Highlight potential continuity challenges
   
5. QUALITY METRICS
   Define specific quality criteria for each chapter
   Establish revision expectations
   
Format your response as a detailed writing plan document with clear sections.
""",
            input_variables=[
                "project_name",
                "genre",
                "premise",
                "target_word_count",
                "character_summary",
                "plot_summary",
            ],
        )

        # Create character summary for the prompt
        character_summary = ""
        for name, character in state.characters.items():
            character_summary += (
                f"{name}: {character.role} - {character.background[:100]}...\n"
            )

        if not character_summary:
            character_summary = "Characters still being developed."

        # Create plot summary for the prompt
        plot_summary = ""
        for i, point in enumerate(state.plot_points[:10]):  # Limit to 10 points
            plot_summary += f"{i+1}. {point.title}: {point.description[:100]}...\n"

        if not plot_summary:
            plot_summary = "Plot still being developed."

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                target_word_count=state.target_word_count,
                character_summary=character_summary,
                plot_summary=plot_summary,
            )
        )

        # Extract sections from the response
        sections = {
            "chapter_sequencing": "",
            "writing_priorities": "",
            "stylistic_guidelines": "",
            "consistency_requirements": "",
            "quality_metrics": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "CHAPTER SEQUENCING" in line or "1. CHAPTER SEQUENCING" in line:
                current_section = "chapter_sequencing"
                continue
            elif "WRITING PRIORITIES" in line or "2. WRITING PRIORITIES" in line:
                current_section = "writing_priorities"
                continue
            elif "STYLISTIC GUIDELINES" in line or "3. STYLISTIC GUIDELINES" in line:
                current_section = "stylistic_guidelines"
                continue
            elif (
                "CONSISTENCY REQUIREMENTS" in line
                or "4. CONSISTENCY REQUIREMENTS" in line
            ):
                current_section = "consistency_requirements"
                continue
            elif "QUALITY METRICS" in line or "5. QUALITY METRICS" in line:
                current_section = "quality_metrics"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        return {"writing_plan": response.content, "sections": sections}

    def assess_chapter_quality(
        self, chapter: Chapter, quality_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess the quality of a chapter against established criteria."""
        prompt = PromptTemplate(
            template="""You are a writing quality assessor evaluating a novel chapter.

Chapter Information:
Title: {title}
Number: {number}
POV Character: {pov_character}
Summary: {summary}

Content:
{content}

Quality Criteria:
{quality_criteria}

Evaluate this chapter on all quality criteria. For each criterion:
1. Assign a score from 0.0 to 1.0
2. Provide specific examples from the chapter that justify the score
3. Offer constructive suggestions for improvement

Also provide an overall assessment of the chapter's strengths and weaknesses,
and identify any inconsistencies or issues that need to be addressed.

Format your response with clear sections for each criterion and a final summary.
""",
            input_variables=[
                "title",
                "number",
                "pov_character",
                "summary",
                "content",
                "quality_criteria",
            ],
        )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                number=chapter.number,
                pov_character=chapter.pov_character or "Not specified",
                summary=chapter.summary,
                content=(
                    chapter.content[:2000] + "..."
                    if len(chapter.content) > 2000
                    else chapter.content
                ),
                quality_criteria="\n".join(
                    [f"- {k}: {v}" for k, v in quality_criteria.items()]
                ),
            )
        )

        # Extract scores from the response
        # This is a simplified extraction - in a real implementation, would be more robust
        scores = {}
        for criterion in quality_criteria.keys():
            pattern = rf"{criterion}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    scores[criterion] = float(match.group(1))
                except:
                    scores[criterion] = 0.5  # Default if parsing fails
            else:
                scores[criterion] = 0.5  # Default if not found

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5

        return {
            "assessment": response.content,
            "scores": scores,
            "overall_score": overall_score,
        }

    def coordinate_revisions(self, state: NovelState) -> Dict[str, List[str]]:
        """Coordinate revision priorities for all chapters."""
        prompt = PromptTemplate(
            template="""You are a revision coordinator identifying priority revisions for a novel.

Novel Details:
Title: {project_name}
Genre: {genre}

Chapter Status:
{chapter_status}

Based on the current state of all chapters, identify:

1. PRIORITY REVISIONS
   Which chapters need immediate revision and why
   Specific aspects that need improvement in each priority chapter
   
2. CONSISTENCY ISSUES
   Any continuity or consistency issues between chapters
   Characters, plot points, or settings that are inconsistently portrayed
   
3. STRUCTURAL RECOMMENDATIONS
   Any structural changes needed (reordering, combining, or splitting chapters)
   Pacing adjustments required
   
4. THEMATIC REINFORCEMENT
   Opportunities to strengthen theme integration
   Chapters where themes could be more explicitly developed

Provide a comprehensive revision strategy that addresses the most critical issues first.
""",
            input_variables=["project_name", "genre", "chapter_status"],
        )

        # Create chapter status summary
        chapter_status = ""
        for num, chapter in sorted(state.chapters.items()):
            quality_str = (
                ", ".join([f"{k}: {v:.2f}" for k, v in chapter.quality_metrics.items()])
                if chapter.quality_metrics
                else "Not assessed"
            )
            chapter_status += f"Chapter {num}: {chapter.title} - {chapter.word_count} words - Revision count: {chapter.revision_count} - Quality: {quality_str}\n"

        if not chapter_status:
            chapter_status = "No chapters have been written yet."

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                chapter_status=chapter_status,
            )
        )

        # Extract revision priorities
        sections = {
            "priority_revisions": [],
            "consistency_issues": [],
            "structural_recommendations": [],
            "thematic_reinforcement": [],
        }

        current_section = None

        for line in response.content.split("\n"):
            if "PRIORITY REVISIONS" in line or "1. PRIORITY REVISIONS" in line:
                current_section = "priority_revisions"
                continue
            elif "CONSISTENCY ISSUES" in line or "2. CONSISTENCY ISSUES" in line:
                current_section = "consistency_issues"
                continue
            elif (
                "STRUCTURAL RECOMMENDATIONS" in line
                or "3. STRUCTURAL RECOMMENDATIONS" in line
            ):
                current_section = "structural_recommendations"
                continue
            elif (
                "THEMATIC REINFORCEMENT" in line or "4. THEMATIC REINFORCEMENT" in line
            ):
                current_section = "thematic_reinforcement"
                continue

            if current_section and line.strip():
                sections[current_section].append(line.strip())

        return sections


class ChapterWriterAgent:
    """Chapter Writer Agent that generates chapter content."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ChapterWriter"

    def write_chapter(
        self, chapter_outline: Any, state: NovelState, style_guide: Dict[str, Any]
    ) -> Chapter:
        """Write a complete chapter based on the chapter outline."""
        # Get character information
        characters_info = ""
        for name, character in state.characters.items():
            if name == chapter_outline.pov_character:
                # More detail for POV character
                characters_info += f"POV CHARACTER - {name}: {character.role}\n"
                characters_info += f"Background: {character.background[:300]}...\n"
                if character.dialogue_patterns:
                    characters_info += f"Speech style: {character.dialogue_patterns.get('speech_style', 'Not specified')}\n"
                characters_info += "\n"
            else:
                # Brief info for other characters
                characters_info += (
                    f"{name}: {character.role} - {character.background[:100]}...\n"
                )

        # Get setting information
        settings_info = ""
        for name, setting in state.settings.items():
            settings_info += (
                f"{name}: {setting.get('description', 'No description')}...\n"
            )

        prompt = PromptTemplate(
            template="""You are a professional novelist writing a chapter for a {genre} novel.

Chapter Information:
Title: {title}
Number: {chapter_number}
POV Character: {pov_character}
Setting: {setting}
Summary: {summary}

Opening Hook: {opening_hook}
Closing Hook: {closing_hook}

Character Development Points:
{character_development}

Plot Advancement:
{plot_advancement}

Themes to Explore:
{themes_explored}

Character Information:
{characters_info}

Setting Information:
{settings_info}

Style Guide:
{style_guide}

Write a complete, polished chapter based on this outline. The chapter should:
1. Be approximately {word_count} words
2. Have a compelling opening that hooks the reader
3. Develop the POV character in ways specified
4. Advance the plot through the described events
5. Explore the identified themes naturally
6. End with a satisfying conclusion that makes the reader want to continue
7. Adhere to the style guide provided
8. Include dialogue that reflects each character's unique voice
9. Include sensory details that bring the setting to life

Use your strongest narrative writing to create a chapter that feels like it belongs in a bestselling novel.
""",
            input_variables=[
                "genre",
                "title",
                "chapter_number",
                "pov_character",
                "setting",
                "summary",
                "opening_hook",
                "closing_hook",
                "character_development",
                "plot_advancement",
                "themes_explored",
                "characters_info",
                "settings_info",
                "style_guide",
                "word_count",
            ],
        )

        # Format character development for the prompt
        character_dev = "\n".join(
            [
                f"- {char}: {dev}"
                for char, dev in chapter_outline.character_development.items()
            ]
        )

        # Format plot advancement for the prompt
        plot_adv = "\n".join(
            [f"- {point}" for point in chapter_outline.plot_advancement]
        )

        # Format themes for the prompt
        themes = "\n".join([f"- {theme}" for theme in chapter_outline.themes_explored])

        # Format style guide for the prompt
        style_guide_text = "\n".join([f"- {k}: {v}" for k, v in style_guide.items()])

        response = self.llm.invoke(
            prompt.format(
                genre=state.genre,
                title=chapter_outline.title,
                chapter_number=chapter_outline.chapter_number,
                pov_character=chapter_outline.pov_character,
                setting=chapter_outline.setting,
                summary=chapter_outline.summary,
                opening_hook=chapter_outline.opening_hook,
                closing_hook=chapter_outline.closing_hook,
                character_development=character_dev,
                plot_advancement=plot_adv,
                themes_explored=themes,
                characters_info=characters_info,
                settings_info=settings_info,
                style_guide=style_guide_text,
                word_count=chapter_outline.estimated_word_count,
            )
        )

        # Count words in the chapter
        word_count = len(response.content.split())

        # Create the chapter
        chapter = Chapter(
            number=chapter_outline.chapter_number,
            title=chapter_outline.title,
            pov_character=chapter_outline.pov_character,
            summary=chapter_outline.summary,
            content=response.content,
            word_count=word_count,
            completed=True,
            revision_count=0,
            quality_metrics={},
        )

        return chapter

    def revise_chapter(
        self,
        chapter: Chapter,
        revision_notes: Dict[str, Any],
        style_guide: Dict[str, Any],
    ) -> Chapter:
        """Revise a chapter based on feedback and revision notes."""
        prompt = PromptTemplate(
            template="""You are a professional novelist revising a chapter based on editorial feedback.

Chapter Information:
Title: {title}
Number: {chapter_number}
POV Character: {pov_character}
Summary: {summary}

Current Content:
{content}

Revision Notes:
{revision_notes}

Style Guide:
{style_guide}

Revise this chapter to address all the revision notes while maintaining continuity with the rest of the novel.
Your revision should:
1. Fix all identified issues
2. Strengthen the chapter based on the feedback
3. Maintain the original tone and voice
4. Ensure consistency with the style guide
5. Preserve the essential plot points and character moments
6. Enhance descriptive elements, dialogue, and pacing as needed

Provide a complete, revised version of the chapter that addresses all feedback points.
""",
            input_variables=[
                "title",
                "chapter_number",
                "pov_character",
                "summary",
                "content",
                "revision_notes",
                "style_guide",
            ],
        )

        # Format revision notes for the prompt
        revision_notes_text = "\n".join([f"- {note}" for note in revision_notes])

        # Format style guide for the prompt
        style_guide_text = "\n".join([f"- {k}: {v}" for k, v in style_guide.items()])

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                chapter_number=chapter.number,
                pov_character=chapter.pov_character or "Not specified",
                summary=chapter.summary,
                content=chapter.content,
                revision_notes=revision_notes_text,
                style_guide=style_guide_text,
            )
        )

        # Count words in the revised chapter
        word_count = len(response.content.split())

        # Update the chapter
        revised_chapter = chapter.model_copy()
        revised_chapter.content = response.content
        revised_chapter.word_count = word_count
        revised_chapter.revision_count += 1

        return revised_chapter


class ContinuityManagerAgent:
    """Continuity Manager Agent that ensures narrative consistency across chapters."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ContinuityManager"

    def check_continuity(
        self, current_chapter: Chapter, previous_chapters: List[Chapter]
    ) -> Dict[str, Any]:
        """Check continuity between the current chapter and previous chapters."""
        if not previous_chapters:
            return {"continuity_issues": [], "consistency_score": 1.0}

        prompt = PromptTemplate(
            template="""You are a continuity editor checking a novel chapter for consistency with previous chapters.

Current Chapter:
Title: {current_title}
Number: {current_number}
Content: {current_content}

Previous Chapters Summary:
{previous_chapters_summary}

Check this chapter for any continuity issues, including:
1. Character consistency (behavior, knowledge, abilities)
2. Plot continuity (events referenced correctly, cause-effect relationships)
3. Setting consistency (locations, descriptions, physics of the world)
4. Timeline consistency (passage of time, sequence of events)
5. Object persistence (items introduced or used consistently)
6. Thematic consistency (themes developed consistently)

For each issue found:
- Describe the specific inconsistency
- Reference where it appears in the current chapter
- Explain how it contradicts previous chapters
- Suggest a specific correction

Also provide an overall consistency score from 0.0 to 1.0 and a summary of the chapter's continuity strengths.
""",
            input_variables=[
                "current_title",
                "current_number",
                "current_content",
                "previous_chapters_summary",
            ],
        )

        # Create summary of previous chapters
        previous_chapters_summary = ""
        for chapter in previous_chapters[
            -3:
        ]:  # Limit to the last 3 chapters for token constraints
            previous_chapters_summary += f"Chapter {chapter.number}: {chapter.title}\n"
            previous_chapters_summary += f"Summary: {chapter.summary}\n"
            previous_chapters_summary += f"Key events: {chapter.content[:300]}...\n\n"

        response = self.llm.invoke(
            prompt.format(
                current_title=current_chapter.title,
                current_number=current_chapter.number,
                current_content=(
                    current_chapter.content[:2000] + "..."
                    if len(current_chapter.content) > 2000
                    else current_chapter.content
                ),
                previous_chapters_summary=previous_chapters_summary,
            )
        )

        # Extract continuity issues
        issues = []
        current_issue = ""
        in_issue = False

        for line in response.content.split("\n"):
            if any(
                line.startswith(prefix)
                for prefix in ["1.", "2.", "3.", "4.", "5.", "Issue", "Inconsistency"]
            ):
                if current_issue:
                    issues.append(current_issue)
                current_issue = line.strip()
                in_issue = True
            elif in_issue and line.strip():
                current_issue += " " + line.strip()
            elif in_issue and not line.strip():
                if current_issue:
                    issues.append(current_issue)
                    current_issue = ""
                in_issue = False

        if current_issue:
            issues.append(current_issue)

        # Extract consistency score
        consistency_score = 0.7  # Default score
        score_line = [
            line
            for line in response.content.split("\n")
            if "consistency score" in line.lower()
        ]
        if score_line:
            try:
                score_text = score_line[0]
                score = float(
                    [s for s in score_text.split() if s.replace(".", "").isdigit()][0]
                )
                consistency_score = score
            except:
                pass

        return {
            "continuity_issues": issues,
            "consistency_score": consistency_score,
            "full_analysis": response.content,
        }

    def track_narrative_elements(self, chapters: List[Chapter]) -> Dict[str, Any]:
        """Track important narrative elements across all chapters."""
        if not chapters:
            return {"elements": {}, "tracking_report": "No chapters to track."}

        prompt = PromptTemplate(
            template="""You are a narrative tracker identifying and tracking important elements across a novel.

Novel Chapters:
{chapters_summary}

Create a comprehensive tracking document that monitors:

1. CHARACTERS
   For each character, track:
   - First appearance
   - Knowledge acquired in each chapter
   - Emotional states/changes
   - Key decisions and actions
   
2. PLOT THREADS
   For each plot thread, track:
   - Introduction point
   - Development points
   - Resolution status
   
3. OBJECTS/ITEMS
   Track important objects:
   - Introduction
   - Current location/ownership
   - Significance changes
   
4. SETTINGS
   For each location, track:
   - First appearance
   - Description consistency
   - Characters who have been there
   
5. MYSTERIES/QUESTIONS
   Track narrative questions:
   - When raised
   - Clues provided
   - Resolution status
   
Format this as a reference document that could be used to ensure continuity in future chapters.
""",
            input_variables=["chapters_summary"],
        )

        # Create summary of chapters
        chapters_summary = ""
        for chapter in chapters:
            chapters_summary += f"Chapter {chapter.number}: {chapter.title}\n"
            chapters_summary += f"Summary: {chapter.summary}\n"
            chapters_summary += f"Excerpt: {chapter.content[:200]}...\n\n"

        response = self.llm.invoke(prompt.format(chapters_summary=chapters_summary))

        # Extract tracked elements by section
        sections = {
            "characters": "",
            "plot_threads": "",
            "objects": "",
            "settings": "",
            "mysteries": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if "CHARACTERS" in line or "1. CHARACTERS" in line:
                current_section = "characters"
                continue
            elif "PLOT THREADS" in line or "2. PLOT THREADS" in line:
                current_section = "plot_threads"
                continue
            elif "OBJECTS/ITEMS" in line or "3. OBJECTS/ITEMS" in line:
                current_section = "objects"
                continue
            elif "SETTINGS" in line or "4. SETTINGS" in line:
                current_section = "settings"
                continue
            elif "MYSTERIES/QUESTIONS" in line or "5. MYSTERIES/QUESTIONS" in line:
                current_section = "mysteries"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        return {"elements": sections, "tracking_report": response.content}


class DescriptionSpecialistAgent:
    """Description Specialist Agent that enhances sensory and setting details."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "DescriptionSpecialist"

    def enhance_descriptions(
        self, chapter: Chapter, settings: Dict[str, Any]
    ) -> Chapter:
        """Enhance the descriptive elements in a chapter."""
        prompt = PromptTemplate(
            template="""You are a description specialist enhancing the sensory details in a novel chapter.

Chapter Information:
Title: {title}
Setting: {settings}

Original Content:
{content}

Enhance the descriptive elements of this chapter to create a more immersive reading experience.
Focus on:

1. SENSORY DETAILS
   Add rich sensory information (sight, sound, smell, taste, touch)
   
2. SETTING DESCRIPTIONS
   Expand and enrich setting descriptions to create a stronger sense of place
   
3. CHARACTER PHYSICALITY
   Add details about character movements, expressions, and physical reactions
   
4. EMOTIONAL ATMOSPHERE
   Enhance descriptions that convey the emotional atmosphere of scenes
   
5. SYMBOLIC ELEMENTS
   Introduce or enhance descriptive elements that have symbolic resonance

Your enhancements should:
- Blend seamlessly with the existing style and tone
- Feel organic, not forced or purple prose
- Support characterization and plot
- Vary in length and detail based on the importance of the moment
- Use fresh, original language avoiding clichés

Provide the complete enhanced chapter.
""",
            input_variables=["title", "settings", "content"],
        )

        # Format settings information
        settings_text = ""
        for name, setting in settings.items():
            settings_text += (
                f"{name}: {setting.get('description', 'No description available')}\n"
            )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title, settings=settings_text, content=chapter.content
            )
        )

        # Update the chapter
        enhanced_chapter = chapter.model_copy()
        enhanced_chapter.content = response.content
        enhanced_chapter.word_count = len(response.content.split())

        return enhanced_chapter

    def analyze_description_quality(self, text: str) -> Dict[str, Any]:
        """Analyze the quality of descriptions in a text."""
        prompt = PromptTemplate(
            template="""You are a literary analyst specializing in descriptive writing. Analyze the following text for the quality of its descriptive elements.

Text:
{text}

Analyze this text for:

1. SENSORY BALANCE
   Evaluate the balance of different sensory details (visual, auditory, olfactory, gustatory, tactile)
   
2. DESCRIPTIVE TECHNIQUES
   Identify the descriptive techniques used (metaphor, simile, personification, etc.)
   
3. SETTING IMMERSION
   Assess how effectively the setting is brought to life
   
4. CHARACTER PHYSICALITY
   Evaluate how well character physical presence is conveyed
   
5. EMOTIONAL ATMOSPHERE
   Analyze how description contributes to emotional tone
   
6. DESCRIPTIVE LANGUAGE
   Evaluate the originality and effectiveness of the descriptive language

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific examples from the text
- Suggestions for improvement

Also provide an overall description quality score and summary.
""",
            input_variables=["text"],
        )

        response = self.llm.invoke(
            prompt.format(text=text[:3000])
        )  # Limit text length for token constraints

        # Extract scores from the response
        scores = {}
        aspects = [
            "sensory balance",
            "descriptive techniques",
            "setting immersion",
            "character physicality",
            "emotional atmosphere",
            "descriptive language",
        ]

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
            "overall_score": overall_score,
        }
