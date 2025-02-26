# agents/reader_experience/__init__.py
import re
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from storybook.config import Config
from storybook.utils.state import Chapter, NovelState


class EmotionalArcAnalyzerAgent:
    """Emotional Arc Analyzer Agent that verifies emotional impact trajectory."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "EmotionalArcAnalyzer"

    def analyze_emotional_arc(self, state: NovelState) -> Dict[str, Any]:
        """Analyze the emotional arc across the entire novel."""
        prompt = PromptTemplate(
            template="""You are an emotional arc specialist analyzing the emotional journey in a novel.

Novel Details:
Title: {project_name}
Genre: {genre}
Premise: {premise}
Themes: {themes}

Chapter Summaries:
{chapter_summaries}

Analyze the emotional arc of this novel to determine:

1. READER EMOTIONAL JOURNEY
   Map the intended emotional states readers should experience throughout the novel
   Identify emotional highs, lows, and turning points
   
2. EMOTIONAL COHERENCE
   Evaluate whether the emotional progression is logical and satisfying
   Identify any jarring emotional transitions
   
3. THEME-EMOTION ALIGNMENT
   Analyze how emotions reinforce the novel's themes
   Identify missed opportunities for thematic-emotional resonance
   
4. CHARACTER EMOTION MAPPING
   Track how character emotions drive the emotional experience
   Identify points where character and reader emotions should align or diverge
   
5. GENRE EXPECTATIONS
   Evaluate if the emotional arc meets genre expectations
   Suggest adjustments to better satisfy reader expectations

6. EMOTIONAL PAYOFF ASSESSMENT
   Analyze whether emotional setups receive satisfying payoffs
   Identify opportunities to strengthen emotional impact

Create a comprehensive emotional arc analysis with specific recommendations for improvement.
""",
            input_variables=[
                "project_name",
                "genre",
                "premise",
                "themes",
                "chapter_summaries",
            ],
        )

        # Create chapter summaries
        chapter_summaries = ""
        for num, chapter in sorted(state.chapters.items()):
            chapter_summaries += f"Chapter {num}: {chapter.title}\n"
            chapter_summaries += f"Summary: {chapter.summary}\n"
            # Add a brief excerpt to help with emotional analysis
            if chapter.content:
                chapter_summaries += f"Excerpt: {chapter.content[:200]}...\n"
            chapter_summaries += "\n"

        if not chapter_summaries:
            chapter_summaries = "Chapter details not yet available."

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                themes=", ".join(state.themes),
                chapter_summaries=chapter_summaries,
            )
        )

        # Extract sections from the response
        sections = {
            "reader_journey": "",
            "emotional_coherence": "",
            "theme_emotion_alignment": "",
            "character_emotion_mapping": "",
            "genre_expectations": "",
            "emotional_payoff": "",
        }

        current_section = None

        for line in response.content.split("\n"):
            if (
                "READER EMOTIONAL JOURNEY" in line
                or "1. READER EMOTIONAL JOURNEY" in line
            ):
                current_section = "reader_journey"
                continue
            elif "EMOTIONAL COHERENCE" in line or "2. EMOTIONAL COHERENCE" in line:
                current_section = "emotional_coherence"
                continue
            elif (
                "THEME-EMOTION ALIGNMENT" in line
                or "3. THEME-EMOTION ALIGNMENT" in line
            ):
                current_section = "theme_emotion_alignment"
                continue
            elif (
                "CHARACTER EMOTION MAPPING" in line
                or "4. CHARACTER EMOTION MAPPING" in line
            ):
                current_section = "character_emotion_mapping"
                continue
            elif "GENRE EXPECTATIONS" in line or "5. GENRE EXPECTATIONS" in line:
                current_section = "genre_expectations"
                continue
            elif (
                "EMOTIONAL PAYOFF ASSESSMENT" in line
                or "6. EMOTIONAL PAYOFF ASSESSMENT" in line
            ):
                current_section = "emotional_payoff"
                continue

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        # Calculate overall emotional coherence score
        coherence_score = 0.7  # Default score
        score_line = [
            line
            for line in sections["emotional_coherence"].split("\n")
            if "score" in line.lower()
        ]
        if score_line:
            try:
                score_text = score_line[0]
                score = float(
                    [s for s in score_text.split() if s.replace(".", "").isdigit()][0]
                )
                coherence_score = score
            except:
                pass

        return {
            "emotional_analysis": response.content,
            "sections": sections,
            "coherence_score": coherence_score,
        }

    def enhance_emotional_impact(
        self, chapter: Chapter, emotional_goal: str
    ) -> Chapter:
        """Enhance the emotional impact of a chapter to achieve a specific emotional goal."""
        prompt = PromptTemplate(
            template="""You are an emotional impact specialist enhancing the emotional resonance of a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}
Summary: {summary}

Emotional Goal: {emotional_goal}

Original Content:
{content}

Revise this chapter to strengthen its emotional impact and achieve the specified emotional goal. Focus on:

1. EMOTIONAL LANGUAGE
   Enhance language choices to evoke the target emotions
   Use more emotionally resonant sensory details
   
2. CHARACTER EMOTIONAL DEPTH
   Deepen character emotional expressions and reactions
   Add more internal emotional processing where appropriate
   
3. PACING FOR EMOTIONAL IMPACT
   Adjust pacing to heighten emotional moments
   Create appropriate build-up and release
   
4. EMOTIONAL SYMBOLISM
   Add or enhance symbolic elements that reinforce the emotional tone
   Create emotional motifs that build throughout the chapter
   
5. CONTRAST AND AMPLIFICATION
   Create emotional contrasts to highlight the target emotion
   Use scene transitions to build emotional momentum

Your revision should maintain all plot points and character decisions while enhancing only the emotional resonance.

Provide the complete revised chapter.
""",
            input_variables=[
                "title",
                "chapter_number",
                "summary",
                "emotional_goal",
                "content",
            ],
        )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                chapter_number=chapter.number,
                summary=chapter.summary,
                emotional_goal=emotional_goal,
                content=chapter.content,
            )
        )

        # Update the chapter
        enhanced_chapter = chapter.model_copy()
        enhanced_chapter.content = response.content
        enhanced_chapter.word_count = len(response.content.split())

        return enhanced_chapter


class HookOptimizationAgent:
    """Hook Optimization Agent that strengthens chapter openings and closings."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "HookOptimization"

    def optimize_hooks(self, chapter: Chapter) -> Chapter:
        """Optimize the opening and closing hooks of a chapter."""
        prompt = PromptTemplate(
            template="""You are a hook specialist optimizing the opening and closing hooks in a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}
Summary: {summary}

Original Content:
{content}

Enhance the opening and closing hooks of this chapter to maximize reader engagement and create a compelling page-turner effect. Focus on:

1. OPENING HOOK ENHANCEMENT
   Create an irresistible opening paragraph that immediately draws the reader in
   Use mystery, action, striking image, compelling question, or emotional moment
   Establish tone, introduce conflict, or raise story questions
   
2. CLOSING HOOK OPTIMIZATION
   End the chapter with a powerful hook that compels readers to continue
   Create appropriate suspense, raise new questions, or introduce complications
   Ensure the closing provides both satisfaction and forward momentum
   
3. HOOK-CONTENT ALIGNMENT
   Ensure hooks align naturally with the chapter's content and purpose
   Avoid false promises or hooks that feel manipulative
   
4. PACING ADJUSTMENT
   Adjust the pacing leading into and out of hooks for maximum impact
   Create rhythm that emphasizes the hooks

Your optimization should maintain the existing plot and character development while enhancing only the opening and closing sections.

Provide the complete revised chapter.
""",
            input_variables=["title", "chapter_number", "summary", "content"],
        )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                chapter_number=chapter.number,
                summary=chapter.summary,
                content=chapter.content,
            )
        )

        # Update the chapter
        optimized_chapter = chapter.model_copy()
        optimized_chapter.content = response.content
        optimized_chapter.word_count = len(response.content.split())

        return optimized_chapter

    def analyze_hook_effectiveness(self, text: str) -> Dict[str, Any]:
        """Analyze the effectiveness of opening and closing hooks in a text."""
        prompt = PromptTemplate(
            template="""You are a hook specialist analyzing the effectiveness of opening and closing hooks in novel text.

Text:
{text}

Analyze this text for:

1. OPENING HOOK EFFECTIVENESS
   Evaluate how effectively it draws readers in
   Identify the hook type and technique used
   
2. CLOSING HOOK EFFECTIVENESS
   Evaluate how compelling the ending is for continued reading
   Identify the hook type and technique used
   
3. HOOK-CONTENT ALIGNMENT
   Assess whether hooks align with and deliver on their promises
   Evaluate organic integration with the content
   
4. HOOK DISTINCTIVENESS
   Analyze originality and freshness of hook approaches
   Assess memorability and impact

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific analysis of the technique used
- Suggestions for improvement

Also provide an overall hook effectiveness score and summary.
""",
            input_variables=["text"],
        )

        response = self.llm.invoke(
            prompt.format(text=text[:3000])
        )  # Limit text length for token constraints

        # Extract scores from the response
        scores = {}
        aspects = [
            "opening hook effectiveness",
            "closing hook effectiveness",
            "hook-content alignment",
            "hook distinctiveness",
        ]

        for aspect in aspects:
            pattern = rf"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    scores[aspect.replace(" ", "_").replace("-", "_")] = float(
                        match.group(1)
                    )
                except:
                    scores[aspect.replace(" ", "_").replace("-", "_")] = (
                        0.5  # Default if parsing fails
                    )
            else:
                scores[aspect.replace(" ", "_").replace("-", "_")] = (
                    0.5  # Default if not found
                )

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5

        return {
            "analysis": response.content,
            "scores": scores,
            "overall_score": overall_score,
        }


class ReadabilitySpecialistAgent:
    """Readability Specialist Agent that adjusts language complexity for target audience."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ReadabilitySpecialist"

    def adjust_readability(self, chapter: Chapter, target_audience: str) -> Chapter:
        """Adjust the readability of a chapter for a specific target audience."""
        prompt = PromptTemplate(
            template="""You are a readability specialist adjusting a novel chapter for a specific target audience.

Chapter Information:
Title: {title}
Number: {chapter_number}
Target Audience: {target_audience}

Original Content:
{content}

Adjust the readability of this chapter to be optimal for the specified target audience. Focus on:

1. VOCABULARY APPROPRIATENESS
   Adjust word choices to match audience comprehension level
   Ensure specialized terminology is appropriately introduced or simplified
   
2. SENTENCE COMPLEXITY
   Modify sentence length and structure for audience reading level
   Balance simple and complex sentences for appropriate rhythm
   
3. CONCEPT ACCESSIBILITY
   Ensure abstract concepts are presented at appropriate complexity
   Add clarification where needed for the target audience
   
4. ENGAGEMENT ELEMENTS
   Enhance elements that specifically engage this target audience
   Adjust pacing and detail level to match audience preferences
   
5. CULTURAL RELEVANCE
   Ensure references and context are accessible to the target audience
   Provide necessary context for any unfamiliar elements

Your adjustments should maintain the story, plot points, and character development while optimizing only the presentation for readability.

Provide the complete revised chapter.
""",
            input_variables=["title", "chapter_number", "target_audience", "content"],
        )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                chapter_number=chapter.number,
                target_audience=target_audience,
                content=chapter.content,
            )
        )

        # Update the chapter
        adjusted_chapter = chapter.model_copy()
        adjusted_chapter.content = response.content
        adjusted_chapter.word_count = len(response.content.split())

        return adjusted_chapter

    def analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze the readability metrics of a text."""
        prompt = PromptTemplate(
            template="""You are a readability analyst calculating readability metrics for a text.

Text:
{text}

Calculate and analyze the following readability metrics:

1. APPROXIMATE GRADE LEVEL
   Estimate the U.S. school grade level appropriate for this text
   
2. SENTENCE COMPLEXITY
   Analyze average sentence length and complexity
   Identify sentence structure patterns
   
3. VOCABULARY COMPLEXITY
   Assess word difficulty and specialized terminology usage
   Evaluate the balance of common and advanced vocabulary
   
4. PARAGRAPH STRUCTURE
   Analyze paragraph length and organization
   Evaluate flow between paragraphs
   
5. ACCESSIBILITY BARRIERS
   Identify any elements that might create comprehension difficulties
   Suggest clarifications for complex concepts

For each aspect, provide specific data points and examples from the text.

Also provide an overall readability assessment, identifying the ideal target audience age range and education level.
""",
            input_variables=["text"],
        )

        response = self.llm.invoke(
            prompt.format(text=text[:3000])
        )  # Limit text length for token constraints

        # Extract grade level from the response
        grade_level = 8  # Default middle grade level
        grade_pattern = r"grade level.*?(\d+(?:\.\d+)?)"
        grade_match = re.search(grade_pattern, response.content, re.IGNORECASE)
        if grade_match:
            try:
                grade_level = float(grade_match.group(1))
            except:
                pass

        # Extract target audience from the response
        age_range = "Adult"
        age_pattern = r"age range.*?(\d+(?:-\d+)?)"
        age_match = re.search(age_pattern, response.content, re.IGNORECASE)
        if age_match:
            try:
                age_range = age_match.group(1)
            except:
                pass

        return {
            "analysis": response.content,
            "grade_level": grade_level,
            "target_age_range": age_range,
            "full_readability_report": response.content,
        }


class PageTurnerDesignerAgent:
    """Page-Turner Designer Agent that enhances addictive reading qualities."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "PageTurnerDesigner"

    def enhance_page_turner_qualities(self, chapter: Chapter) -> Chapter:
        """Enhance the page-turner qualities of a chapter."""
        prompt = PromptTemplate(
            template="""You are a page-turner designer enhancing the addictive reading qualities of a novel chapter.

Chapter Information:
Title: {title}
Number: {chapter_number}
Summary: {summary}

Original Content:
{content}

Enhance this chapter to maximize its page-turner qualities and make it more addictively readable. Focus on:

1. MICRO-TENSION ENHANCEMENT
   Add moment-to-moment tension in dialogue and action
   Create subtle conflicts and uncertainties that propel the reader forward
   
2. PACING OPTIMIZATION
   Adjust paragraph and sentence length for optimal momentum
   Create rhythm variations that maintain forward propulsion
   
3. CURIOSITY DRIVERS
   Introduce small mysteries and unanswered questions throughout
   Plant subtle foreshadowing that creates anticipation
   
4. IMMERSION DEEPENERS
   Enhance sensory details that pull readers more deeply into scenes
   Create more immediate and present-tense feeling experiences
   
5. PROPULSIVE LANGUAGE
   Adjust word choice and syntax for forward momentum
   Use language techniques that create reading flow and reduce stopping points

Your enhancements should maintain the story, plot points, and character development while optimizing only the addictive reading qualities.

Provide the complete enhanced chapter.
""",
            input_variables=["title", "chapter_number", "summary", "content"],
        )

        response = self.llm.invoke(
            prompt.format(
                title=chapter.title,
                chapter_number=chapter.number,
                summary=chapter.summary,
                content=chapter.content,
            )
        )

        # Update the chapter
        enhanced_chapter = chapter.model_copy()
        enhanced_chapter.content = response.content
        enhanced_chapter.word_count = len(response.content.split())

        return enhanced_chapter

    def analyze_page_turner_qualities(self, text: str) -> Dict[str, Any]:
        """Analyze the page-turner qualities of a text."""
        prompt = PromptTemplate(
            template="""You are a page-turner analyst evaluating the addictive reading qualities of a novel excerpt.

Text:
{text}

Analyze this text for:

1. MICRO-TENSION
   Evaluate moment-to-moment tension in dialogue and interaction
   Assess the presence of subtle conflicts and uncertainties
   
2. PACING EFFECTIVENESS
   Analyze paragraph and sentence rhythm for momentum
   Evaluate scene length and transition quality
   
3. CURIOSITY GENERATION
   Assess how effectively the text creates questions and mysteries
   Evaluate the implementation of information gaps and hooks
   
4. IMMERSION QUALITY
   Analyze the depth and effectiveness of sensory immersion
   Evaluate how present and immediate the reading experience feels
   
5. LANGUAGE PROPULSION
   Assess how word choice and syntax create forward momentum
   Evaluate the flow of language and absence of obstacles to reading

For each aspect, provide:
- A score from 0.0 to 1.0
- Specific examples from the text
- Suggestions for improvement

Also provide an overall "page-turner score" and summary of the text's addictive reading qualities.
""",
            input_variables=["text"],
        )

        response = self.llm.invoke(
            prompt.format(text=text[:3000])
        )  # Limit text length for token constraints

        # Extract scores from the response
        scores = {}
        aspects = [
            "micro-tension",
            "pacing effectiveness",
            "curiosity generation",
            "immersion quality",
            "language propulsion",
        ]

        for aspect in aspects:
            pattern = rf"{aspect}.*?(\d+\.\d+)"
            match = re.search(pattern, response.content, re.IGNORECASE)
            if match:
                try:
                    scores[aspect.replace("-", "_").replace(" ", "_")] = float(
                        match.group(1)
                    )
                except:
                    scores[aspect.replace("-", "_").replace(" ", "_")] = (
                        0.5  # Default if parsing fails
                    )
            else:
                scores[aspect.replace("-", "_").replace(" ", "_")] = (
                    0.5  # Default if not found
                )

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.5

        return {
            "analysis": response.content,
            "scores": scores,
            "overall_score": overall_score,
        }
