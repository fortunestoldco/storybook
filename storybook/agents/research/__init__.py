from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from storybook.utils.state import NovelState, ResearchItem
from storybook.config import Config


class ResearchTopic(BaseModel):
    """Research topic to be investigated."""

    topic: str = Field(description="Name of the research topic")
    description: str = Field(
        description="Detailed description of what needs to be researched"
    )
    relevance: str = Field(description="How this topic is relevant to the novel")
    priority: int = Field(description="Priority from 1-5, with 1 being highest")
    required_depth: int = Field(description="Required research depth from 1-5")


class ResearchSupervisorAgent:
    """Research Supervisor Agent that coordinates the research process."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ResearchSupervisor"

    def identify_research_needs(self, state: NovelState) -> List[ResearchTopic]:
        """Identify what research topics are needed based on the novel concept."""
        parser = PydanticOutputParser(pydantic_object=ResearchTopic)

        prompt = PromptTemplate(
            template="""You are a research coordinator for a novel project. Based on the novel details below, 
identify key research topics that would be necessary to write an authentic and compelling story.

Novel Title: {project_name}
Genre: {genre}
Subgenres: {subgenres}
Premise: {premise}
Themes: {themes}

Identify 5-10 specific research topics that would be essential for this novel. 
Consider historical periods, technical domains, cultural elements, or specialized knowledge areas that would add authenticity.

For each topic, provide:
- Topic name
- Detailed description of what needs to be researched
- Explanation of how this topic is relevant to the novel
- Priority (1-5, with 1 being highest)
- Required depth (1-5, with 5 being most in-depth)

{format_instructions}
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=["project_name", "genre", "subgenres", "premise", "themes"],
        )

        input_data = {
            "project_name": state.project_name,
            "genre": state.genre,
            "subgenres": ", ".join(state.subgenres),
            "premise": state.premise,
            "themes": ", ".join(state.themes),
        }

        response = self.llm.invoke(prompt.format(**input_data))

        try:
            # In a real implementation, we'd handle parsing multiple topics
            # This is simplified for demonstration
            research_topic = parser.parse(response.content)
            return [research_topic]
        except Exception as e:
            print(f"Error parsing research topics: {e}")
            # Fallback to a basic research topic
            return [
                ResearchTopic(
                    topic="General Background Research",
                    description="Basic background information related to the novel's setting and themes",
                    relevance="Provides foundational knowledge for the novel",
                    priority=1,
                    required_depth=3,
                )
            ]

    def evaluate_research(self, research_item: ResearchItem) -> Dict[str, Any]:
        """Evaluate the quality and completeness of a research item."""
        prompt = PromptTemplate(
            template="""You are a research quality evaluator for a novel project.

Research Topic: {topic}
Research Content:
{content}
Sources: {sources}

Evaluate this research on:
1. Accuracy and factual correctness
2. Comprehensiveness (does it cover all important aspects?)
3. Relevance to the novel project
4. Potential for creative application
5. Reliability of sources

Provide a detailed evaluation with a quality score from 0.0 to 1.0, where:
- 0.0-0.3: Inadequate, requires complete rework
- 0.4-0.6: Partial, needs significant additional research
- 0.7-0.8: Good, could use minor additions
- 0.9-1.0: Excellent, complete and ready to use

Also include specific recommendations for any additional information needed.
""",
            input_variables=["topic", "content", "sources"],
        )

        response = self.llm.invoke(
            prompt.format(
                topic=research_item.topic,
                content=research_item.content,
                sources=", ".join(research_item.sources),
            )
        )

        # Simple extraction of score from response
        # In a real implementation, we would parse this more robustly
        score_line = [
            line for line in response.content.split("\n") if "score" in line.lower()
        ]
        score = 0.7  # Default score

        if score_line:
            try:
                score_text = score_line[0]
                score = float(
                    [s for s in score_text.split() if s.replace(".", "").isdigit()][0]
                )
            except:
                pass

        return {
            "evaluation": response.content,
            "quality_score": score,
            "verified": score >= 0.7,
        }

    def compile_research(self, state: NovelState) -> Dict[str, Any]:
        """Compile all research items into a cohesive research document."""
        if not state.research:
            return {"compiled_research": "", "completeness": 0.0}

        # Extract all research items
        research_items = list(state.research.values())

        prompt = PromptTemplate(
            template="""You are a research compiler for a novel project.

You need to compile the following research items into a cohesive, well-organized research document that will serve as a reference for the novel writer.

Novel Project: {project_name}
Genre: {genre}
Premise: {premise}

Research Items:
{research_items}

Create a comprehensive research document that:
1. Organizes the information in a logical structure
2. Highlights key insights relevant to the story
3. Identifies connections between different research topics
4. Notes any remaining research gaps that should be addressed

Format the document with clear headings, subheadings, and sections for easy reference.
""",
            input_variables=["project_name", "genre", "premise", "research_items"],
        )

        # Format research items for the prompt
        formatted_research = ""
        for item in research_items:
            formatted_research += f"TOPIC: {item.topic}\n"
            formatted_research += f"CONTENT: {item.content}\n"
            formatted_research += f"SOURCES: {', '.join(item.sources)}\n\n"

        response = self.llm.invoke(
            prompt.format(
                project_name=state.project_name,
                genre=state.genre,
                premise=state.premise,
                research_items=formatted_research,
            )
        )

        # Calculate completeness based on verified research items
        verified_count = sum(1 for item in research_items if item.verified)
        completeness = verified_count / len(research_items) if research_items else 0.0

        return {"compiled_research": response.content, "completeness": completeness}


class HistoricalResearchAgent:
    """Historical Research Agent that gathers historical information."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "HistoricalResearch"

    def research_historical_period(
        self, period: str, aspects: List[str]
    ) -> ResearchItem:
        """Research a specific historical period focusing on requested aspects."""
        prompt = PromptTemplate(
            template="""You are a historical research specialist gathering information for a novel set in or involving {period}.

The novelist needs detailed information on the following aspects:
{aspects}

Provide comprehensive, accurate historical information about {period} focusing on these aspects.
Include:
1. General overview of the period
2. Specific details about daily life, culture, and society
3. Notable events, figures, and developments
4. Common misconceptions that writers often get wrong
5. Sensory details that would make the setting come alive (sights, sounds, smells, etc.)
6. Language patterns, slang, or terminology specific to the period

Your research must be factually accurate and detailed enough to support authentic worldbuilding.
""",
            input_variables=["period", "aspects"],
        )

        response = self.llm.invoke(
            prompt.format(
                period=period, aspects="\n".join([f"- {aspect}" for aspect in aspects])
            )
        )

        # In a real implementation, we would include actual sources
        sources = [
            "Historical Encyclopedia",
            "Period Studies Journal",
            "Cultural History Database",
        ]

        return ResearchItem(
            topic=f"Historical Period: {period}",
            content=response.content,
            sources=sources,
            relevance_score=0.9,  # Default score that would be refined in a real system
            verified=False,  # Requires verification by supervisor
        )


class TechnicalDomainAgent:
    """Technical Domain Agent that researches specialized knowledge areas."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "TechnicalDomain"

    def research_domain(self, domain: str, specific_topics: List[str]) -> ResearchItem:
        """Research a specialized knowledge domain."""
        prompt = PromptTemplate(
            template="""You are a subject matter expert in {domain} providing specialized knowledge for a novelist.

The novelist needs accurate, detailed information on the following aspects of {domain}:
{specific_topics}

Provide comprehensive information about {domain} focusing on these topics.
Include:
1. Core concepts and terminology
2. Processes, procedures, or methodologies
3. Common practices and techniques
4. Professional jargon and how it's used
5. Common misconceptions in fiction about this domain
6. Concrete details that would make depictions authentic
7. Any ethical considerations a writer should be aware of

Your research should be accurate, accessible to a non-expert, and detailed enough to support authentic writing.
""",
            input_variables=["domain", "specific_topics"],
        )

        response = self.llm.invoke(
            prompt.format(
                domain=domain,
                specific_topics="\n".join([f"- {topic}" for topic in specific_topics]),
            )
        )

        # In a real implementation, we would include actual sources
        sources = ["Technical Journal", "Professional Handbook", "Industry Publication"]

        return ResearchItem(
            topic=f"Technical Domain: {domain}",
            content=response.content,
            sources=sources,
            relevance_score=0.85,  # Default score
            verified=False,  # Requires verification
        )


class CulturalAuthenticityAgent:
    """Cultural Authenticity Agent that ensures cultural representations are accurate."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "CulturalAuthenticity"

    def research_cultural_elements(
        self, culture: str, elements: List[str]
    ) -> ResearchItem:
        """Research cultural elements for authentic representation."""
        prompt = PromptTemplate(
            template="""You are a cultural consultant specializing in {culture}.

The novelist needs authentic information on the following aspects of {culture}:
{elements}

Provide comprehensive, respectful, and accurate information about these elements of {culture}.
Include:
1. Accurate cultural details and practices
2. Historical and contemporary context
3. Variation and diversity within the culture
4. Common stereotypes or misrepresentations to avoid
5. Respectful language and terminology
6. Authentic cultural perspectives
7. Sensitivity considerations for writers outside this culture

Your information should help the writer create authentic, nuanced, and respectful representations.
""",
            input_variables=["culture", "elements"],
        )

        response = self.llm.invoke(
            prompt.format(
                culture=culture,
                elements="\n".join([f"- {element}" for element in elements]),
            )
        )

        # In a real implementation, we would include authoritative cultural sources
        sources = [
            "Cultural Studies Journal",
            "Anthropological Research",
            "Cultural Organization Publication",
        ]

        return ResearchItem(
            topic=f"Cultural Elements: {culture}",
            content=response.content,
            sources=sources,
            relevance_score=0.9,  # Default score
            verified=False,  # Requires verification
        )
