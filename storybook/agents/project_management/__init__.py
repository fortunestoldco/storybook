from typing import Dict, List, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from storybook.utils.state import NovelState, ProjectStatus
from storybook.config import storybookConfig


class ProjectConcept(BaseModel):
    """Project concept with market analysis."""

    title: str = Field(description="Proposed title for the novel")
    genre: str = Field(description="Primary genre of the novel")
    subgenres: List[str] = Field(description="Secondary genres or subgenres")
    target_audience: str = Field(
        description="Target audience age range and demographics"
    )
    premise: str = Field(description="1-2 sentence high concept premise")
    market_potential: float = Field(
        description="Estimated market potential score (0-1)"
    )
    uniqueness_factor: float = Field(description="How unique this concept is (0-1)")
    comparable_titles: List[str] = Field(description="Similar successful titles")
    themes: List[str] = Field(description="Main themes to be explored")
    estimated_word_count: int = Field(description="Estimated final word count")


class ProjectLeadAgent:
    """Project Lead Agent that manages the overall novel project."""

    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ProjectLead"

    def initialize_project(
        self, project_name: str, genre: str = None, target_audience: str = None
    ) -> NovelState:
        """Initialize a new novel project."""
        project_id = str(uuid.uuid4())

        return NovelState(
            project_id=project_id,
            project_name=project_name,
            status=ProjectStatus.INITIALIZED,
            genre=genre or "",
            target_audience=target_audience or self.config.target_audience,
            target_word_count=self.config.target_word_count,
        )

    def evaluate_concept(
        self, concept: ProjectConcept, state: NovelState
    ) -> Dict[str, Any]:
        """Evaluate a novel concept and provide feedback."""
        prompt = PromptTemplate(
            template="""You are a literary agent and publishing expert evaluating a novel concept for commercial potential.

Concept:
Title: {title}
Genre: {genre}
Subgenres: {subgenres}
Target Audience: {target_audience}
Premise: {premise}
Comparable Titles: {comparable_titles}
Themes: {themes}

Based on current market trends and reader preferences, evaluate this concept on:
1. Commercial viability
2. Genre fit and market saturation
3. Uniqueness and originality
4. Target audience alignment
5. Thematic resonance

Provide a detailed analysis with specific recommendations for improvement.
""",
            input_variables=[
                "title",
                "genre",
                "subgenres",
                "target_audience",
                "premise",
                "comparable_titles",
                "themes",
            ],
        )

        evaluation_input = {
            "title": concept.title,
            "genre": concept.genre,
            "subgenres": ", ".join(concept.subgenres),
            "target_audience": concept.target_audience,
            "premise": concept.premise,
            "comparable_titles": ", ".join(concept.comparable_titles),
            "themes": ", ".join(concept.themes),
        }

        response = self.llm.invoke(prompt.format(**evaluation_input))

        # Update state with approved concept details
        state.genre = concept.genre
        state.subgenres = concept.subgenres
        state.target_audience = concept.target_audience
        state.premise = concept.premise
        state.themes = concept.themes
        state.target_word_count = concept.estimated_word_count

        # Add message to state log
        state.add_message(
            sender=self.name,
            recipient="System",
            content=f"Project concept evaluated: {concept.title}",
            metadata={"evaluation": response.content, "concept": concept.model_dump()},
        )

        return {
            "evaluation": response.content,
            "approval_score": concept.market_potential * 0.7
            + concept.uniqueness_factor * 0.3,
            "state": state,
        }

    def set_project_phase(
        self, state: NovelState, new_status: ProjectStatus
    ) -> NovelState:
        """Update the project phase/status."""
        state.update_status(new_status)
        state.current_phase = new_status.value
        state.phase_progress = 0.0

        state.add_message(
            sender=self.name,
            recipient="System",
            content=f"Project phase updated to: {new_status.value}",
        )

        return state


class MarketResearchAgent:
    """Market Research Agent that analyzes market trends and reader preferences."""

    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "MarketResearch"

    def analyze_market_trends(self, genres: List[str]) -> Dict[str, Any]:
        """Analyze current market trends for specified genres."""
        prompt = PromptTemplate(
            template="""You are a publishing industry analyst with expertise in market trends.

Conduct a detailed analysis of the current market for the following genres:
{genres}

For each genre, provide:
1. Current popularity (trending up, stable, or declining)
2. Reader demographics
3. Price points and formats (e-book, print, audio) performance
4. Common successful tropes and themes
5. Oversaturated elements to avoid
6. Emerging opportunities or niches
7. Average word count expectations

Format your response as a structured market analysis report with clear sections for each genre.
""",
            input_variables=["genres"],
        )

        response = self.llm.invoke(prompt.format(genres=", ".join(genres)))

        # This would be more sophisticated in a real implementation
        # with actual data sources and trend analysis

        return {
            "market_analysis": response.content,
            "timestamp": datetime.now().isoformat(),
        }

    def generate_novel_concepts(
        self, market_analysis: str, count: int = 3
    ) -> List[ProjectConcept]:
        """Generate novel concepts based on market analysis."""
        parser = PydanticOutputParser(pydantic_object=ProjectConcept)

        prompt = PromptTemplate(
            template="""You are a creative book concept developer with deep understanding of the publishing market.

Based on this market analysis:
{market_analysis}

Generate {count} commercial viable novel concepts that have strong market potential.

For each concept, provide:
- An attention-grabbing title
- Primary genre
- 1-3 subgenres
- Target audience description
- A compelling 1-2 sentence premise
- Market potential score (0-1)
- Uniqueness factor (0-1)
- 3-5 comparable successful titles
- 3-5 main themes to explore
- Estimated word count based on genre expectations

{format_instructions}

Make each concept distinct and oriented toward a different market segment or reader profile.
""",
            partial_variables={"format_instructions": parser.get_format_instructions()},
            input_variables=["market_analysis", "count"],
        )

        response = self.llm.invoke(
            prompt.format(market_analysis=market_analysis, count=count)
        )

        # In a real implementation, we would handle parsing errors
        try:
            concepts = [parser.parse(response.content)]
            return concepts
        except Exception as e:
            print(f"Error parsing concepts: {e}")
            # Fallback to a default concept
            return [
                ProjectConcept(
                    title="Untitled Novel Concept",
                    genre="Fiction",
                    subgenres=["Literary"],
                    target_audience="Adult",
                    premise="A character faces a challenge and grows through the experience.",
                    market_potential=0.5,
                    uniqueness_factor=0.5,
                    comparable_titles=["Similar Book 1", "Similar Book 2"],
                    themes=["Growth", "Change"],
                    estimated_word_count=80000,
                )
            ]


class ConceptDevelopmentAgent:
    """Concept Development Agent that refines novel concepts."""

    def __init__(self, config: storybookConfig):
        self.config = config
        self.llm = ChatOpenAI(**config.get_llm_kwargs())
        self.name = "ConceptDevelopment"

    def refine_concept(self, concept: ProjectConcept) -> ProjectConcept:
        """Refine a novel concept to improve its market potential."""
        prompt = PromptTemplate(
            template="""You are a book concept developer who helps authors refine their novel ideas for maximum market appeal.

Initial Concept:
Title: {title}
Genre: {genre}
Subgenres: {subgenres}
Target Audience: {target_audience}
Premise: {premise}
Comparable Titles: {comparable_titles}
Themes: {themes}
Market Potential: {market_potential}
Uniqueness Factor: {uniqueness_factor}

Refine this concept to increase both its commercial appeal and artistic merit. 
Consider adjusting any element to create a more compelling package.
Provide a revised version with improvements to the title, premise, and thematic focus.

{format_instructions}
""",
            partial_variables={
                "format_instructions": PydanticOutputParser(
                    pydantic_object=ProjectConcept
                ).get_format_instructions()
            },
            input_variables=[
                "title",
                "genre",
                "subgenres",
                "target_audience",
                "premise",
                "comparable_titles",
                "themes",
                "market_potential",
                "uniqueness_factor",
            ],
        )

        concept_dict = concept.model_dump()
        concept_dict["subgenres"] = ", ".join(concept.subgenres)
        concept_dict["comparable_titles"] = ", ".join(concept.comparable_titles)
        concept_dict["themes"] = ", ".join(concept.themes)

        response = self.llm.invoke(prompt.format(**concept_dict))

        try:
            parser = PydanticOutputParser(pydantic_object=ProjectConcept)
            refined_concept = parser.parse(response.content)
            return refined_concept
        except Exception as e:
            print(f"Error parsing refined concept: {e}")
            return concept  # Return original if parsing fails
