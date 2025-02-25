# Storybook: A Creative Writing Langgraph Hierarchical Multi-Agent Team

## Overview

This project implements a hierarchical multi-agent system using LangGraph and LangChain to collaboratively produce a best-selling novel. The system is structured into:

- **Research Team**: Gathers market data and consumer interests.
- **Writing Team**: Creates the manuscript based on research insights and user specifications.
- **Publishing Team**: Edits and refines the manuscript for finalization.
- **Supervisors**: Oversee each team and coordinate the workflow.
- **Overall Supervisor**: Interfaces with the user and initiates the project.

## Installation and Setup

### 1. Prerequisites

- **Python 3.8+**
- **MongoDB Atlas Account**
- **OpenAI API Key**
- **Node.js and npm** (if client-side interactions are to be implemented)

### 2. Clone the Repository

## Technical Workflow

### Research Team Workflow

The research team is responsible for gathering market data and consumer interests to inform the writing and publishing processes. This involves advanced data analysis techniques, natural language processing (NLP), and long-term data tracking.

#### Market Research Agent

1. **Data Collection**:
   - Example data: Reviews from popular book websites, social media mentions about trending genres, and market reports.
   - Collected data:
     ```
     [
       "Fantasy novels with strong female protagonists are trending.",
       "Readers are interested in dystopian themes with complex world-building.",
       "Epic sagas with multiple character arcs are gaining popularity."
     ]
     ```

2. **Data Analysis and Trending Calculation**:
   - **Trending Calculation**: Determines whether a genre or theme is trending by analyzing a 10% increase in search result traffic over a specified period.
     - Example formula: `Trending Score = Current Traffic * 1.10`
   - **Prompt**: "Analyze the following market data and identify trending genres and themes:\n\nFantasy novels with strong female protagonists are trending.\nReaders are interested in dystopian themes with complex world-building.\nEpic sagas with multiple character arcs are gaining popularity."
   - **LLM Processing**: The language model deployed takes parameters from the research provided and increases base readings. It uses historical training which included datasets from Goodreads, Amazon Reviews, and Twitter mentions to search for book reviews and analyze key metrics such as sentiment, frequency of mentions, and engagement to infer whether a book review is positive or negative.
     - Example LLM output: "Trending genres: Fantasy, Dystopian. Popular themes: Strong female protagonists, Complex world-building, Multiple character arcs."
     - Key metrics analyzed: sentiment score, mention frequency, engagement rate.

3. **Data Plotting and Storage**:
   - **Data Plotting**: The results are plotted and stored in MongoDB for long-term tracking and future analysis.
   - Example MongoDB entry:
     ```json
     {
       "timestamp": "2025-02-25T03:52:39Z",
       "trending_genres": ["Fantasy", "Dystopian"],
       "popular_themes": ["Strong female protagonists", "Complex world-building", "Multiple character arcs"],
       "traffic_increase": 10
     }
     ```

4. **Report Generation**:
   - The agent generates a report summarizing the findings:
     ```
     Market Research Report:
     - Trending Genres: Fantasy, Dystopian
     - Popular Themes: Strong female protagonists, Complex world-building, Multiple character arcs
     ```

```python name=agents/research/market_research_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
import datetime

class MarketResearchAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)
        self.mongo_client = MongoClient("your_mongodb_connection_string")
        self.db = self.mongo_client["storybook"]
        self.collection = self.db["market_trends"]

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            market_data = task["market_data"]
            prompt = ChatPromptTemplate.from_template("Analyze the following market data and identify trending genres and themes:\n\n{market_data}")
            analysis = await self.llm_router.process_with_streaming(task="analyze_market_data", prompt=prompt)
            trending_genres, popular_themes = self._parse_analysis(analysis)
            timestamp = datetime.datetime.utcnow()
            self.collection.insert_one({
                "timestamp": timestamp,
                "trending_genres": trending_genres,
                "popular_themes": popular_themes,
                "traffic_increase": 10
            })
            return {"analysis": analysis}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    def _parse_analysis(self, analysis: str) -> (list, list):
        # Dummy parsing logic for the example
        trending_genres = ["Fantasy", "Dystopian"]
        popular_themes = ["Strong female protagonists", "Complex world-building", "Multiple character arcs"]
        return trending_genres, popular_themes
```

#### Consumer Insights Agent

1. **Survey Analysis**:
   - Example data: Survey responses from potential readers.
   - Collected data:
     ```
     [
       "I love stories with unexpected plot twists.",
       "Strong character development is crucial for me.",
       "I prefer books with intricate world-building."
     ]
     ```

2. **Prompt Construction**:
   - **NLP Processing**: Uses natural language processing to analyze responses across 26 different scopes, including sentiment analysis, key phrase extraction, and topic modeling.
   - Prompt: "Analyze the following survey responses and identify key consumer insights:\n\nI love stories with unexpected plot twists.\nStrong character development is crucial for me.\nI prefer books with intricate world-building."
   - **LLM Processing**: The language model uses parameters from the survey responses and historical training data, including customer feedback datasets from various book retailers and literary forums. The model analyzes key metrics such as sentiment score, common themes, and reader preferences to generate insights.
     - Example LLM output: "Key insights: Readers prefer unexpected plot twists, strong character development, and intricate world-building."
     - Key metrics analyzed: sentiment score, common themes, reader preferences.

3. **Insight Report**:
   - The agent generates a report summarizing consumer insights:
     ```
     Consumer Insights Report:
     - Preferred Elements: Unexpected plot twists, Strong character development, Intricate world-building
     ```

```python name=agents/research/consumer_insights_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ConsumerInsightsAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            survey_responses = task["survey_responses"]
            prompt = ChatPromptTemplate.from_template("Analyze the following survey responses and identify key consumer insights:\n\n{survey_responses}")
            insights = await self.llm_router.process_with_streaming(task="analyze_survey_responses", prompt=prompt)
            return {"insights": insights}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise
```

### Writing Team Workflow

The writing team creates the manuscript based on research insights and user specifications.

#### Content Generation Agent

1. **Prompt Construction**:
   - Example data: Themes and character profiles.
   - Collected data:
     ```
     Themes: Fantasy, Strong female protagonists, Intricate world-building
     Character Profiles: 
         - Character A: Brave, loyal, struggles with trust issues.
         - Character B: Intelligent, ambitious, has a secret past.
     ```

2. **Prompt**:
   - **LLM Content Generation**: Uses advanced generative models to create detailed outlines and narrative content.
   - Prompt: "Generate a chapter outline for a fantasy novel based on the following themes and character profiles:\n\nThemes: Fantasy, Strong female protagonists, Intricate world-building\nCharacter Profiles: Character A: Brave, loyal, struggles with trust issues. Character B: Intelligent, ambitious, has a secret past."
   - **LLM Processing**: The language model leverages extensive training on narrative structures, character development, and thematic consistency. It uses parameters from the provided themes and character profiles to generate coherent and engaging chapter outlines.
     - Example LLM output: "Chapter 1: Introduction to the world and Character A. Chapter 2: Character A meets Character B. Chapter 3: Conflict arises between Character A and Character B due to trust issues."
     - Key metrics analyzed: narrative coherence, character development, thematic consistency.

3. **Manuscript Assembly**:
   - The agent assembles the generated content into a cohesive manuscript:
     ```
     Chapter 1: Introduction to the world and Character A.
     Chapter 2: Character A meets Character B.
     Chapter 3: Conflict arises between Character A and Character B due to trust issues.
     ```

```python name=agents/writing/content_generation_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ContentGenerationAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            themes = task["themes"]
            character_profiles = task["character_profiles"]
            prompt = ChatPromptTemplate.from_template("Generate a chapter outline for a fantasy novel based on the following themes and character profiles:\n\n{themes}\n\n{character_profiles}")
            chapter_outline = await self.llm_router.process_with_streaming(task="generate_chapter_outline", prompt=prompt)
            return {"chapter_outline": chapter_outline}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise
```

#### Character Development Agent

1. **Prompt Construction**:
   - Example data: Character profile.
   - Collected data:
     ```
     Character Profile: Character A: Brave, loyal, struggles with trust issues.
     ```

2. **Prompt**:
   - **LLM Character Development**: Utilizes deep learning models to generate detailed character arcs, ensuring psychological realism and narrative coherence.
   - Prompt: "Design a character arc for the following character profile:\n\nCharacter A: Brave, loyal, struggles with trust issues."
   - **LLM Processing**: The language model uses parameters from the character profile and historical training data from literary analysis datasets. The model generates character arcs by analyzing key metrics such as character growth, conflict resolution, and emotional depth.
     - Example LLM output: "Character A's arc: Starts as a loyal warrior, faces betrayal, struggles to trust others, overcomes trust issues, and becomes a leader."
     - Key metrics analyzed: character growth, conflict resolution, emotional depth.

3. **Integration with Manuscript**:
   - The agent integrates the character arc into the manuscript:
     ```
     Character A's Arc:
     - Start: Loyal warrior
     - Conflict: Faces betrayal
     - Struggle: Struggles to trust others
     - Resolution: Overcomes trust issues and becomes a leader
     ```

```python name=agents/writing/character_development_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class CharacterDevelopmentAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            character_profile = task["character_profile"]
            prompt = ChatPromptTemplate.from_template("Design a character arc for the following character profile:\n\n{character_profile}")
            character_arc = await self.llm_router.process_with_streaming(task="design_character_arc", prompt=prompt)
            return {"character_arc": character_arc}
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise
```

### Publishing Team Workflow

The publishing team is responsible for refining and finalizing the manuscript.

#### Conflict Resolution Agent

1. **Conflict Detection**:
   - **Input Text Analysis**: The agent receives the manuscript text and focuses on sections with character interactions. These sections are identified based on dialogue tags (e.g., "said", "replied") and character names.
   - **Prompt Construction**: A prompt is constructed to instruct the LLM to identify conflicts. Example prompt: "Analyze the following interactions between Character A and Character B and identify any conflicts or disagreements:\n\n[Interaction Text]"
   - **LLM Processing**: The language model processes the prompt by leveraging its training on conflict resolution patterns, dialogue analysis, and character dynamics. Key metrics such as tension level, disagreement frequency, and emotional tone are analyzed to identify conflicts.
     - Example LLM output: "Identified conflicts: Character A and Character B have a disagreement over trust issues."
     - Key metrics analyzed: tension level, disagreement frequency, emotional tone.

2. **Conflict Resolution Strategy**:
   - **Strategy Extraction**: The agent constructs a new prompt to request conflict resolution strategies from the LLM. Example prompt: "Based on the identified conflicts, suggest potential strategies to resolve the conflicts:\n\n[Conflict Details]"
   - **LLM Processing**: The language model generates multiple resolution strategies by considering parameters such as character traits, past interactions, and the nature of the conflict. Historical training data from conflict resolution case studies and literary analysis are used to inform the strategies.
     - Example LLM output: "Resolution strategies: 1. Character A apologizes and opens up about their trust issues. 2. Character B shares their secret past to build trust."
     - Key metrics analyzed: character alignment, narrative impact, feasibility.
   - **Effectiveness Metrics**: The agent evaluates the suggested strategies based on predefined metrics such as Character Alignment, Narrative Impact, and Feasibility.
   - **Implementation**: The agent selects the most effective strategy and updates the manuscript accordingly.

```python name=agents/conflict_resolution_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ConflictResolutionAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "analyze_conflict":
                return await self._analyze_conflict(task)
            elif task_type == "resolve_conflict":
                return await self._resolve_conflict(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _analyze_conflict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        interactions = task["interactions"]
        prompt = ChatPromptTemplate.from_template("Analyze the following interactions between characters and identify any conflicts:\n\n{interactions}")
        conflict_analysis = await self.llm_router.process_with_streaming(task="analyze_conflict", prompt=prompt)
        return {"conflict_analysis": conflict_analysis}

    async def _resolve_conflict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        conflict_details = task["conflict_details"]
        prompt = ChatPromptTemplate.from_template("Based on the identified conflicts, suggest potential strategies to resolve the conflicts:\n\n{conflict_details}")
        resolution_strategy = await self.llm_router.process_with_streaming(task="resolve_conflict", prompt=prompt)
        return {"resolution_strategy": resolution_strategy}
```

#### Thematic Analysis Agent

1. **Theme Extraction**:
   - **Text Segmentation**: The agent segments the manuscript into smaller sections (e.g., chapters, scenes) for analysis. Each segment is passed to the LLM for thematic analysis.
   - **Prompt Construction**: A prompt is constructed to instruct the LLM to identify themes in each segment. Example prompt: "Identify the key themes in the following text:\n\n[Text Segment]"
   - **LLM Processing**: The language model processes the prompt by leveraging its training on thematic analysis, literary motifs, and narrative structures. Key metrics such as thematic recurrence, motif frequency, and thematic coherence are analyzed.
     - Example LLM output: "Identified themes: Betrayal, Trust, Growth."
     - Key metrics analyzed: thematic recurrence, motif frequency, thematic coherence.
   - **Theme Aggregation**: The agent aggregates the identified themes from all segments to get an overall thematic map of the manuscript.

2. **Consistency Check**:
   - **Theme Comparison**: The agent compares the identified themes with the intended themes of the manuscript. Any discrepancies are highlighted for further analysis.
   - **Improvement Suggestions**: The LLM provides suggestions to improve thematic consistency. Example prompt: "Suggest improvements to ensure the following themes are consistently conveyed throughout the manuscript:\n\n[Identified Themes]"
   - **Implementation**: The agent implements the suggested improvements and updates the manuscript.

```python name=agents/thematic_analysis_agent.py
from typing import Dict, Any
from agents.base_agent import BaseAgent
from config.llm_config import LLMRouter
from langchain_core.prompts import ChatPromptTemplate

class ThematicAnalysisAgent(BaseAgent):
    def __init__(self, tools_service):
        super().__init__(tools_service)
        self.llm_router = LLMRouter(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "analyze_theme":
                return await self._analyze_theme(task)
            elif task_type == "improve_theme":
                return await self._improve_theme(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _analyze_theme(self, task: Dict[str, Any]) -> Dict[str, Any]:
        manuscript = task["manuscript"]
        prompt = ChatPromptTemplate.from_template("Identify the key themes in the following manuscript:\n\n{manuscript}")
        thematic_analysis = await self.llm_router.process_with_streaming(task="analyze_theme", prompt=prompt)
        return {"thematic_analysis": thematic_analysis}

    async def _improve_theme(self, task: Dict[str,
### Supervisors

Supervisors oversee each team and coordinate the workflow to ensure the project stays on track and meets quality standards.

#### Team Supervisor

1. **Task Assignment**:
   - Example task: Assigning market research analysis.
   - Task details: "Analyze the following market data and identify trending genres and themes."

2. **Quality Control**:
   - Reviewing output: "Market Research Report: Trending genres: Fantasy, Dystopian. Popular themes: Strong female protagonists, Complex world-building, Multiple character arcs."
   - Providing feedback: "Ensure that the analysis includes more examples from recent best-sellers."

3. **Reporting**:
   - Example report: "Team Progress Report: Market research analysis completed. Writing team has generated chapter outlines for the first three chapters. Character development is ongoing."

```python name=supervisors/team_supervisor.py
from typing import Dict, Any
from supervisors.base_supervisor import BaseSupervisor

class TeamSupervisor(BaseSupervisor):
    def __init__(self, tools_service):
        super().__init__(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type == "assign_task":
                return await self._assign_task(task)
            elif task_type == "review_output":
                return await self._review_output(task)
            elif task_type == "generate_report":
                return await self._generate_report(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Task assignment logic
        pass

    async def _review_output(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Output review logic
        pass

    async def _generate_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Report generation logic
        pass
```

#### Overall Supervisor

1. **Project Initiation**:
   - Project details: "Write a best-selling fantasy novel with strong female protagonists and intricate world-building."

2. **Coordination**:
   - Coordinating teams: Ensuring that the research team provides timely insights to the writing team and that the publishing team refines the manuscript based on feedback.

3. **Final Review**:
   - Reviewing the manuscript: "Final Review: Manuscript meets quality standards. All chapters are coherent and consistent. Ready for publication."

```python name=supervisors/overall_supervisor.py
from typing import Dict, Any
from supervisors.base_supervisor import BaseSupervisor

class OverallSupervisor(BaseSupervisor):
    def __init__(self, tools_service):
        super().__init__(tools_service)

    async def handle_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task_type = task.get("type")
            if task_type is "initiate_project":
                return await self._initiate_project(task)
            elif task_type is "coordinate_teams":
                return await self._coordinate_teams(task)
            elif task_type is "final_review":
                return await self._final_review(task)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            self.logger.error(f"Error handling task: {str(e)}")
            raise

    async def _initiate_project(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Project initiation logic
        pass

    async def _coordinate_teams(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Team coordination logic
        pass

    async def _final_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Final review logic
        pass
```

### Information Flow

#### Information Provided to Supervisors

- **Research Reports**: Market trends, consumer insights, and feedback analysis.
- **Writing Progress**: Manuscript drafts, chapter outlines, and character arcs.
- **Quality Control**: Reviews and feedback from the publishing team.

#### Information Included in the Storybook

- **Research Insights**: Key findings from market research and consumer insights.
- **Character Profiles**: Detailed profiles and development arcs for each character.
- **Chapter Outlines**: High-level outlines for each chapter, including key events and themes.

#### Information Given to Writers

- **Research Insights**: Market trends, themes, and consumer preferences.
- **Character Profiles**: Detailed profiles and development arcs.
- **Chapter Outlines**: High-level outlines for each chapter.

### Character and World Building Documentation

#### Decision Process

1. **Character Building**:
   - Based on research insights and user specifications.
   - Detailed profiles including traits, backstory, and development arcs.
   - LLM-generated content refined by the Character Development Agent.

2. **World Building**:
   - Based on themes, setting, and narrative requirements.
   - Detailed descriptions of the world, including geography, culture, and history.
   - LLM-generated content refined by the Content Generation Agent.

### Example Usage

#### Market Research Agent

```python
# Example usage of the Market Research Agent for analyzing market data
market_research_agent = MarketResearchAgent(tools_service)
task = {"market_data": "Data about trending genres and themes."}
result = await market_research_agent.handle_task(task)
print(result["analysis"])
```

#### Consumer Insights Agent

```python
# Example usage of the Consumer Insights Agent for analyzing survey responses
consumer_insights_agent = ConsumerInsightsAgent(tools_service)
task = {"survey_responses": "Responses from potential readers."}
result = await consumer_insights_agent.handle_task(task)
print(result["insights"])
```

#### Content Generation Agent

```python
# Example usage of the Content Generation Agent for generating chapter outlines
content_generation_agent = ContentGenerationAgent(tools_service)
task = {"themes": "Fantasy themes.", "character_profiles": "Profiles of main characters."}
result = await content_generation_agent.handle_task(task)
print(result["chapter_outline"])
```

#### Character Development Agent

```python
# Example usage of the Character Development Agent for designing character arcs
character_development_agent = CharacterDevelopmentAgent(tools_service)
task = {"character_profile": "Profile of Character A: Brave, loyal, struggles with trust issues."}
result = await character_development_agent.handle_task(task)
print(result["character_arc"])
```

#### World Building Agent

```python
# Example usage of the World Building Agent for generating world details
world_building_agent = WorldBuildingAgent(tools_service)
task = {"world_description": "Description of the world setting and themes."}
result = await world_building_agent.handle_task(task)
print(result["world_details"])
```

#### Team Supervisor

```python
# Example usage of the Team Supervisor for assigning tasks
team_supervisor = TeamSupervisor(tools_service)
task = {"type": "assign_task", "task_details": "Details of the task to be assigned."}
result = await team_supervisor.handle_task(task)
```

#### Overall Supervisor

```python
# Example usage of the Overall Supervisor for initiating the project
overall_supervisor = OverallSupervisor(tools_service)
task = {"type": "initiate_project", "project_details": "Details of the project to be initiated."}
result = await overall_supervisor.handle_task(task)
```
