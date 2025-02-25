from agents.writing.world_builder_agent import WorldBuilderAgent
from agents.writing.character_builder_agent import CharacterBuilderAgent
from agents.writing.story_writer_agent import StoryWriterAgent
from agents.writing.dialogue_writer_agent import DialogueWriterAgent
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import json

class WritingTeamSupervisor(BaseAgent):
    def __init__(self, mongodb_service: MongoDBService, **kwargs):
        super().__init__(**kwargs)
        self.mongodb_service = mongodb_service
        self.current_task = None
        self.team_status = {
            "world_builder": "idle",
            "character_builder": "idle",
            "story_writer": "idle",
            "dialogue_writer": "idle"
        }
        self.system_prompts = self._load_system_prompts()
        self.supervision_chains = self._initialize_supervision_chains()
        self.story_writer_agents = self._initialize_story_writer_agents()
        self.dialogue_writer_agents = self._initialize_dialogue_writer_agents()
        self.default_story_writer = self.story_writer_agents["ChatAnthropic"]
        self.default_dialogue_writer = self.dialogue_writer_agents["ChatAnthropic"]

    def _load_system_prompts(self) -> Dict[str, Any]:
        """Load system prompts from configuration file."""
        with open("config/agent_parameters.json", "r") as file:
            return json.load(file)

    def _initialize_supervision_chains(self) -> Dict[str, LLMChain]:
        """Initialize specialized chains for team supervision."""
        chains = {}
        
        # Quality assessment chain
        quality_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("quality_assessment_instruction", "")),
            ("human", "{content}"),
            ("ai", "{assessment}")
        ])
        chains["quality_assessment"] = LLMChain(
            llm=self.llm,
            prompt=quality_prompt,
            verbose=True
        )
        
        # Coordination chain
        coordination_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts.get("team_coordination_instruction", "")),
            ("human", "{task_status}"),
            ("ai", "{coordination_plan}")
        ])
        chains["coordination"] = LLMChain(
            llm=self.llm,
            prompt=coordination_prompt,
            verbose=True
        )
        
        return chains

    def _initialize_story_writer_agents(self) -> Dict[str, StoryWriterAgent]:
        """Initialize different LLM-based story writer agents."""
        return {
            "ChatOpenAI": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOpenAI(),
                name="OpenAI_Writer"
            ),
            "ChatAnthropic": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatAnthropic(),
                name="Anthropic_Writer"
            ),
            "ChatHuggingFace": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatHuggingFace(),
                name="HF_Writer"
            ),
            "ChatGoogleGenerativeAI": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatGoogleGenerativeAI(),
                name="Google_Writer"
            ),
            "ChatOllama": StoryWriterAgent(
                mongodb_service=self.mongodb_service,
                llm=ChatOllama(),
                name="Ollama_Writer"
            )
        }

    def _initialize_dialogue_writer_agents(self) -> Dict[str, DialogueWriterAgent]:
        """Initialize different LLM-based dialogue writer agents."""
        return {
