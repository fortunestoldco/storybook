from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.domain.verification import FactVerificationTool
from storybook.agents.base_agent import BaseAgent

class FactVerificationSpecialist(BaseAgent):
    """Specialist responsible for verifying factual accuracy in the novel."""
    
    def __init__(self):
        super().__init__(
            name="fact_verification_specialist",
            tools=[FactVerificationTool()]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Verify facts in the novel."""
        task = state.current_input.get("task", {})
        facts = task.get("facts", [])
        domain = task.get("domain", "")
        
        verification_result = await self.tools[0].arun(
            facts=facts,
            domain=domain
        )
        
        return {
            "messages": [AIMessage(content="Fact verification completed")],
            "verification_updates": verification_result
        }
