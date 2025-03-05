from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.voice import VoiceConsistencyTool
from storybook.agents.base_agent import BaseAgent

class VoiceConsistencyMonitor(BaseAgent):
    """Agent responsible for monitoring voice consistency."""
    
    def __init__(self):
        super().__init__(
            name="voice_consistency_monitor",
            tools=[
                VoiceConsistencyTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Process voice consistency tasks."""
        task = state.current_input.get("task", {})
        
        consistency = await self.tools[0].invoke({
            "content": state.project.content,
            "section_id": task.get("section_id")
        })
        return {
            "messages": [AIMessage(content="Voice consistency check completed")],
            "voice_updates": {"consistency": consistency}
        }
