from typing import Dict, Any
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from storybook.state import NovelSystemState
from storybook.tools.formatting import (
    FormatValidationTool,
    StyleGuideComplianceTool,
    PublishingStandardsTool
)
from storybook.agents.base_agent import BaseAgent

class FormattingStandardsExpert(BaseAgent):
    """Expert ensuring compliance with formatting standards."""
    
    def __init__(self):
        super().__init__(
            name="formatting_standards_expert",
            tools=[
                FormatValidationTool(),
                StyleGuideComplianceTool(),
                PublishingStandardsTool()
            ]
        )
        self._validate_tools()
    
    async def process(
        self,
        state: NovelSystemState,
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Validate and ensure formatting standards."""
        task = state.current_input.get("task", "")
        
        if "style_guide" in task.lower():
            compliance = await self.tools[1].arun(
                content=state.project.content,
                style_guide=state.project.style_preferences,
                format_requirements=task.get("requirements", {})
            )
            return {
                "messages": [AIMessage(content="Style guide compliance verified")],
                "formatting_updates": {"style_compliance": compliance}
            }
        
        if "publishing" in task.lower():
            standards = await self.tools[2].arun(
                content=state.project.content,
                target_format=task.get("format", ""),
                publisher_requirements=task.get("publisher_requirements", {})
            )
            return {
                "messages": [AIMessage(content="Publishing standards verified")],
                "formatting_updates": {"publishing_standards": standards}
            }
        
        # Default to format validation
        validation = await self.tools[0].arun(
            content=state.project.content,
            format_requirements=task.get("requirements", {}),
            style_guide=state.project.style_preferences
        )
        
        return {
            "messages": [AIMessage(content="Format validation completed")],
            "formatting_updates": {"validation": validation}
        }
