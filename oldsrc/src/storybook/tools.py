from langchain_core.tools import BaseTool

class ToolsService:
    def __init__(self, llm_router):
        self.llm_router = llm_router

    def get_tool(self, tool_name):
        return BaseTool(self.llm_router, tool_name)
