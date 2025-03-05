import pytest
from typing import Dict, Any

from storybook.tools.base import NovelWritingTool

class TestTool(NovelWritingTool):
    name = "test_tool"
    description = "Test tool implementation"
    
    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return {"result": "test"}

@pytest.mark.asyncio
async def test_tool_invoke():
    """Test tool invocation."""
    tool = TestTool()
    result = await tool.invoke({})
    assert result["result"] == "test"

@pytest.mark.asyncio
async def test_tool_error_handling():
    """Test tool error handling."""
    class ErrorTool(NovelWritingTool):
        name = "error_tool"
        description = "Tool that raises error"
        
        async def _arun(self, **kwargs) -> Dict[str, Any]:
            raise ValueError("Test error")
    
    tool = ErrorTool()
    result = await tool.invoke({})
    assert "error" in result
    assert result["status"] == "failed"