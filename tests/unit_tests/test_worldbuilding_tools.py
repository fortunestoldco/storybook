import pytest
from typing import Dict, Any

from storybook.tools.worldbuilding import (
    WorldDesignTool,
    SystemDesignTool,
    ConsistencyCheckTool
)

@pytest.mark.asyncio
async def test_world_design_tool():
    """Test world design tool functionality."""
    tool = WorldDesignTool()
    result = await tool.invoke({
        "content": {},
        "world_type": "fantasy"
    })
    
    assert "world_design" in result
    assert all(k in result["world_design"] for k in [
        "type",
        "geography",
        "cultures",
        "history",
        "systems",
        "rules",
        "development_notes"
    ])

@pytest.mark.asyncio
async def test_system_design_tool():
    """Test system design tool functionality."""
    tool = SystemDesignTool()
    result = await tool.invoke({
        "content": {},
        "system_type": "magic"
    })
    
    assert "system_design" in result
    assert all(k in result["system_design"] for k in [
        "type",
        "rules",
        "components",
        "interactions",
        "limitations",
        "implications"
    ])

@pytest.mark.asyncio
async def test_consistency_check_tool():
    """Test worldbuilding consistency checks."""
    tool = ConsistencyCheckTool()
    result = await tool._arun(
        content={},
        element_id="magic_system_1"
    )
    
    assert "consistency" in result
    assert all(k in result["consistency"] for k in [
        "element_id",
        "violations",
        "suggestions"
    ])