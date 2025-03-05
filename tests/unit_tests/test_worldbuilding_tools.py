import pytest
from typing import Dict, Any

from storybook.tools.worldbuilding.core import (
    WorldbuildingTool,
    ConsistencyCheckTool
)

@pytest.mark.asyncio
async def test_worldbuilding_tool():
    """Test world element creation."""
    tool = WorldbuildingTool()
    result = await tool._arun(
        content={},
        world_elements={"magic_system": {}}
    )
    
    assert "world" in result
    assert all(k in result["world"] for k in [
        "setting",
        "rules",
        "systems",
        "cultures"
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