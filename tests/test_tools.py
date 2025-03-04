import pytest
from typing import Dict, Any
from unittest.mock import Mock

from storybook.tools.quality import QualityMetricsTool, QualityGateTool
from storybook.tools.plot import PlotThreadTool, ConflictDesignTool
from storybook.tools.character import PsychologyProfileTool

@pytest.mark.asyncio
async def test_quality_metrics_tool():
    tool = QualityMetricsTool()
    result = await tool._arun(
        content={"chapter_1": "Test content"},
        metrics={"readability": 0.8}
    )
    assert "quality_metrics" in result

@pytest.mark.asyncio
async def test_plot_thread_tool():
    tool = PlotThreadTool()
    result = await tool._arun(
        content={"plot": "Main plot line"},
        plot_id="main_plot"
    )
    assert "thread" in result

@pytest.mark.asyncio
async def test_psychology_profile_tool():
    tool = PsychologyProfileTool()
    result = await tool._arun(
        character_id="protagonist",
        content={"characters": {"protagonist": {"name": "John"}}}
    )
    assert "profile" in result