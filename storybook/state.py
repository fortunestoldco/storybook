from typing import Dict, List, Optional, Union

from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph

from storybook.messages import Audio, Image, Story, System, User

class AgentState(BaseModel):
    """Represents the state of the agent."""

    input: str = Field(..., description="The user input")
    intermediate_steps: Optional[List[Dict]] = Field(
        default_factory=list,
        description="The intermediate steps taken by the agent",
        extra={"widget": {"type": "hidden"}},
    )
    story: Story = Field(default=None, description="The story")
    image: Image = Field(default=None, description="The latest image")

class State(BaseModel):
    """Represents the state of the conversation."""    
    agent: AgentState = Field(default=None, description="The agent state")
    messages: List[Union[User, System, Story, Image, Audio]] = Field(
        default_factory=list, description="The list of messages in the conversation"
    )
    user: str = Field(..., description="user prompt")
    text: str = Field(..., description="text of the story")
    image: Union[str, bytes] = Field(..., description="image of the story")
    audio: Union[str, bytes] = Field(..., description="audio of the story")

    @property
    def graph(self):
        graph = StateGraph(State)
        graph.add_node("agent", lambda x: x.agent)
        graph.set_entry_point("agent")
        return graph
