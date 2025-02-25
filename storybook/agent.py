from typing import Annotated, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

from storybook.messages import Story, Image
from storybook.tools import get_tools

class AgentInput(BaseModel):
    input: str = Field(..., description="The user input")
    intermediate_steps: Optional[List[Dict]] = Field(
        default_factory=list,
        description="The intermediate steps taken by the agent",
        extra={"widget": {"type": "hidden"}},
    )
    
class AgentState(BaseModel):
    input: str = Field(..., description="The user input")
    intermediate_steps: Optional[List[Dict]] = Field(
        default_factory=list,
        description="The intermediate steps taken by the agent",
        extra={"widget": {"type": "hidden"}},
    )
    story: Story = Field(default=None, description="The story")
    image: Image = Field(default=None, description="The latest image")
    
def get_agent_prompt(system_prompt: str):
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="input", input_mapping=lambda x: x['input']),
            MessagesPlaceholder(variable_name="intermediate_steps", input_mapping=lambda x: x['intermediate_steps']),
        ]
    )

def get_agent_instructions(model: str, system_prompt: str):
    tools = get_tools()
    tool_executor = ToolExecutor(tools)
    prompt = get_agent_prompt(system_prompt)
    llm_with_tools = ChatOpenAI(model=model, temperature=0).bind_tools(tools)

    return {
        "input": lambda x: x["input"],
        "intermediate_steps": lambda x: _format_intermediate_steps(
            x["intermediate_steps"]
        ),
    } | prompt | llm_with_tools | CustomOutputParser() | tool_executor

def _format_intermediate_steps(
    intermediate_steps: List[Dict],
) -> List[BaseMessage]:
    """Format the intermediate steps."""
    formatted: List[BaseMessage] = []

    if intermediate_steps:
        for step in intermediate_steps:
            if not isinstance(step, dict):
                raise TypeError("Each step in intermediate_steps must be a dictionary")

            action = step.get('action')
            observation = step.get('observation')

            if action is None or observation is None:
                raise ValueError("Each step must have both 'action' and 'observation'")

            formatted.append(
                FunctionMessage(content=str(observation), name=action.get("tool"))
            )
    return formatted

def agent_node(
    retriever, llm_chain
) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    """Generate the next instruction in the story."""

    @chain
    def invoke(state: AgentState) -> Dict[str, str]:
        """Generate the next instruction in the story."""
        input_ = state.input
        intermediate_steps_ = state.intermediate_steps
        story = state.story
        image= state.image
        
        # If no story, get context and start story
        if not story:
            context = retriever.invoke({"input": input_})
            response = llm_chain.invoke(
                {
                    "input": input_,
                    "context": context,
                }
            )
            return {"input": response.content}

        # If story, continue story
        else:
            response = llm_chain.invoke(
                {
                    "input": input_,
                    "story": story.content,
                }
            )
            return {"input": response.content}

    return invoke

def generate_story_node(llm_chain):
    @chain
    def invoke(state: AgentState) -> Dict[str, str]:
        """Generate the next instruction in the story."""
        try:
            input_ = state.input
            story = state.story
            image = state.image
        except AttributeError:
            return state
        # If first iteration, generate story
        if not story:
            # Generate title
            response = llm_chain.invoke({"instruction": input_})
            return {"story": Story(content=response, type="story")}

        # If not first iteration, continue story
        else:
            response = llm_chain.invoke({"instruction": input_})
            return {
                "story": Story(content=f"{story.content} {response}", type="story")
            }

    return invoke


def generate_image_node(translator, midjourney):
    @chain
    def invoke(state: AgentState) -> Dict[str, str]:
        """Generate an image based on the story."""
        try:
            input_ = state.input
            story = state.story
            image = state.image
        except AttributeError:
            return state
        
        if story is not None:
            # Translate to English
            if story.content != state.intermediate_steps[-1]["observation"]:
                translation = translator.invoke({"text": story.content})

                # Generate image
                image_description = midjourney.invoke({"text": translation})
                return {"image": Image(content=image_description, type="image")}

    return invoke

class CustomOutputParser:
    def invoke(self, input: BaseMessage) -> ToolInvocation:
        # Parse out the function invocation
        function_call = input.additional_kwargs["tool_calls"][0]
        tool = function_call["function"]["name"]
        input_ = function_call["function"]["arguments"]
        # Extract the arguments
        return ToolInvocation(
            tool=tool,
            tool_input=input_,
        )
