# storybook/agents/base_agent.py

"""
Base agent implementation for the Storybook system.
"""

import uuid
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.callbacks import CallbackManagerForChainRun

from storybook.config.system_config import AgentConfig
from storybook.tools.nlp.minilm_analyzer import MiniLMAnalyzer

class BaseAgent(ABC):
    """Base class for all agents in the Storybook system."""
    
    def __init__(
        self, 
        name: str,
        description: str,
        system_prompt: str,
        llm: Runnable,
        tools: Dict[str, Any] = None,
        config: AgentConfig = None,
        minilm_analyzer: Optional[MiniLMAnalyzer] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            name: The name of the agent
            description: A description of the agent's role
            system_prompt: The system prompt for the agent
            llm: The language model to use
            tools: Optional dictionary of tools the agent can use
            config: Configuration for the agent
            minilm_analyzer: Optional MiniLM analyzer for self-evaluation
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or {}
        self.config = config or AgentConfig()
        self.minilm_analyzer = minilm_analyzer
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        self.chain = self.prompt_template | self.llm
        
    def run(
        self, 
        input_data: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Run the agent on the input data.
        
        Args:
            input_data: The input data for the agent
            config: Optional configuration for the runnable
            
        Returns:
            The processed output
        """
        # Process the input
        processed_input = self.process_input(input_data)
        
        # Run the chain
        output = self.chain.invoke({"input": processed_input}, config=config)
        output_text = output.content if hasattr(output, 'content') else str(output)
        
        # Self-evaluate if enabled
        if self.config.self_evaluation_enabled and self.minilm_analyzer:
            evaluation = self.self_evaluate(output_text, input_data)
            
            # Improve if needed
            iterations = 0
            while (
                evaluation["overall_score"] < self.get_quality_threshold(input_data) and 
                iterations < self.config.improvement_iterations
            ):
                improved_output = self.improve_output(output_text, evaluation, input_data)
                output_text = improved_output
                evaluation = self.self_evaluate(output_text, input_data)
                iterations += 1
        
        # Process the output
        processed_output = self.process_output(output_text, input_data)
        
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "input": input_data,
            "output": processed_output,
            "raw_output": output_text,
            "evaluation": evaluation if self.config.self_evaluation_enabled and self.minilm_analyzer else None
        }
    
    def process_input(self, input_data: Dict[str, Any]) -> str:
        """
        Process the input data before sending to the LLM.
        
        Args:
            input_data: The input data
            
        Returns:
            Processed input as a string
        """
        # Default implementation just converts to string
        return str(input_data)
    
    def process_output(self, output_text: str, input_data: Dict[str, Any]) -> Any:
        """
        Process the output text after receiving from the LLM.
        
        Args:
            output_text: The output text from the LLM
            input_data: The original input data
            
        Returns:
            Processed output
        """
        # Default implementation just returns the text
        return output_text
    
    def self_evaluate(self, output_text: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Self-evaluate the output using MiniLM analysis.
        
        Args:
            output_text: The output text to evaluate
            input_data: The original input data
            
        Returns:
            Evaluation results
        """
        if not self.minilm_analyzer:
            return {"overall_score": 1.0, "message": "No MiniLM analyzer available"}
        
        criteria = self.get_evaluation_criteria(input_data)
        return self.minilm_analyzer.analyze_output(output_text, criteria)
    
    def improve_output(
        self, 
        output_text: str, 
        evaluation: Dict[str, Any], 
        input_data: Dict[str, Any]
    ) -> str:
        """
        Improve the output based on self-evaluation.
        
        Args:
            output_text: The original output text
            evaluation: The evaluation results
            input_data: The original input data
            
        Returns:
            Improved output text
        """
        improvement_prompt = (
            f"Your previous response needs improvement. Here's the original output:\n\n"
            f"{output_text}\n\n"
            f"Here's the evaluation:\n\n"
            f"{evaluation}\n\n"
            f"Please provide an improved version addressing these issues."
        )
        
        improvement_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.process_input(input_data)),
            AIMessage(content=output_text),
            HumanMessage(content=improvement_prompt)
        ]
        
        improvement_result = self.llm.invoke(improvement_messages)
        return improvement_result.content if hasattr(improvement_result, 'content') else str(improvement_result)
    
    def get_evaluation_criteria(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get criteria for self-evaluation.
        
        Args:
            input_data: The input data
            
        Returns:
            Evaluation criteria
        """
        # Default implementation, should be overridden by subclasses
        return {
            "relevance": {
                "weight": 1.0,
                "description": "How relevant is the response to the input?"
            },
            "completeness": {
                "weight": 1.0,
                "description": "How complete is the response?"
            },
            "coherence": {
                "weight": 1.0,
                "description": "How coherent and well-structured is the response?"
            }
        }
    
    def get_quality_threshold(self, input_data: Dict[str, Any]) -> float:
        """
        Get the quality threshold for the current input.
        
        Args:
            input_data: The input data
            
        Returns:
            Quality threshold between 0 and 1
        """
        # Default implementation
        return 0.75
