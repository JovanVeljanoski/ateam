from typing import Any, Literal, Optional, Type, Protocol
from abc import ABC, abstractmethod

import openai
from openai._types import NOT_GIVEN, NotGiven

import pydantic

from .base_tool import BaseTool, SharedState


class RunnableTool(Protocol):
    def run(self) -> Any: ...


class BaseAgent(ABC):
    def __init__(
            self,
            role: str = 'You are a helpful assistant.',
            model: str = 'gpt-4.1-nano-2025-04-14',
            tools: Optional[list[Type[BaseTool]]] = None,
            max_tool_calls: int = 11,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            output_format: Type[pydantic.BaseModel] | NotGiven = NOT_GIVEN,
            reasoning: Optional[Literal["low", "medium", "high"]] = None,
            temperature: float = 0.85,
            top_p: float = 1.0,
            verbose: bool = False,
    ):
        """
        Initialize an AI agent.

        Args:
            role: The role or persona of the agent. This is essentially the system or developer prompt passed to the underlying LLM.
            model: The model to use.
            tools: A list of tools to use. The tools are pydantic models that inherit from `BaseTool` and implement the `run` method.
            max_tool_calls: The maximum number of tool calls to make. This is the maximum number of times the agent can call a tool.
            api_key: The API key to use. If not provided, the `OPENAI_API_KEY` environment variable is used.
            base_url: The base URL to use.
            output_format: The output format to use. If provided, it should be a pydantic model that inherits from `pydantic.BaseModel`.
            reasoning: The reasoning level to use. Accepted values are "low", "medium", "high".
            temperature: The `temperature` parameter passed to the underlying LLM.
            top_p: The `top_p` parameter passed to the underlying LLM.
            verbose: If true, print the output of every step. Useful for monitoring the agent's behavior.

        Returns:
            The agent instance.
        """

        # Gemini family of models do not support tools and structured output simultaneously
        if model.startswith('gemini-') and (output_format is not NOT_GIVEN) and (tools is not None):
            raise ValueError('Gemini family of models do not support tools and structured output simultaneously.')

        # Currently reasoning is supported only for the o-series from OpenAI
        reasoning_dict = {'effort': reasoning} if reasoning and model.startswith('o') else NOT_GIVEN

        self.role = role
        self.model = model
        self.tools = tools or []
        self.max_tool_calls = max_tool_calls
        self.output_format = output_format
        self.reasoning: Any = reasoning_dict
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        self.shared_state = SharedState()
        self.openai_tools_schema: list = []

        for tool in self.tools:
            self.openai_tools_schema.append(openai.pydantic_function_tool(tool))
            tool._shared_state = self.shared_state

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @abstractmethod
    def run(self, msg: str) -> str | pydantic.BaseModel:
        """
        Run the agent.

        Args:
            msg: The initial message/ instructions to the agent.

        Returns:
            The response from the agent.
        """
        pass

    def as_tool(self, name: str, description: str):
        """
        Convert the agent to a tool.
        This allows the agent to be used as a tool in other agents.

        Args:
            name: The name of the tool.
            description: The description of the tool.

        Returns:
            A pydantic model that inherits from `BaseTool` and implements the `run` method.
        """
        agent_instance = self  # Capture agent reference

        field_definitions = {}
        field_definitions['input'] = (str, pydantic.Field(..., description='The input or instructions'))

        BaseAgentTool = pydantic.create_model(
            f"Base{name}",
            __base__=BaseTool,
            __doc__=description,
            **field_definitions,
        )

        class AgentTool(BaseAgentTool):
            input: str

            def run(self):
                return agent_instance.run(self.input)

        AgentTool.__name__ = name
        AgentTool.__qualname__ = name
        AgentTool.__doc__ = description

        return AgentTool
