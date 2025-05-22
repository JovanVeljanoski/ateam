from typing import Any, Literal, Optional, Type, Protocol, cast

import openai
from openai._types import NOT_GIVEN, NotGiven

import pydantic

from .BaseTool import BaseTool, SharedState


class RunnableTool(Protocol):
    def run(self) -> Any: ...


class Agent:
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
        self.role = role
        self.model = model
        self.tools = tools or []
        self.max_tool_calls = max_tool_calls
        self.output_format = output_format
        self.reasoning: Any = {'effort': reasoning} if reasoning and model.startswith('o') else NOT_GIVEN
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose

        self.shared_state = SharedState()
        self.openai_tools_schema: list = []

        for tool in self.tools:
            self.openai_tools_schema.append(openai.pydantic_function_tool(tool))
            tool._shared_state = self.shared_state

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def run(self, msg: str) -> str:
        messages: list[Any] = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": msg},
        ]

        for _ in range(self.max_tool_calls):

            try:
                response = self.client.responses.parse(
                    input=messages,
                    model=self.model,
                    tools=self.openai_tools_schema,
                    tool_choice="auto",
                    store=False,
                    text_format=self.output_format,
                    reasoning=self.reasoning,
                    temperature=NOT_GIVEN if self.model.startswith('o') else self.temperature,
                    top_p=NOT_GIVEN if self.model.startswith('o') else self.top_p
                )
            except Exception as e:
                print(f"OpenAI API call failed: {e}")
                return 'Error: OpenAI API call failed'

            for output in response.output:
                if output.type == 'message':
                    messages.append(output)
                    return response.output_text

                elif output.type == 'function_call':
                    if self.verbose:
                        print(f'Function call: {output.name}')
                    function_call_msg = {
                        'type': 'function_call',
                        'call_id': output.call_id,
                        'name': output.name,
                        'arguments': output.arguments
                    }
                    messages.append(function_call_msg)
                    function_output = cast(RunnableTool, output.parsed_arguments).run()
                    if self.verbose:
                        print(f'Function {output.name} output: {function_output}')
                    messages.append(
                        {
                            'type': 'function_call_output',
                            'call_id': output.call_id,
                            'output': str(function_output)
                        }
                    )

        return f'An answer could not be provided after {self.max_tool_calls} tool calls.'

    def as_tool(self, name: str, description: str):
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
