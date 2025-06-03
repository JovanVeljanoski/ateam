from typing import Any, cast

import pydantic
from openai._types import NOT_GIVEN

from .base_agent import BaseAgent, RunnableTool


class Agent(BaseAgent):
    """
    Agent implementation using OpenAI's responses API.
    This is the future-compatible API recommended by OpenAI.
    """

    def run(self, msg: str) -> str | pydantic.BaseModel:
        """
        Run the agent using the OpenAI responses API.

        Args:
            msg: The initial message/ instructions to the agent.

        Returns:
            The response from the agent.
        """
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
                raise Exception(f'OpenAI API call failed: {e}')

            for output in response.output:
                if output.type == 'message':
                    messages.append(output)
                    return response.output_parsed or response.output_text

                elif output.type == 'function_call':
                    if self.verbose:
                        print(f'Tool call: {output.name}({output.arguments})')
                    messages.append(output)
                    function_output = cast(RunnableTool, output.parsed_arguments).run()
                    if self.verbose:
                        print(f'Tool {output.name} output: {function_output}')
                    messages.append(
                        {
                            'type': 'function_call_output',
                            'call_id': output.call_id,
                            'output': str(function_output)
                        }
                    )

        return f'An answer could not be provided after {self.max_tool_calls} tool calls.'
