from typing import Any, cast

import pydantic
from openai._types import NOT_GIVEN

from ..base_agent import BaseAgent, RunnableTool


class Agent(BaseAgent):
    """
    Agent implementation using OpenAI's chat.completions API.
    This provides compatibility with other model providers.
    """

    def run(self, msg: str) -> str | pydantic.BaseModel:
        """
        Run the agent using the OpenAI chat.completions API.

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
                kwargs = {
                    "messages": messages,
                    "model": self.model,
                    "tools": self.openai_tools_schema,
                    "tool_choice": "auto",
                    "response_format": self.output_format,
                    "reasoning_effort": self.reasoning.get('effort', NOT_GIVEN) if self.reasoning else NOT_GIVEN,
                    "temperature": NOT_GIVEN if self.model.startswith('o') else self.temperature,
                    "top_p": NOT_GIVEN if self.model.startswith('o') else self.top_p,
                }
                if not self.model.startswith("gemini-"):
                    kwargs["store"] = False

                response = self.client.beta.chat.completions.parse(**kwargs)

            except Exception as e:
                raise Exception(f'OpenAI API call failed: {e}')

            for output in response.choices:
                if output.message.tool_calls:
                    messages.append({"role": "assistant", "tool_calls": [dict(t) for t in output.message.tool_calls]})
                    for tool_call in output.message.tool_calls:
                        if self.verbose:
                            print(f'Tool call: {tool_call.function.name}({tool_call.function.arguments})')
                        tool_output = cast(RunnableTool, tool_call.function.parsed_arguments).run()
                        if self.verbose:
                            print(f'Tool {tool_call.function.name} output: {tool_output}')
                        messages.append({
                            "role": "tool",
                            "content": str(tool_output),
                            "tool_call_id": tool_call.id
                        })
                elif output.message.parsed or output.message.content:
                    messages.append({"role": "assistant", "content": output.message.content or str(output.message.parsed)})
                    return output.message.parsed or output.message.content  # type: ignore
                else:
                    raise Exception(f'Unexpected response from OpenAI: {output.message}')

        return f'An answer could not be provided after {self.max_tool_calls} tool calls.'
