from abc import ABC, abstractmethod
from typing import Any, Optional

import openai.types.beta.threads.runs.tool_call

import pydantic


class SharedState:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        if not isinstance(key, str):
            raise ValueError("`key` must be a string")
        self.data[key] = value

    def get(self, key, default=None):
        if not isinstance(key, str):
            raise ValueError("`key` must be a string")
        return self.data.get(key, default)

    def print_data(self):
        for key, value in self.data.items():
            print(f"{key}: {value}")


class BaseTool(pydantic.BaseModel, ABC):
    _shared_state: Optional[SharedState] = None
    _tool_call: Optional[openai.types.beta.threads.runs.tool_call.ToolCall] = None

    def __init__(self, **kwargs):
        if not self.__class__._shared_state:
            self.__class__._shared_state = SharedState()
        super().__init__(**kwargs)

    class ToolConfig:
        strict: bool = False

    @abstractmethod
    def run(self) -> Any: ...
