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
    """
    Base class for all tools. All tools should inherit from this class.
    The `run` method is the actual implementation of the tool, and is the function that will be called when the tool is used.
    Each individual tool should implement its own `run` method.
    """
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
