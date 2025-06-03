import importlib.metadata

from .agent import Agent
from .base_tool import BaseTool

# Also import chat module for the new API
from . import chat

try:
    __version__ = importlib.metadata.version("ateam")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development when package is not installed
    __version__ = "0.0.0"

__all__ = ['Agent', 'BaseTool', 'chat', '__version__']
