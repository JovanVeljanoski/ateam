import importlib.metadata

from .agent import Agent
from .BaseTool import BaseTool

try:
    __version__ = importlib.metadata.version("ateam")
except importlib.metadata.PackageNotFoundError:
    # Fallback for development when package is not installed
    __version__ = "0.0.0"

__all__ = ['Agent', 'BaseTool', '__version__']
