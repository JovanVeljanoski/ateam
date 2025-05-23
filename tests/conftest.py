import pytest


@pytest.fixture(autouse=True)
def reset_tool_shared_state():
    """Reset shared state between tests to avoid interference."""
    # This fixture runs automatically before each test
    from ateam.BaseTool import BaseTool

    # Clear any existing shared state on all tool classes
    for cls in BaseTool.__subclasses__():
        if hasattr(cls, '_shared_state'):
            cls._shared_state = None

    yield

    # Clean up after test
    for cls in BaseTool.__subclasses__():
        if hasattr(cls, '_shared_state'):
            cls._shared_state = None
