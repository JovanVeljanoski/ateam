import pytest
from pydantic import Field, ValidationError

from ateam import BaseTool, Agent
from ateam.base_tool import SharedState


class MockTool(BaseTool):
    """A mock tool for testing purposes."""

    name: str = Field(..., description="Name parameter for testing")
    value: int = Field(default=42, description="Value parameter with default")

    def run(self) -> str:
        return f"MockTool executed with name={self.name}, value={self.value}"


class TestBaseTool:
    """Test the BaseTool class core functionality."""

    def test_tool_creation_and_validation(self):
        """Test basic tool creation and parameter validation."""
        tool = MockTool(name="test")
        assert tool.name == "test"
        assert tool.value == 42  # default value

        tool_with_value = MockTool(name="test2", value=100)
        assert tool_with_value.name == "test2"
        assert tool_with_value.value == 100

    def test_tool_validation_errors(self):
        """Test pydantic validation on tool parameters."""
        # Test missing required field
        with pytest.raises(ValidationError):
            MockTool()  # type: ignore  # missing required 'name' field

        # Test invalid type
        with pytest.raises(ValidationError):
            MockTool(name="test", value="not_an_int")  # type: ignore

    def test_tool_creates_own_shared_state(self):
        """Test that tools create their own shared state when instantiated."""
        # Reset any existing shared state
        MockTool._shared_state = None

        tool = MockTool(name="test")
        # Tool should automatically create shared state for its class
        assert tool._shared_state is not None
        assert isinstance(tool._shared_state, SharedState)

    def test_tool_with_agent_gets_agent_shared_state(self):
        """Test that tools get agent's shared state when used with an agent."""
        agent = Agent(tools=[MockTool])

        # After agent initialization, tool class should have agent's shared state
        assert MockTool._shared_state is agent.shared_state

        # Create tool instance
        tool = MockTool(name="test")
        assert tool._shared_state is agent.shared_state

    def test_run_method_implementation(self):
        """Test the run method implementation."""
        tool = MockTool(name="test_tool", value=123)
        result = tool.run()
        assert result == "MockTool executed with name=test_tool, value=123"

    def test_abstract_method_enforcement(self):
        """Test that BaseTool cannot be instantiated without implementing run."""

        class IncompleteToolClass(BaseTool):  # type: ignore
            name: str = Field(..., description="Name parameter")
            # Missing run method implementation

        # This should raise TypeError because run method is not implemented
        with pytest.raises(TypeError):
            IncompleteToolClass(name="test")  # type: ignore

    def test_tool_config(self):
        """Test that ToolConfig is properly set."""
        tool = MockTool(name="test")
        assert hasattr(tool, 'ToolConfig')
        assert tool.ToolConfig.strict is False
