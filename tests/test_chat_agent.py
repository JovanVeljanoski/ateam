from unittest.mock import patch
import pytest
from pydantic import Field, BaseModel
from ateam.chat import Agent
from ateam import BaseTool


class SimpleTool(BaseTool):
    """A simple tool for testing agent integration."""

    message: str = Field(..., description="Message to process")

    def run(self) -> str:
        return f"SimpleTool processed: {self.message}"


class CalculatorTool(BaseTool):
    """A calculator tool for testing."""

    operation: str = Field(..., description="Operation to perform (add, subtract)")
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")

    def run(self) -> str:
        if self.operation == "add":
            result = self.a + self.b
        elif self.operation == "subtract":
            result = self.a - self.b
        else:
            return f"Unknown operation: {self.operation}"

        return f"Result: {result}"


class OutputFormat(BaseModel):
    """Output format for structured responses."""
    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(..., description="Confidence level between 0 and 1")


class TestChatAgent:
    """Test the chat Agent class core functionality."""

    def test_agent_initialization_defaults(self):
        """Test agent initialization with default parameters."""
        agent = Agent()

        assert agent.role == 'You are a helpful assistant.'
        assert agent.model == 'gpt-4.1-nano-2025-04-14'
        assert agent.tools == []
        assert agent.max_tool_calls == 11
        assert agent.temperature == 0.85
        assert agent.top_p == 1.0
        assert agent.verbose is False
        assert agent.shared_state is not None
        assert agent.openai_tools_schema == []

    def test_agent_initialization_custom_params(self):
        """Test agent initialization with custom parameters."""
        agent = Agent(
            role="You are a test assistant.",
            model="gpt-3.5-turbo",
            max_tool_calls=5,
            temperature=0.5,
            top_p=0.9,
            verbose=True
        )

        assert agent.role == "You are a test assistant."
        assert agent.model == "gpt-3.5-turbo"
        assert agent.max_tool_calls == 5
        assert agent.temperature == 0.5
        assert agent.top_p == 0.9
        assert agent.verbose is True

    def test_agent_with_tools_basic(self):
        """Test agent initialization with tools."""
        agent = Agent(tools=[SimpleTool])

        assert len(agent.tools) == 1
        assert SimpleTool in agent.tools
        assert len(agent.openai_tools_schema) == 1

        # Verify tool gets agent's shared state
        assert SimpleTool._shared_state is agent.shared_state

    @patch('openai.OpenAI')
    def test_agent_openai_client_initialization(self, mock_openai):
        """Test that OpenAI client is properly initialized."""
        Agent(api_key="test_key", base_url="https://test.com")

        # Verify OpenAI client was called with correct parameters
        mock_openai.assert_called_once_with(api_key="test_key", base_url="https://test.com")

    def test_as_tool_method(self):
        """Test the as_tool method that converts an agent to a tool."""
        agent = Agent(role="You are a calculator assistant.")

        # Convert agent to tool
        AgentTool = agent.as_tool("CalculatorAgent", "An agent that performs calculations")

        # Test the generated tool class
        assert AgentTool.__name__ == "CalculatorAgent"
        assert AgentTool.__doc__ == "An agent that performs calculations"

        # Test tool instantiation
        tool_instance = AgentTool(input="What is 2 + 2?")
        assert tool_instance.input == "What is 2 + 2?"
        assert hasattr(tool_instance, 'run')

    def test_reasoning_parameter_handling(self):
        """Test reasoning parameter handling for different models."""
        # Test with o-model (should set reasoning)
        agent_o = Agent(model="o1-preview", reasoning="high")
        assert agent_o.reasoning == {'effort': 'high'}

        # Test with non-o model (should use NOT_GIVEN)
        agent_gpt = Agent(model="gpt-4", reasoning="high")
        # For non-o models, reasoning should be NOT_GIVEN
        from openai._types import NOT_GIVEN
        assert agent_gpt.reasoning == NOT_GIVEN

        # Test without reasoning parameter
        agent_no_reasoning = Agent()
        assert agent_no_reasoning.reasoning == NOT_GIVEN

    def test_gemini_with_structured_output_raises_error(self):
        """Test that using Gemini model with structured output raises ValueError."""
        with pytest.raises(ValueError, match="Gemini family of models do not support tools and structured output simultaneously."):
            Agent(
                model="gemini-2.0-flash-lite-001",
                output_format=OutputFormat,
                tools=[SimpleTool]
            )


class TestChatAgentToolIntegration:
    """Test integration between chat Agent and tools."""

    def test_tool_shared_state_persistence(self):
        """Test that shared state persists across tool executions."""
        agent = Agent(tools=[SimpleTool])

        # Create multiple tool instances
        tool1 = SimpleTool(message="first")
        tool2 = SimpleTool(message="second")

        # Both should share the same state as the agent
        assert tool1._shared_state is agent.shared_state
        assert tool2._shared_state is agent.shared_state

        # State changes should be visible across all instances
        shared_state1 = tool1._shared_state
        shared_state2 = tool2._shared_state
        if shared_state1 is not None:
            shared_state1.set("counter", 1)
        if shared_state2 is not None:
            assert shared_state2.get("counter") == 1
        assert agent.shared_state.get("counter") == 1

    def test_multiple_tool_types_shared_state(self):
        """Test shared state across different tool types in the same agent."""
        agent = Agent(tools=[SimpleTool, CalculatorTool])

        simple_tool = SimpleTool(message="test")
        calc_tool = CalculatorTool(operation="add", a=1, b=2)

        # All tools should share the agent's state
        assert simple_tool._shared_state is agent.shared_state
        assert calc_tool._shared_state is agent.shared_state

        # State should be shared across different tool types
        simple_shared_state = simple_tool._shared_state
        calc_shared_state = calc_tool._shared_state
        if simple_shared_state is not None:
            simple_shared_state.set("shared_data", "accessible_to_all")
        if calc_shared_state is not None:
            assert calc_shared_state.get("shared_data") == "accessible_to_all"
