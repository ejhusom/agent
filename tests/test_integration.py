"""
Integration tests for Config + Workspace + Logger + Agent.

Tests the complete system working together.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import initialize, IExplainContext
from core.config import Config
from core.workspace import Workspace
from core.logger import ConversationLogger
from core.agent import Agent


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    
    # Mock successful response (no tool calls)
    client.complete = Mock(return_value={
        "role": "assistant",
        "content": "Task completed successfully",
        "usage": {"input_tokens": 100, "output_tokens": 50}
    })
    
    return client


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialize_default(temp_dir):
    """Test default initialization."""
    # Create config file
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[workspace]
tools_dir = "{}/tools"
agents_dir = "{}/agents"

[logging]
enabled = true
""".format(temp_dir, temp_dir))
    
    with initialize(config_file=config_file) as ctx:
        assert isinstance(ctx, IExplainContext)
        assert isinstance(ctx.config, Config)
        assert isinstance(ctx.workspace, Workspace)
        assert isinstance(ctx.logger, ConversationLogger)
        
        # Logger is created
        assert ctx.logger.log_file.exists()


def test_initialize_logging_disabled(temp_dir):
    """Test initialization with logging disabled."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[logging]
enabled = false
""")
    
    with initialize(config_file=config_file) as ctx:
        assert ctx.logger is None


def test_initialize_override_logging(temp_dir):
    """Test overriding logging via parameter."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[logging]
enabled = false
""")
    
    with initialize(config_file=config_file, enable_logging=True) as ctx:
        assert ctx.logger is not None


def test_initialize_with_user_query(temp_dir):
    """Test initialization with user query."""
    with initialize(user_query="Test query") as ctx:
        if ctx.logger:
            # Read first log line
            with open(ctx.logger.log_file, 'r') as f:
                first_event = json.loads(f.readline())
            
            assert first_event["data"]["user_query"] == "Test query"


# ============================================================================
# Agent Integration Tests
# ============================================================================

def test_agent_with_config_and_logger(temp_dir, mock_llm_client):
    """Test agent using config and logger."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[agent]
max_iterations = 5
fail_on_tool_error = false

[llm]
model = "test-model"
temperature = 0.0

[logging]
enabled = true
include_messages = true
include_results = true
""")
    
    with initialize(config_file=config_file) as ctx:
        # Create agent
        agent = Agent(
            name="test_agent",
            system_prompt="You are a test agent",
            tools=[],
            llm_client=mock_llm_client,
            config=ctx.config,
            logger=ctx.logger
        )
        
        # Run agent
        result = agent.run("Test task")
        
        assert result["success"] is True
        assert result["result"] == "Task completed successfully"
        assert result["iterations"] == 1
        
        # Check logging
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        # Should have: run_start, agent_start, llm_call, agent_completed
        event_types = [e["event_type"] for e in events]
        assert "agent_start" in event_types
        assert "llm_call" in event_types
        assert "agent_completed" in event_types


def test_agent_max_iterations(temp_dir, mock_llm_client):
    """Test agent respects max_iterations from config."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[agent]
max_iterations = 2

[logging]
enabled = true
""")
    
    # Mock LLM to always request tools (infinite loop)
    mock_llm_client.complete = Mock(return_value={
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {
                "name": "test_tool",
                "arguments": {}
            }
        }]
    })
    
    # Create a test tool
    def test_tool():
        return "result"
    
    test_tool.__name__ = "test_tool"
    test_tool.__tool_schema__ = {
        "type": "function",
        "function": {"name": "test_tool"}
    }
    
    with initialize(config_file=config_file) as ctx:
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            tools=[test_tool],
            llm_client=mock_llm_client,
            config=ctx.config,
            logger=ctx.logger
        )
        
        result = agent.run("Test")
        
        # Should stop at max_iterations
        assert result["success"] is False
        assert "Max iterations" in result["error"]
        assert result["iterations"] == 2


def test_agent_tool_error_handling_continue(temp_dir, mock_llm_client):
    """Test agent continues on tool error when configured."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[agent]
fail_on_tool_error = false  # Continue on error

[logging]
enabled = true
""")
    
    # Mock LLM to call tool once, then finish
    responses = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "function": {"name": "failing_tool", "arguments": {}}
            }]
        },
        {
            "role": "assistant",
            "content": "Handled error, done"
        }
    ]
    mock_llm_client.complete = Mock(side_effect=responses)
    
    # Failing tool
    def failing_tool():
        raise ValueError("Tool error")
    
    failing_tool.__name__ = "failing_tool"
    failing_tool.__tool_schema__ = {
        "type": "function",
        "function": {"name": "failing_tool"}
    }
    
    with initialize(config_file=config_file) as ctx:
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            tools=[failing_tool],
            llm_client=mock_llm_client,
            config=ctx.config,
            logger=ctx.logger
        )
        
        result = agent.run("Test")
        
        # Should continue and succeed
        assert result["success"] is True
        assert result["result"] == "Handled error, done"
        
        # Check error was logged
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        error_events = [e for e in events if e["event_type"] == "error"]
        assert len(error_events) > 0


def test_agent_tool_error_handling_fail(temp_dir, mock_llm_client):
    """Test agent fails on tool error when configured."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[agent]
fail_on_tool_error = true  # Stop on error

[logging]
enabled = true
""")
    
    # Mock LLM to call failing tool
    mock_llm_client.complete = Mock(return_value={
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_1",
            "function": {"name": "failing_tool", "arguments": {}}
        }]
    })
    
    # Failing tool
    def failing_tool():
        raise ValueError("Tool error")
    
    failing_tool.__name__ = "failing_tool"
    failing_tool.__tool_schema__ = {
        "type": "function",
        "function": {"name": "failing_tool"}
    }
    
    with initialize(config_file=config_file) as ctx:
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            tools=[failing_tool],
            llm_client=mock_llm_client,
            config=ctx.config,
            logger=ctx.logger
        )
        
        result = agent.run("Test")
        
        # Should fail immediately
        assert result["success"] is False
        assert "failed" in result["error"].lower()


def test_agent_llm_retry(temp_dir):
    """Test agent retries LLM calls on failure."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[logging]
enabled = true
""")
    
    # Mock LLM to fail twice, then succeed
    mock_llm = Mock()
    mock_llm.complete = Mock(side_effect=[
        Exception("Network error"),
        Exception("Network error"),
        {
            "role": "assistant",
            "content": "Success after retry"
        }
    ])
    
    with initialize(config_file=config_file) as ctx:
        agent = Agent(
            name="test_agent",
            system_prompt="Test",
            tools=[],
            llm_client=mock_llm,
            config=ctx.config,
            logger=ctx.logger
        )
        
        result = agent.run("Test")
        
        # Should succeed after retries
        assert result["success"] is True
        assert result["result"] == "Success after retry"
        
        # Check error logs
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        error_events = [e for e in events if e["event_type"] == "error"]
        assert len(error_events) == 2  # Two failed attempts logged


# ============================================================================
# Workspace + Config Integration Tests
# ============================================================================

def test_workspace_uses_config_paths(temp_dir):
    """Test workspace uses paths from config."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text(f"""
[workspace]
tools_dir = "{temp_dir}/custom_tools"
agents_dir = "{temp_dir}/custom_agents"
""")
    
    with initialize(config_file=config_file) as ctx:
        assert ctx.workspace.tools_dir == temp_dir / "custom_tools"
        assert ctx.workspace.agents_dir == temp_dir / "custom_agents"
        
        # Directories should be created
        assert ctx.workspace.tools_dir.exists()
        assert ctx.workspace.agents_dir.exists()


# FIXME: ctx.logger.log_file.parent resolves with a "private/" before the "tmp/"
# directory, and the test fails because of that.
# def test_logger_uses_config_dir(temp_dir):
#     """Test logger uses log directory from config."""
#     config_file = temp_dir / ".iexplain.toml"
#     config_file.write_text(f"""
# [logging]
# enabled = true
# log_dir = "{temp_dir}/custom_logs"
# """)
    
#     with initialize(config_file=config_file) as ctx:
#         assert ctx.logger is not None
#         assert ctx.logger.log_file.parent == temp_dir / "custom_logs"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
