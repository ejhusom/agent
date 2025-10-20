"""
Tests for integrated Supervisor.

Tests the Supervisor's ability to create tools, create agents,
and delegate tasks, all with full logging and workspace persistence.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import initialize
from core.supervisor import Supervisor


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
    """Create mock LLM client for supervisor."""
    client = Mock()
    return client


@pytest.fixture
def config_file(temp_dir):
    """Create test config file."""
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text("""
[agent]
max_iterations = 10
fail_on_tool_error = false

[logging]
enabled = true
include_messages = true
include_results = true
""")
    return config_file


# ============================================================================
# Supervisor Initialization Tests
# ============================================================================

def test_supervisor_init(temp_dir, mock_llm_client):
    """Test supervisor initialization."""
    # Create fresh config in temp_dir
    config_file = temp_dir / ".iexplain.toml"
    config_file.write_text(f"""
[workspace]
tools_dir = "{temp_dir}/tools"
agents_dir = "{temp_dir}/agents"

[agent]
max_iterations = 10

[logging]
enabled = true
""")
    
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        assert supervisor.name == "supervisor"
        assert supervisor.tools == {}  # No tools in fresh workspace
        assert supervisor.agents == {}


def test_supervisor_loads_existing_tools(config_file, mock_llm_client):
    """Test supervisor loads tools from workspace."""
    with initialize(config_file=config_file) as ctx:
        # Create a tool in workspace
        tool_code = "def test_tool():\n    return 42"
        tool_schema = {
            "type": "function",
            "function": {"name": "test_tool"}
        }
        ctx.workspace.save_tool("test_tool", tool_code, tool_schema)
        
        # Create supervisor - should load the tool
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        assert "test_tool" in supervisor.tools
        assert supervisor.tools["test_tool"]() == 42


# ============================================================================
# Tool Creation Tests
# ============================================================================

def test_create_tool(config_file, mock_llm_client):
    """Test creating a tool."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        # Create tool
        result = supervisor._create_tool(
            name="add",
            description="Add two numbers",
            code="def add(a, b):\n    return a + b",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }
        )
        
        assert "created successfully" in result.lower()
        assert "add" in supervisor.tools
        
        # Test tool works
        assert supervisor.tools["add"](2, 3) == 5
        
        # Check tool saved to workspace
        tool_data = ctx.workspace.load_tool("add")
        assert tool_data["name"] == "add"
        assert "def add" in tool_data["code"]


def test_create_tool_logs_event(config_file, mock_llm_client):
    """Test that tool creation is logged."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        supervisor._create_tool(
            name="test_tool",
            description="Test",
            code="def test_tool():\n    return 42",
            parameters={"type": "object"}
        )
        
        # Check log
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        tool_events = [e for e in events if e["event_type"] == "tool_created"]
        assert len(tool_events) == 1
        assert tool_events[0]["data"]["tool_name"] == "test_tool"


def test_create_tool_invalid_code(config_file, mock_llm_client):
    """Test creating tool with invalid code fails gracefully."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        with pytest.raises(SyntaxError):
            supervisor._create_tool(
                name="bad_tool",
                description="Bad",
                code="def bad_tool(\n    invalid syntax",
                parameters={"type": "object"}
            )


# ============================================================================
# Agent Creation Tests
# ============================================================================

def test_create_agent(config_file, mock_llm_client):
    """Test creating an agent."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        # Create a tool first
        supervisor._create_tool(
            name="parse",
            description="Parse data",
            code="def parse(data):\n    return data.split()",
            parameters={"type": "object", "properties": {"data": {"type": "string"}}}
        )
        
        # Create agent
        result = supervisor._create_agent(
            name="parser",
            description="You parse data",
            tool_names=["parse"]
        )
        
        assert "created successfully" in result.lower()
        assert "parser" in supervisor.agents
        
        # Check agent has tool
        agent = supervisor.agents["parser"]
        assert len(agent.tools) == 1
        
        # Check agent saved to workspace
        agent_data = ctx.workspace.load_agent("parser")
        assert agent_data["name"] == "parser"
        assert "parse" in agent_data["tools"]


def test_create_agent_logs_event(config_file, mock_llm_client):
    """Test that agent creation is logged."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        # Create tool
        supervisor._create_tool(
            name="tool1",
            description="Tool",
            code="def tool1():\n    return 1",
            parameters={"type": "object"}
        )
        
        # Create agent
        supervisor._create_agent(
            name="test_agent",
            description="Test agent",
            tool_names=["tool1"]
        )
        
        # Check log
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        agent_events = [e for e in events if e["event_type"] == "agent_created"]
        assert len(agent_events) == 1
        assert agent_events[0]["data"]["agent_name"] == "test_agent"


def test_create_agent_missing_tool(config_file, mock_llm_client):
    """Test creating agent with non-existent tool fails."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        with pytest.raises(ValueError, match="does not exist"):
            supervisor._create_agent(
                name="agent",
                description="Agent",
                tool_names=["nonexistent_tool"]
            )


# ============================================================================
# Delegation Tests
# ============================================================================

def test_delegate_to_agent(config_file):
    """Test delegating task to agent."""
    # Mock LLM that returns simple response
    mock_llm = Mock()
    mock_llm.complete = Mock(return_value={
        "role": "assistant",
        "content": "Task completed: 42"
    })
    
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        # Create tool and agent
        supervisor._create_tool(
            name="compute",
            description="Compute",
            code="def compute():\n    return 42",
            parameters={"type": "object"}
        )
        
        supervisor._create_agent(
            name="worker",
            description="You compute things",
            tool_names=["compute"]
        )
        
        # Delegate
        result = supervisor._delegate_to_agent(
            agent_name="worker",
            task="Compute the answer"
        )
        
        assert result == "Task completed: 42"


def test_delegate_logs_event(config_file):
    """Test that delegation is logged."""
    mock_llm = Mock()
    mock_llm.complete = Mock(return_value={
        "role": "assistant",
        "content": "Done"
    })
    
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        # Create agent
        supervisor._create_tool(
            name="tool1",
            description="Tool",
            code="def tool1():\n    return 1",
            parameters={"type": "object"}
        )
        supervisor._create_agent(
            name="worker",
            description="Worker",
            tool_names=["tool1"]
        )
        
        # Delegate
        supervisor._delegate_to_agent("worker", "Do task")
        
        # Check log
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        delegate_events = [e for e in events if e["event_type"] == "agent_delegated"]
        assert len(delegate_events) == 1
        assert delegate_events[0]["data"]["to_agent"] == "worker"


def test_delegate_to_nonexistent_agent(config_file, mock_llm_client):
    """Test delegating to non-existent agent fails."""
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm_client,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        with pytest.raises(ValueError, match="does not exist"):
            supervisor._delegate_to_agent("nonexistent", "task")


# ============================================================================
# Full Workflow Tests
# ============================================================================

def test_supervisor_full_workflow(config_file):
    """Test complete supervisor workflow: create tool, create agent, delegate."""
    # Mock LLM for supervisor and agent
    responses = [
        # Supervisor decides to create tool
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_1",
                "function": {
                    "name": "create_tool",
                    "arguments": {
                        "name": "parse_log",
                        "description": "Parse log line",
                        "code": "def parse_log(line):\n    return line.split()",
                        "parameters": {
                            "type": "object",
                            "properties": {"line": {"type": "string"}}
                        }
                    }
                }
            }]
        },
        # Supervisor decides to create agent
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_2",
                "function": {
                    "name": "create_agent",
                    "arguments": {
                        "name": "log_analyzer",
                        "description": "You analyze logs",
                        "tool_names": ["parse_log"]
                    }
                }
            }]
        },
        # Supervisor delegates to agent
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_3",
                "function": {
                    "name": "delegate_to_agent",
                    "arguments": {
                        "agent_name": "log_analyzer",
                        "task": "Analyze this log"
                    }
                }
            }]
        },
        # Agent response
        {
            "role": "assistant",
            "content": "Log analyzed: ERROR found"
        },
        # Supervisor returns final result
        {
            "role": "assistant",
            "content": "Analysis complete: ERROR found"
        }
    ]
    
    mock_llm = Mock()
    mock_llm.complete = Mock(side_effect=responses)
    
    with initialize(config_file=config_file) as ctx:
        supervisor = Supervisor(
            llm_client=mock_llm,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        result = supervisor.run("Analyze error.log")
        
        assert result["success"] is True
        assert "ERROR found" in result["result"]
        assert "parse_log" in result["tools_created"]
        assert "log_analyzer" in result["agents_created"]
        
        # Check logs
        with open(ctx.logger.log_file, 'r') as f:
            events = [json.loads(line) for line in f]
        
        event_types = [e["event_type"] for e in events]
        assert "supervisor_start" in event_types
        assert "tool_created" in event_types
        assert "agent_created" in event_types
        assert "agent_delegated" in event_types
        assert "supervisor_completed" in event_types


def test_supervisor_max_iterations(config_file):
    """Test supervisor respects max_iterations."""
    # Track iteration count
    iteration = [0]
    
    def mock_complete(*args, **kwargs):
        iteration[0] += 1
        # Return different tool names each iteration to avoid duplicates
        return {
            "role": "assistant",
            "tool_calls": [{
                "id": f"call_{iteration[0]}",
                "function": {
                    "name": "create_tool",
                    "arguments": {
                        "name": f"tool_{iteration[0]}",
                        "description": "Tool",
                        "code": f"def tool_{iteration[0]}():\n    return {iteration[0]}",
                        "parameters": {"type": "object"}
                    }
                }
            }]
        }
    
    mock_llm = Mock()
    mock_llm.complete = mock_complete
    
    with initialize(config_file=config_file) as ctx:
        # Set low max_iterations
        ctx.config.agent.max_iterations = 2
        
        supervisor = Supervisor(
            llm_client=mock_llm,
            config=ctx.config,
            workspace=ctx.workspace,
            logger=ctx.logger
        )
        
        result = supervisor.run("Task")
        
        assert result["success"] is False
        assert "max iterations" in result["error"].lower()
        assert result["iterations"] == 2


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
