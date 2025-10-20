"""
Tests for Workspace class.

Tests cover:
- Directory auto-detection (git repo vs user dir)
- Tool save/load/list/delete
- Agent save/load/list/delete
- Error handling
- Edge cases
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.workspace import (
    Workspace,
    WorkspaceError,
    ToolNotFoundError,
    AgentNotFoundError,
    ToolInfo,
    AgentInfo
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp()
    workspace = Workspace(
        tools_dir=f"{temp_dir}/tools",
        agents_dir=f"{temp_dir}/agents",
        workspace_dir=f"{temp_dir}/workspace"
    )
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


# ============================================================================
# Directory Detection Tests
# ============================================================================

def test_detect_git_repo(temp_dir):
    """Test that workspace detects git repository."""
    # Create fake git repo
    (temp_dir / ".git").mkdir()
    
    with patch('pathlib.Path.cwd', return_value=temp_dir):
        workspace = Workspace()
        assert workspace.tools_dir == temp_dir / "iexplain" / "tools"


def test_detect_no_git_repo(temp_dir):
    """Test that workspace falls back to user directory when not in git repo."""
    with patch('pathlib.Path.cwd', return_value=temp_dir), \
         patch('pathlib.Path.home', return_value=temp_dir):
        workspace = Workspace()
        assert workspace.tools_dir == temp_dir / ".local" / "share" / "iexplain" / "tools"


def test_xdg_data_home_respected(temp_dir):
    """Test that XDG_DATA_HOME is respected."""
    xdg_dir = temp_dir / "custom-data"
    
    with patch('pathlib.Path.cwd', return_value=temp_dir), \
         patch.dict('os.environ', {'XDG_DATA_HOME': str(xdg_dir)}):
        workspace = Workspace()
        assert workspace.tools_dir == xdg_dir / "iexplain" / "tools"


def test_custom_directories():
    """Test that custom directories override auto-detection."""
    custom_tools = "/custom/tools"
    custom_agents = "/custom/agents"
    
    workspace = Workspace(
        tools_dir=custom_tools,
        agents_dir=custom_agents
    )
    
    assert workspace.tools_dir == Path(custom_tools)
    assert workspace.agents_dir == Path(custom_agents)


def test_directories_created(temp_dir):
    """Test that workspace directories are created."""
    workspace = Workspace(
        tools_dir=str(temp_dir / "tools"),
        agents_dir=str(temp_dir / "agents"),
        workspace_dir=str(temp_dir / "workspace")
    )
    
    assert workspace.tools_dir.exists()
    assert workspace.agents_dir.exists()
    assert workspace.workspace_dir.exists()


# ============================================================================
# Tool Management Tests
# ============================================================================

def test_save_tool(temp_workspace):
    """Test saving a tool."""
    code = "def my_tool(x): return x * 2"
    schema = {
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "Test tool"
        }
    }
    
    temp_workspace.save_tool("my_tool", code, schema)
    
    # Check files exist
    assert (temp_workspace.tools_dir / "my_tool.json").exists()
    assert (temp_workspace.tools_dir / "my_tool.py").exists()


def test_save_tool_invalid_name(temp_workspace):
    """Test that invalid tool names are rejected."""
    with pytest.raises(WorkspaceError, match="Invalid tool name"):
        temp_workspace.save_tool("123-invalid", "code", {})
    
    with pytest.raises(WorkspaceError, match="Invalid tool name"):
        temp_workspace.save_tool("invalid-name", "code", {})


def test_load_tool(temp_workspace):
    """Test loading a tool."""
    code = "def my_tool(x): return x * 2"
    schema = {"type": "function"}
    
    temp_workspace.save_tool("my_tool", code, schema)
    
    loaded = temp_workspace.load_tool("my_tool")
    
    assert loaded["name"] == "my_tool"
    assert loaded["code"] == code
    assert loaded["schema"] == schema
    assert "created_at" in loaded
    assert "version" in loaded


def test_load_nonexistent_tool(temp_workspace):
    """Test loading a tool that doesn't exist."""
    with pytest.raises(ToolNotFoundError, match="Tool not found: nonexistent"):
        temp_workspace.load_tool("nonexistent")


def test_load_tool_missing_code_file(temp_workspace):
    """Test loading tool with missing code file."""
    # Create metadata but not code
    metadata = {"name": "incomplete", "schema": {}}
    metadata_file = temp_workspace.tools_dir / "incomplete.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    with pytest.raises(ToolNotFoundError):
        temp_workspace.load_tool("incomplete")


def test_load_tool_invalid_json(temp_workspace):
    """Test loading tool with invalid JSON metadata."""
    # Create invalid JSON file
    metadata_file = temp_workspace.tools_dir / "broken.json"
    code_file = temp_workspace.tools_dir / "broken.py"
    
    with open(metadata_file, 'w') as f:
        f.write("not valid json {")
    
    with open(code_file, 'w') as f:
        f.write("def broken(): pass")
    
    with pytest.raises(WorkspaceError, match="Invalid JSON"):
        temp_workspace.load_tool("broken")


def test_list_tools_empty(temp_workspace):
    """Test listing tools when none exist."""
    tools = temp_workspace.list_tools()
    assert tools == []


def test_list_tools(temp_workspace):
    """Test listing multiple tools."""
    # Create several tools
    for i in range(3):
        temp_workspace.save_tool(
            f"tool_{i}",
            f"def tool_{i}(): pass",
            {"type": "function"}
        )
    
    tools = temp_workspace.list_tools()
    
    assert len(tools) == 3
    assert all(isinstance(t, ToolInfo) for t in tools)
    assert [t.name for t in tools] == ["tool_0", "tool_1", "tool_2"]


def test_list_tools_skips_incomplete(temp_workspace):
    """Test that list_tools skips incomplete tools."""
    # Create complete tool
    temp_workspace.save_tool("complete", "def complete(): pass", {})
    
    # Create incomplete tool (metadata only)
    metadata_file = temp_workspace.tools_dir / "incomplete.json"
    with open(metadata_file, 'w') as f:
        json.dump({"name": "incomplete"}, f)
    
    tools = temp_workspace.list_tools()
    
    assert len(tools) == 1
    assert tools[0].name == "complete"


def test_tool_exists(temp_workspace):
    """Test checking if tool exists."""
    assert not temp_workspace.tool_exists("nonexistent")
    
    temp_workspace.save_tool("exists", "code", {})
    
    assert temp_workspace.tool_exists("exists")


def test_delete_tool(temp_workspace):
    """Test deleting a tool."""
    temp_workspace.save_tool("to_delete", "code", {})
    
    assert temp_workspace.tool_exists("to_delete")
    
    temp_workspace.delete_tool("to_delete")
    
    assert not temp_workspace.tool_exists("to_delete")


def test_delete_nonexistent_tool(temp_workspace):
    """Test deleting a tool that doesn't exist."""
    with pytest.raises(ToolNotFoundError):
        temp_workspace.delete_tool("nonexistent")


# ============================================================================
# Agent Management Tests
# ============================================================================

def test_save_agent(temp_workspace):
    """Test saving an agent."""
    temp_workspace.save_agent(
        name="test_agent",
        system_prompt="You are a test agent",
        tools=["tool1", "tool2"],
        config={"max_iterations": 5}
    )
    
    assert (temp_workspace.agents_dir / "test_agent.json").exists()


def test_save_agent_invalid_name(temp_workspace):
    """Test that invalid agent names are rejected."""
    with pytest.raises(WorkspaceError, match="Invalid agent name"):
        temp_workspace.save_agent("123-invalid", "prompt", [])


def test_load_agent(temp_workspace):
    """Test loading an agent."""
    temp_workspace.save_agent(
        name="test_agent",
        system_prompt="Test prompt",
        tools=["tool1", "tool2"],
        config={"key": "value"}
    )
    
    loaded = temp_workspace.load_agent("test_agent")
    
    assert loaded["name"] == "test_agent"
    assert loaded["system_prompt"] == "Test prompt"
    assert loaded["tools"] == ["tool1", "tool2"]
    assert loaded["config"] == {"key": "value"}
    assert "created_at" in loaded
    assert "version" in loaded


def test_load_nonexistent_agent(temp_workspace):
    """Test loading an agent that doesn't exist."""
    with pytest.raises(AgentNotFoundError, match="Agent not found"):
        temp_workspace.load_agent("nonexistent")


def test_load_agent_invalid_json(temp_workspace):
    """Test loading agent with invalid JSON."""
    agent_file = temp_workspace.agents_dir / "broken.json"
    with open(agent_file, 'w') as f:
        f.write("not valid json")
    
    with pytest.raises(WorkspaceError, match="Invalid JSON"):
        temp_workspace.load_agent("broken")


def test_load_agent_missing_fields(temp_workspace):
    """Test loading agent with missing required fields."""
    agent_file = temp_workspace.agents_dir / "incomplete.json"
    with open(agent_file, 'w') as f:
        json.dump({"name": "incomplete"}, f)  # Missing system_prompt, tools
    
    with pytest.raises(WorkspaceError, match="Missing required field"):
        temp_workspace.load_agent("incomplete")


def test_list_agents_empty(temp_workspace):
    """Test listing agents when none exist."""
    agents = temp_workspace.list_agents()
    assert agents == []


def test_list_agents(temp_workspace):
    """Test listing multiple agents."""
    for i in range(3):
        temp_workspace.save_agent(
            f"agent_{i}",
            f"Prompt {i}",
            [f"tool_{i}"]
        )
    
    agents = temp_workspace.list_agents()
    
    assert len(agents) == 3
    assert all(isinstance(a, AgentInfo) for a in agents)
    assert [a.name for a in agents] == ["agent_0", "agent_1", "agent_2"]


def test_agent_exists(temp_workspace):
    """Test checking if agent exists."""
    assert not temp_workspace.agent_exists("nonexistent")
    
    temp_workspace.save_agent("exists", "prompt", [])
    
    assert temp_workspace.agent_exists("exists")


def test_delete_agent(temp_workspace):
    """Test deleting an agent."""
    temp_workspace.save_agent("to_delete", "prompt", [])
    
    assert temp_workspace.agent_exists("to_delete")
    
    temp_workspace.delete_agent("to_delete")
    
    assert not temp_workspace.agent_exists("to_delete")


def test_delete_nonexistent_agent(temp_workspace):
    """Test deleting an agent that doesn't exist."""
    with pytest.raises(AgentNotFoundError):
        temp_workspace.delete_agent("nonexistent")


# ============================================================================
# Workspace Info Tests
# ============================================================================

def test_get_info(temp_workspace):
    """Test getting workspace information."""
    # Create some tools and agents
    temp_workspace.save_tool("tool1", "code", {})
    temp_workspace.save_agent("agent1", "prompt", [])
    
    info = temp_workspace.get_info()
    
    assert "tools_dir" in info
    assert "agents_dir" in info
    assert "workspace_dir" in info
    assert info["tool_count"] == 1
    assert info["agent_count"] == 1
    assert "is_project" in info


# ============================================================================
# Edge Cases
# ============================================================================

def test_tool_with_special_characters_in_code(temp_workspace):
    """Test saving tool with special characters in code."""
    code = '''
def tool(text: str) -> str:
    """Process text with special chars: <>'"&"""
    return text.upper()
'''
    
    temp_workspace.save_tool("special", code, {})
    loaded = temp_workspace.load_tool("special")
    
    assert loaded["code"] == code


def test_tool_with_unicode(temp_workspace):
    """Test saving tool with unicode characters."""
    code = "def tool(): return '你好世界'"
    
    temp_workspace.save_tool("unicode_tool", code, {})
    loaded = temp_workspace.load_tool("unicode_tool")
    
    assert loaded["code"] == code


def test_agent_with_empty_tools_list(temp_workspace):
    """Test saving agent with no tools."""
    temp_workspace.save_agent("no_tools", "prompt", tools=[])
    loaded = temp_workspace.load_agent("no_tools")
    
    assert loaded["tools"] == []


def test_concurrent_access_same_tool(temp_workspace):
    """Test that the same tool can be loaded multiple times."""
    temp_workspace.save_tool("shared", "code", {})
    
    loaded1 = temp_workspace.load_tool("shared")
    loaded2 = temp_workspace.load_tool("shared")
    
    assert loaded1 == loaded2


def test_path_expansion(temp_dir):
    """Test that ~ is expanded in paths."""
    # Create a path with ~ in it
    home_dir = Path.home()
    custom_path = str(home_dir / "custom" / "tools")
    
    workspace = Workspace(tools_dir="~/custom/tools")
    
    # Check that ~ was expanded (path is absolute)
    assert workspace.tools_dir.is_absolute()
    assert "~" not in str(workspace.tools_dir)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
