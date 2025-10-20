"""
Workspace management for tool and agent persistence.

The Workspace handles:
- Auto-detection of project vs user directory
- Loading and saving tools/agents
- Directory management
- Configuration-based path overrides
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime


class WorkspaceError(Exception):
    """Base exception for workspace-related errors."""
    pass


class ToolNotFoundError(WorkspaceError):
    """Raised when a tool cannot be found."""
    pass


class AgentNotFoundError(WorkspaceError):
    """Raised when an agent cannot be found."""
    pass


@dataclass
class ToolInfo:
    """Metadata about a tool."""
    name: str
    location: Path
    created_at: str
    version: str = "1.0.0"


@dataclass
class AgentInfo:
    """Metadata about an agent."""
    name: str
    location: Path
    created_at: str
    version: str = "1.0.0"


class Workspace:
    """
    Manages tool and agent persistence.
    
    Auto-detects workspace location:
    - If in git repo: ./iexplain/
    - Otherwise: ~/.local/share/iexplain/
    
    Can be overridden via config.
    """
    
    def __init__(
        self,
        tools_dir: Optional[str] = None,
        agents_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None
    ):
        """
        Initialize workspace.
        
        Args:
            tools_dir: Custom tools directory (overrides auto-detection)
            agents_dir: Custom agents directory (overrides auto-detection)
            workspace_dir: Custom sandbox directory (overrides auto-detection)
        """
        # Auto-detect base directory if not provided
        if tools_dir is None:
            base_dir = self._detect_base_dir()
            tools_dir = str(base_dir / "tools")
            agents_dir = str(base_dir / "agents")
            workspace_dir = str(base_dir / "workspace")
        
        self.tools_dir = Path(tools_dir).expanduser()
        self.agents_dir = Path(agents_dir or tools_dir).expanduser()
        self.workspace_dir = Path(workspace_dir or tools_dir).expanduser()
        # TODO: Removed resolve() to prevent issues with tmp/ being resolved into private/tmp/ on macOS. Might consider fixing this later.
        # self.tools_dir = Path(tools_dir).expanduser().resolve()
        # self.agents_dir = Path(agents_dir or tools_dir).expanduser().resolve()
        # self.workspace_dir = Path(workspace_dir or tools_dir).expanduser().resolve()
        
        # Create directories if they don't exist
        self._ensure_directories()
    
    def _detect_base_dir(self) -> Path:
        """
        Auto-detect workspace base directory.
        
        Returns:
            Path to base directory (project or user)
        """
        # Check if we're in a git repository
        if self._in_git_repo():
            return Path.cwd() / "iexplain"
        
        # Fall back to user directory
        return self._get_user_dir() / "iexplain"
    
    def _in_git_repo(self) -> bool:
        """Check if current directory is in a git repository."""
        current = Path.cwd()
        
        # Walk up directory tree looking for .git
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists():
                return True
        
        return False
    
    def _get_user_dir(self) -> Path:
        """Get user data directory (XDG-compliant)."""
        # Check XDG_DATA_HOME environment variable
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        
        if xdg_data_home:
            return Path(xdg_data_home)
        
        # Default to ~/.local/share
        return Path.home() / ".local" / "share"
    
    def _ensure_directories(self) -> None:
        """Create workspace directories if they don't exist."""
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Tool Management
    # ========================================================================
    
    def save_tool(
        self,
        name: str,
        code: str,
        schema: Dict[str, Any],
        version: str = "1.0.0"
    ) -> None:
        """
        Save a tool to disk.
        
        Args:
            name: Tool name (must be valid Python identifier)
            code: Python code implementing the tool
            schema: LLM function schema
            version: Tool version
        
        Raises:
            WorkspaceError: If tool cannot be saved
        """
        # Validate name
        if not name.isidentifier():
            raise WorkspaceError(f"Invalid tool name: {name} (must be valid Python identifier)")
        
        try:
            # Save metadata
            metadata = {
                "name": name,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "schema": schema
            }
            
            metadata_file = self.tools_dir / f"{name}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save code
            code_file = self.tools_dir / f"{name}.py"
            with open(code_file, 'w') as f:
                f.write(code)
        
        except Exception as e:
            raise WorkspaceError(f"Failed to save tool {name}: {e}") from e
    
    def load_tool(self, name: str) -> Dict[str, Any]:
        """
        Load a tool from disk.
        
        Args:
            name: Tool name
        
        Returns:
            Dict with keys: name, code, schema, version, created_at
        
        Raises:
            ToolNotFoundError: If tool doesn't exist
            WorkspaceError: If tool cannot be loaded
        """
        metadata_file = self.tools_dir / f"{name}.json"
        code_file = self.tools_dir / f"{name}.py"
        
        # Check if tool exists
        if not metadata_file.exists() or not code_file.exists():
            raise ToolNotFoundError(f"Tool not found: {name}")
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load code
            with open(code_file, 'r') as f:
                code = f.read()
            
            return {
                "name": metadata["name"],
                "code": code,
                "schema": metadata["schema"],
                "version": metadata.get("version", "1.0.0"),
                "created_at": metadata.get("created_at", "unknown")
            }
        
        except json.JSONDecodeError as e:
            raise WorkspaceError(f"Invalid JSON in tool metadata: {name}") from e
        
        except Exception as e:
            raise WorkspaceError(f"Failed to load tool {name}: {e}") from e
    
    def list_tools(self) -> List[ToolInfo]:
        """
        List all available tools.
        
        Returns:
            List of ToolInfo objects
        """
        tools = []
        
        # Find all .json metadata files
        for metadata_file in self.tools_dir.glob("*.json"):
            name = metadata_file.stem
            code_file = self.tools_dir / f"{name}.py"
            
            # Only include if both metadata and code exist
            if code_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    tools.append(ToolInfo(
                        name=name,
                        location=code_file,
                        created_at=metadata.get("created_at", "unknown"),
                        version=metadata.get("version", "1.0.0")
                    ))
                
                except Exception:
                    # Skip invalid tools
                    continue
        
        return sorted(tools, key=lambda t: t.name)
    
    def tool_exists(self, name: str) -> bool:
        """Check if a tool exists."""
        metadata_file = self.tools_dir / f"{name}.json"
        code_file = self.tools_dir / f"{name}.py"
        return metadata_file.exists() and code_file.exists()
    
    def delete_tool(self, name: str) -> None:
        """
        Delete a tool.
        
        Args:
            name: Tool name
        
        Raises:
            ToolNotFoundError: If tool doesn't exist
        """
        metadata_file = self.tools_dir / f"{name}.json"
        code_file = self.tools_dir / f"{name}.py"
        
        if not self.tool_exists(name):
            raise ToolNotFoundError(f"Tool not found: {name}")
        
        metadata_file.unlink()
        code_file.unlink()
    
    # ========================================================================
    # Agent Management
    # ========================================================================
    
    def save_agent(
        self,
        name: str,
        system_prompt: str,
        tools: List[str],
        config: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0"
    ) -> None:
        """
        Save an agent to disk.
        
        Args:
            name: Agent name
            system_prompt: Agent's system instructions
            tools: List of tool names the agent uses
            config: Optional agent configuration
            version: Agent version
        
        Raises:
            WorkspaceError: If agent cannot be saved
        """
        if not name.isidentifier():
            raise WorkspaceError(f"Invalid agent name: {name}")
        
        try:
            agent_data = {
                "name": name,
                "version": version,
                "created_at": datetime.now().isoformat(),
                "system_prompt": system_prompt,
                "tools": tools,
                "config": config or {}
            }
            
            agent_file = self.agents_dir / f"{name}.json"
            with open(agent_file, 'w') as f:
                json.dump(agent_data, f, indent=2)
        
        except Exception as e:
            raise WorkspaceError(f"Failed to save agent {name}: {e}") from e
    
    def load_agent(self, name: str) -> Dict[str, Any]:
        """
        Load an agent from disk.
        
        Args:
            name: Agent name
        
        Returns:
            Dict with keys: name, system_prompt, tools, config, version, created_at
        
        Raises:
            AgentNotFoundError: If agent doesn't exist
            WorkspaceError: If agent cannot be loaded
        """
        agent_file = self.agents_dir / f"{name}.json"
        
        if not agent_file.exists():
            raise AgentNotFoundError(f"Agent not found: {name}")
        
        try:
            with open(agent_file, 'r') as f:
                agent_data = json.load(f)
            
            return {
                "name": agent_data["name"],
                "system_prompt": agent_data["system_prompt"],
                "tools": agent_data["tools"],
                "config": agent_data.get("config", {}),
                "version": agent_data.get("version", "1.0.0"),
                "created_at": agent_data.get("created_at", "unknown")
            }
        
        except json.JSONDecodeError as e:
            raise WorkspaceError(f"Invalid JSON in agent file: {name}") from e
        
        except KeyError as e:
            raise WorkspaceError(f"Missing required field in agent {name}: {e}") from e
        
        except Exception as e:
            raise WorkspaceError(f"Failed to load agent {name}: {e}") from e
    
    def list_agents(self) -> List[AgentInfo]:
        """
        List all available agents.
        
        Returns:
            List of AgentInfo objects
        """
        agents = []
        
        for agent_file in self.agents_dir.glob("*.json"):
            name = agent_file.stem
            
            try:
                with open(agent_file, 'r') as f:
                    agent_data = json.load(f)
                
                agents.append(AgentInfo(
                    name=name,
                    location=agent_file,
                    created_at=agent_data.get("created_at", "unknown"),
                    version=agent_data.get("version", "1.0.0")
                ))
            
            except Exception:
                # Skip invalid agents
                continue
        
        return sorted(agents, key=lambda a: a.name)
    
    def agent_exists(self, name: str) -> bool:
        """Check if an agent exists."""
        agent_file = self.agents_dir / f"{name}.json"
        return agent_file.exists()
    
    def delete_agent(self, name: str) -> None:
        """
        Delete an agent.
        
        Args:
            name: Agent name
        
        Raises:
            AgentNotFoundError: If agent doesn't exist
        """
        agent_file = self.agents_dir / f"{name}.json"
        
        if not agent_file.exists():
            raise AgentNotFoundError(f"Agent not found: {name}")
        
        agent_file.unlink()
    
    # ========================================================================
    # Workspace Info
    # ========================================================================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get workspace information.
        
        Returns:
            Dict with workspace metadata
        """
        return {
            "tools_dir": str(self.tools_dir),
            "agents_dir": str(self.agents_dir),
            "workspace_dir": str(self.workspace_dir),
            "tool_count": len(self.list_tools()),
            "agent_count": len(self.list_agents()),
            "is_project": self._in_git_repo()
        }
