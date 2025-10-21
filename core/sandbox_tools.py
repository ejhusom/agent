# core/sandbox_tools.py
"""
Standard tools for sandboxed code and command execution.
These can be given to any agent that needs to execute code/commands safely.
"""

from .sandbox import Sandbox
from .config import config

# Shared sandbox instance for agent tools
_agent_sandbox = Sandbox(
    timeout=config.get("sandbox_timeout"),
    workspace=config.get("sandbox_workspace"),
    allow_write=config.get("sandbox_allow_write")
)

def execute_python(code: str) -> dict:
    """Execute Python code in sandbox. Returns output and errors."""
    result = _agent_sandbox.execute(code)
    return {
        "success": result["success"],
        "output": result["output"],
        "error": result["error"]
    }

def run_unix_command(command: str, args: list, input_data: str = None) -> dict:
    """Execute Unix command in sandbox."""
    return _agent_sandbox.execute_command(command, args, input_data)

def run_shell_command(command_line: str) -> dict:
    """Execute shell command line in sandbox."""
    return _agent_sandbox.execute_shell(command_line)

# Tool schemas for registration
SANDBOX_TOOL_SCHEMAS = {
    "execute_python": {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code safely in sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    "run_unix_command": {
        "type": "function",
        "function": {
            "name": "run_unix_command",
            "description": "Execute Unix command (grep, awk, find, etc.) in sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}},
                    "input_data": {"type": "string"}
                },
                "required": ["command", "args"]
            }
        }
    },
    "run_shell_command": {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Execute shell command line with pipes in sandbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "command_line": {"type": "string"}
                },
                "required": ["command_line"]
            }
        }
    }
}