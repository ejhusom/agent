"""
Supervisor: Main orchestration loop with meta-tools.
"""

from typing import Dict, Any, List
from pathlib import Path
from .agent import Agent
from .config import config
from .llm_client import LLMClient
from .logger import get_logger
from .sandbox import Sandbox
from .standard_tools import get_standard_tools
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

class Supervisor:
    """
    Supervisor agent with meta-tools for self-modification.
    
    Can create tools, create agents, execute code, and delegate tasks.
    Has access to both meta-tools AND standard tools.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        agent_registry: AgentRegistry,
        instructions_dir: str = "instructions"
    ):
        """
        Args:
            llm_client: LLM client for API calls
            tool_registry: Dynamic tool registry
            agent_registry: Dynamic agent registry
            instructions_dir: Directory with markdown instructions
        """
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.agent_registry = agent_registry
        self.sandbox = Sandbox()  # Uses workspace/data by default
        self.instructions_dir = Path(instructions_dir)
        
        # Get standard tools (supervisor has access to these too)
        self.standard_tools = get_standard_tools(self.sandbox)
        
        # Create supervisor agent with both meta-tools and standard tools
        all_tools = {**self._get_meta_tools(), **self.standard_tools}
        
        self.agent = Agent(
            name="supervisor",
            system_prompt=self._load_system_prompt(),
            llm_client=llm_client,
            tools=all_tools,
        )
    
    def run(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute supervisor on a task."""
        # Start logging
        logger = get_logger()
        session_id = logger.start_session(
            task=message,
            config={
                "provider": self.llm_client.provider,
                "model": self.llm_client.model,
                "temperature": self.agent.temperature,
                "max_tokens": self.agent.max_tokens
            }
        )

        result = self.agent.run(message, context)

        # End logging
        log_file = logger.end_session(final_result=result["content"])
        print(f"\nSession log saved to: {log_file}")

        return result
    
    def _load_system_prompt(self) -> str:
        """Load supervisor system prompt from instructions."""
        prompt_file = self.instructions_dir / "supervisor.md"
        
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                return f.read()
        
        # Fallback if file doesn't exist
        return """You are a supervisor agent that orchestrates complex tasks.

You can:
- Create new tools by writing Python code
- Create specialized agents for subtasks
- Execute code and Unix commands
- Read/write files
- Delegate tasks to created agents

Standard tools available: execute_python, run_command, run_shell, read_file, write_file, list_files, pwd

Think step by step:
1. Understand the task
2. Identify what capabilities are needed
3. Create tools/agents as needed
4. Execute the task
5. Return results

Be strategic and efficient."""
    
    def _get_meta_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get meta-tools for supervisor (tools that modify the system)."""
        return {
            "create_tool": {
                "function": self._create_tool,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "create_tool",
                        "description": "Create a new tool by providing Python code. The tool will be tested and registered if valid.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Tool name (lowercase, underscores)"
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Python function code. Must be a complete function definition."
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What the tool does"
                                },
                                "parameters_schema": {
                                    "type": "object",
                                    "description": "JSON schema for function parameters"
                                }
                            },
                            "required": ["name", "code", "description", "parameters_schema"]
                        }
                    }
                }
            },
            "create_agent": {
                "function": self._create_agent,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "create_agent",
                        "description": "Create a specialized agent with specific tools and instructions.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Agent name"
                                },
                                "system_prompt": {
                                    "type": "string",
                                    "description": "Agent's system instructions"
                                },
                                "tools": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of tool names from registry AND/OR standard tools (execute_python, run_command, run_shell, read_file, write_file, list_files, pwd)"
                                }
                            },
                            "required": ["name", "system_prompt", "tools"]
                        }
                    }
                }
            },
            "read_instructions": {
                "function": self._read_instructions,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "read_instructions",
                        "description": "Read a markdown instruction file for guidance.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Filename (e.g., 'tool_creation.md')"
                                }
                            },
                            "required": ["filename"]
                        }
                    }
                }
            },
            "delegate_to_agent": {
                "function": self._delegate_to_agent,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "delegate_to_agent",
                        "description": "Delegate a task to a created agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent_name": {
                                    "type": "string",
                                    "description": "Name of the agent to delegate to"
                                },
                                "task": {
                                    "type": "string",
                                    "description": "Task description for the agent"
                                },
                                "context": {
                                    "type": "object",
                                    "description": "Optional context to provide"
                                }
                            },
                            "required": ["agent_name", "task"]
                        }
                    }
                }
            },
            "list_tools": {
                "function": self._list_tools,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_tools",
                        "description": "List all available tools (from registry and standard tools).",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            },
            "list_agents": {
                "function": self._list_agents,
                "schema": {
                    "type": "function",
                    "function": {
                        "name": "list_agents",
                        "description": "List all created agents.",
                        "parameters": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                }
            }
        }
    
    def _create_tool(
        self,
        name: str,
        code: str,
        description: str,
        parameters_schema: Dict
    ) -> Dict[str, Any]:
        """Create and register a new tool."""
        # Test the code first
        test_result = self.sandbox.execute(code)
        
        if not test_result["success"]:
            return {
                "success": False,
                "error": f"Code execution failed: {test_result['error']}"
            }
        
        # Extract function from code
        try:
            namespace = {}
            exec(code, namespace)
            
            # Find the function
            func = None
            for obj in namespace.values():
                if callable(obj) and hasattr(obj, '__name__') and obj.__name__ != '__builtins__':
                    func = obj
                    break
            
            if not func:
                return {
                    "success": False,
                    "error": "No function found in code"
                }
            
            # Create schema
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters_schema
                }
            }
            
            # Register tool
            self.tool_registry.register(name, func, schema, code)
            
            return {
                "success": True,
                "message": f"Tool '{name}' created and registered"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to create tool: {str(e)}"
            }
    
    def _create_agent(
        self,
        name: str,
        system_prompt: str,
        tools: List[str]
    ) -> Dict[str, Any]:
        """Create and register a new agent."""
        # Collect tools from both registry and standard tools
        agent_tools = {}
        
        for tool_name in tools:
            # Check standard tools first
            if tool_name in self.standard_tools:
                agent_tools[tool_name] = self.standard_tools[tool_name]
            # Then check registry
            else:
                tool = self.tool_registry.get(tool_name)
                if tool:
                    agent_tools[tool_name] = tool
                else:
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found in registry or standard tools"
                    }
        
        # Create agent
        agent = Agent(
            name=name,
            system_prompt=system_prompt,
            llm_client=self.llm_client,
            tools=agent_tools
        )
        
        # Register agent
        config_data = {
            "system_prompt": system_prompt,
            "tools": tools
        }
        self.agent_registry.register(name, agent, config_data)
        
        return {
            "success": True,
            "message": f"Agent '{name}' created with {len(agent_tools)} tools"
        }
    
    def _read_instructions(self, filename: str) -> str:
        """Read instruction file."""
        file_path = self.instructions_dir / filename
        
        if not file_path.exists():
            return f"Error: Instruction file '{filename}' not found"
        
        with open(file_path, 'r') as f:
            return f.read()
    
    def _delegate_to_agent(
        self,
        agent_name: str,
        task: str,
        context: Dict = None
    ) -> Dict[str, Any]:
        """Delegate task to an agent."""
        agent = self.agent_registry.get(agent_name)
        
        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }
        
        result = agent.run(task, context, parent_agent="supervisor")
        
        return {
            "success": True,
            "agent": agent_name,
            "response": result["content"]
        }
    
    def _list_tools(self) -> Dict[str, Any]:
        """List all available tools."""
        registry_tools = self.tool_registry.list_tools()
        standard_tool_names = list(self.standard_tools.keys())
        
        return {
            "registry_tools": registry_tools,
            "standard_tools": standard_tool_names,
            "all_tools": registry_tools + standard_tool_names
        }
    
    def _list_agents(self) -> Dict[str, list]:
        """List all created agents."""
        return {
            "agents": self.agent_registry.list_agents()
        }
