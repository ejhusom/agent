"""
Supervisor with integrated Config, Workspace, and Logger support.

The Supervisor is a meta-agent that can:
1. Create tools dynamically
2. Create specialized agents
3. Delegate tasks to agents
4. Coordinate multi-agent workflows

All actions are logged and persisted to the workspace.
"""

from typing import Dict, Any, List
from pathlib import Path

from .config import Config
from .workspace import Workspace
from .logger import ConversationLogger
from .agent import Agent
from .config import config
from .llm_client import LLMClient
from .sandbox import Sandbox
from .standard_tools import get_standard_tools
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry

class Supervisor:
    """
    Meta-agent that creates tools and agents dynamically.
    
    Can create tools, create agents, execute code, and delegate tasks.
    Has access to both meta-tools AND standard tools.
    """
    
    def __init__(
        self,
        llm_client: Any,
        config: Config,
        workspace: Workspace,
        logger: Optional[ConversationLogger] = None
    ):
        """
        Initialize supervisor.
        
        Args:
            llm_client: LLM client for completions
            config: Configuration object
            workspace: Workspace for persistence
            logger: Optional conversation logger
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
        return self.agent.run(message, context)
    
    def _load_system_prompt(self) -> str:
        """Load supervisor system prompt from instructions."""
        prompt_file = self.instructions_dir / "supervisor.md"
        
        iterations = 0
        max_iterations = self.config.agent.max_iterations
        
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
                            "code": {
                                "type": "string",
                                "description": "Python function code"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "JSON schema for function parameters"
                            }
                        },
                        "required": ["name", "description", "code", "parameters"]
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
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_agent",
                    "description": "Delegate a task to an agent",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {
                                "type": "string",
                                "description": "Name of agent to delegate to"
                            },
                            "task": {
                                "type": "string",
                                "description": "Task description for agent"
                            }
                        },
                        "required": ["agent_name", "task"]
                    }
                }
            }
        ]
    
    def _execute_supervisor_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a supervisor meta-tool."""
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        
        start_time = time.time()
        
        try:
            if tool_name == "create_tool":
                result = self._create_tool(**arguments)
            elif tool_name == "create_agent":
                result = self._create_agent(**arguments)
            elif tool_name == "delegate_to_agent":
                result = self._delegate_to_agent(**arguments)
            else:
                raise ValueError(f"Unknown supervisor tool: {tool_name}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.log_tool_call(
                    agent=self.name,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                    success=True,
                    duration_ms=duration_ms,
                    include_result=self.config.logging.include_results
                )
            
            return {"success": True, "result": result}
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            if self.logger:
                self.logger.log_tool_call(
                    agent=self.name,
                    tool_name=tool_name,
                    arguments=arguments,
                    result=None,
                    success=False,
                    duration_ms=duration_ms
                )
                
                self.logger.log_error(
                    agent=self.name,
                    error_type=type(e).__name__,
                    error_message=f"Supervisor tool {tool_name} failed: {e}"
                )
            
            return {"success": False, "error": str(e)}
    
    def _create_tool(
        self,
        name: str,
        description: str,
        code: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Create a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            code: Python function code
            parameters: Parameter schema
        
        Returns:
            Success message
        """
        # Build schema
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        
        # Execute code to get function
        namespace = {}
        exec(code, namespace)
        
        if name not in namespace:
            raise ValueError(f"Code does not define function '{name}'")
        
        tool_func = namespace[name]
        tool_func.__tool_schema__ = schema
        
        # Save to workspace
        self.workspace.save_tool(name, code, schema)
        
        # Add to tools
        self.tools[name] = tool_func
        
        # Log creation
        if self.logger:
            self.logger.log_tool_created(
                agent=self.name,
                tool_name=name,
                code=code,
                schema=schema
            )
        
        return f"Tool '{name}' created successfully"
    
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
            tools=tools,
            llm_client=self.llm_client,
            config=self.config,
            logger=self.logger
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
        task: str
    ) -> Any:
        """
        Delegate a task to an agent.
        
        Args:
            agent_name: Name of agent
            task: Task description
        
        Returns:
            Agent's result
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' does not exist")
        
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
