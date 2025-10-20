"""
Supervisor with integrated Config, Workspace, and Logger support.

The Supervisor is a meta-agent that can:
1. Create tools dynamically
2. Create specialized agents
3. Delegate tasks to agents
4. Coordinate multi-agent workflows

All actions are logged and persisted to the workspace.
"""

import time
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from .config import Config
from .workspace import Workspace
from .logger import ConversationLogger
from .agent import Agent


class SupervisorError(Exception):
    """Supervisor-related errors."""
    pass


class Supervisor:
    """
    Meta-agent that creates tools and agents dynamically.
    
    Features:
    - Creates tools from LLM-generated code
    - Creates specialized agents with tools
    - Delegates tasks to agents
    - All actions logged and persisted
    - Config-driven behavior
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
        self.config = config
        self.workspace = workspace
        self.logger = logger
        
        self.name = "supervisor"
        
        # Track created tools and agents
        self.tools: Dict[str, Callable] = {}
        self.agents: Dict[str, Agent] = {}
        
        # Load existing tools and agents from workspace
        self._load_from_workspace()
    
    def _load_from_workspace(self) -> None:
        """Load existing tools and agents from workspace."""
        # Load tools
        for tool_file in self.workspace.tools_dir.glob("*.py"):
            try:
                tool_name = tool_file.stem
                tool_data = self.workspace.load_tool(tool_name)
                
                # Execute tool code to get function
                namespace = {}
                exec(tool_data["code"], namespace)
                
                if tool_name in namespace:
                    tool_func = namespace[tool_name]
                    tool_func.__tool_schema__ = tool_data["schema"]
                    self.tools[tool_name] = tool_func
            
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        agent=self.name,
                        error_type=type(e).__name__,
                        error_message=f"Failed to load tool {tool_name}: {e}"
                    )
        
        # Note: Agents are not loaded at init, created on-demand
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run supervisor on a user query.
        
        The supervisor will:
        1. Analyze the query
        2. Create necessary tools
        3. Create specialized agent(s)
        4. Delegate to agent(s)
        5. Return result
        
        Args:
            user_query: User's query/task
        
        Returns:
            Dict with success, result, and metadata
        """
        if self.logger:
            self.logger.log_event(
                event_type="supervisor_start",
                agent=self.name,
                data={"query": user_query}
            )
        
        iterations = 0
        max_iterations = self.config.agent.max_iterations
        
        # Supervisor's system prompt
        system_prompt = self._build_system_prompt()
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        try:
            while iterations < max_iterations:
                iterations += 1
                
                # Get supervisor's next action
                response = self._call_llm_with_retry(messages)
                
                if not response:
                    error_msg = "Supervisor LLM call failed"
                    if self.logger:
                        self.logger.log_error(
                            agent=self.name,
                            error_type="LLMError",
                            error_message=error_msg
                        )
                    return {
                        "success": False,
                        "error": error_msg,
                        "iterations": iterations
                    }
                
                messages.append(response)
                
                # Check if supervisor wants to use tools
                tool_calls = response.get("tool_calls", [])
                
                if not tool_calls:
                    # Supervisor is done
                    result = response.get("content", "")
                    
                    if self.logger:
                        self.logger.log_event(
                            event_type="supervisor_completed",
                            agent=self.name,
                            data={
                                "result": result,
                                "iterations": iterations,
                                "tools_created": len(self.tools),
                                "agents_created": len(self.agents)
                            }
                        )
                    
                    return {
                        "success": True,
                        "result": result,
                        "iterations": iterations,
                        "tools_created": list(self.tools.keys()),
                        "agents_created": list(self.agents.keys())
                    }
                
                # Execute supervisor's tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_supervisor_tool(tool_call)
                    tool_results.append(result)
                
                # Add results to conversation
                for i, tool_call in enumerate(tool_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_results[i].get("result", tool_results[i].get("error", "")))
                    })
            
            # Max iterations reached
            error_msg = f"Supervisor max iterations ({max_iterations}) reached"
            
            if self.logger:
                self.logger.log_error(
                    agent=self.name,
                    error_type="MaxIterationsError",
                    error_message=error_msg
                )
            
            return {
                "success": False,
                "error": error_msg,
                "iterations": iterations
            }
        
        except Exception as e:
            if self.logger:
                self.logger.log_error(
                    agent=self.name,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
            
            return {
                "success": False,
                "error": str(e),
                "iterations": iterations
            }
    
    def _build_system_prompt(self) -> str:
        """Build supervisor's system prompt."""
        return """You are a supervisor agent that creates tools and specialized agents to solve tasks.

Your capabilities:
1. create_tool - Create a Python function tool
2. create_agent - Create a specialized agent with tools
3. delegate_to_agent - Delegate a task to an agent

Workflow:
1. Analyze the user's task
2. Create any needed tools (parse_log, filter_errors, etc.)
3. Create a specialized agent with those tools
4. Delegate the task to the agent
5. Return the agent's result to the user

Available tools:
- create_tool(name, description, code, parameters)
- create_agent(name, description, tool_names)
- delegate_to_agent(agent_name, task)

Example:
User: "Analyze error.log"
You: 
1. create_tool("parse_log", ...) 
2. create_agent("log_analyzer", tools=["parse_log"])
3. delegate_to_agent("log_analyzer", "Analyze error.log")
4. Return agent's result
"""
    
    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Call LLM with retry logic."""
        # Get supervisor tools
        supervisor_tools = self._get_supervisor_tool_schemas()
        
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                response = self.llm_client.complete(
                    model=self.config.llm.model,
                    messages=messages,
                    tools=supervisor_tools,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                if self.logger:
                    self.logger.log_llm_call(
                        agent=self.name,
                        model=self.config.llm.model,
                        messages=messages,
                        response=response,
                        duration_ms=duration_ms,
                        include_messages=self.config.logging.include_messages
                    )
                
                return response
            
            except Exception as e:
                if self.logger:
                    self.logger.log_error(
                        agent=self.name,
                        error_type=type(e).__name__,
                        error_message=f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def _get_supervisor_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get tool schemas for supervisor's meta-tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_tool",
                    "description": "Create a new Python function tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Tool name (valid Python identifier)"
                            },
                            "description": {
                                "type": "string",
                                "description": "Tool description"
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
            {
                "type": "function",
                "function": {
                    "name": "create_agent",
                    "description": "Create a specialized agent with tools",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Agent name"
                            },
                            "description": {
                                "type": "string",
                                "description": "Agent's role/purpose"
                            },
                            "tool_names": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Names of tools to give agent"
                            }
                        },
                        "required": ["name", "description", "tool_names"]
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
        description: str,
        tool_names: List[str]
    ) -> str:
        """
        Create a new agent.
        
        Args:
            name: Agent name
            description: Agent purpose/role
            tool_names: Tools to give agent
        
        Returns:
            Success message
        """
        # Get tools
        tools = []
        for tool_name in tool_names:
            if tool_name not in self.tools:
                raise ValueError(f"Tool '{tool_name}' does not exist")
            tools.append(self.tools[tool_name])
        
        # Build system prompt
        system_prompt = f"{description}\n\nYou have access to the following tools: {', '.join(tool_names)}"
        
        # Create agent
        agent = Agent(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            llm_client=self.llm_client,
            config=self.config,
            logger=self.logger
        )
        
        # Save to workspace
        self.workspace.save_agent(name, system_prompt, tool_names)
        
        # Add to agents
        self.agents[name] = agent
        
        # Log creation
        if self.logger:
            self.logger.log_agent_created(
                creator_agent=self.name,
                agent_name=name,
                system_prompt=system_prompt,
                tools=tool_names
            )
        
        return f"Agent '{name}' created successfully with tools: {', '.join(tool_names)}"
    
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
        
        agent = self.agents[agent_name]
        
        # Log delegation
        if self.logger:
            self.logger.log_agent_delegated(
                from_agent=self.name,
                to_agent=agent_name,
                task=task
            )
        
        # Run agent
        result = agent.run(task)
        
        if not result["success"]:
            raise RuntimeError(f"Agent '{agent_name}' failed: {result.get('error', 'Unknown error')}")
        
        return result["result"]
