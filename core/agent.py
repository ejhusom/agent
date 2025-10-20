"""
Agent with integrated Config and Logger support.

This is an updated Agent that uses the new Config and Logger systems
for robust, configurable, and observable agent behavior.
"""

import time
from typing import List, Dict, Any, Optional, Callable

from .config import Config
from .logger import ConversationLogger


class AgentError(Exception):
    """Agent-related errors."""
    pass


class Agent:
    """
    Autonomous agent that uses LLM and tools to complete tasks.
    
    Features:
    - Configurable max iterations and error handling
    - Full conversation logging
    - LLM retry logic
    - Tool error handling (continue or fail based on config)
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: List[Callable],
        llm_client: Any,
        config: Config,
        logger: Optional[ConversationLogger] = None
    ):
        """
        Initialize agent.
        
        Args:
            name: Agent name
            system_prompt: System instructions
            tools: List of available tools
            llm_client: LLM client for completions
            config: Configuration object
            logger: Optional conversation logger
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.llm_client = llm_client
        self.config = config
        self.logger = logger
        
        # Build tool schemas for LLM
        self.tool_schemas = self._build_tool_schemas()
    
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """Build tool schemas from functions."""
        schemas = []
        for tool in self.tools:
            if hasattr(tool, '__tool_schema__'):
                schemas.append(tool.__tool_schema__)
        return schemas
    
    def run(self, message: str) -> Dict[str, Any]:
        """
        Run agent on a message.
        
        Args:
            message: User message/task
        
        Returns:
            Dict with success, result, and metadata
        """
        # Log agent start
        if self.logger:
            self.logger.log_event(
                event_type="agent_start",
                agent=self.name,
                data={"message": message}
            )
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message}
        ]
        
        iterations = 0
        max_iterations = self.config.agent.max_iterations
        
        try:
            while iterations < max_iterations:
                iterations += 1
                
                # Call LLM with retry logic
                response = self._call_llm_with_retry(messages)
                
                if not response:
                    # LLM call failed after retries
                    error_msg = "LLM call failed after retries"
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
                
                # Add assistant response to conversation
                messages.append(response)
                
                # Check if agent wants to use tools
                tool_calls = response.get("tool_calls", [])
                
                if not tool_calls:
                    # Agent is done (no tool calls)
                    result = response.get("content", "")
                    
                    if self.logger:
                        self.logger.log_agent_completed(
                            agent=self.name,
                            result=result,
                            iterations=iterations,
                            success=True
                        )
                    
                    return {
                        "success": True,
                        "result": result,
                        "iterations": iterations
                    }
                
                # Execute tool calls
                tool_results = []
                for tool_call in tool_calls:
                    result = self._execute_tool_call(tool_call)
                    tool_results.append(result)
                    
                    # Check if tool failed and we should stop
                    if not result["success"] and self.config.agent.fail_on_tool_error:
                        if self.logger:
                            self.logger.log_agent_completed(
                                agent=self.name,
                                result=result,
                                iterations=iterations,
                                success=False
                            )
                        return {
                            "success": False,
                            "error": f"Tool {tool_call['function']['name']} failed: {result['error']}",
                            "iterations": iterations
                        }
                
                # Add tool results to conversation
                for i, tool_call in enumerate(tool_calls):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(tool_results[i].get("result", tool_results[i].get("error", "")))
                    })
            
            # Max iterations reached
            error_msg = f"Max iterations ({max_iterations}) reached"
            
            if self.logger:
                self.logger.log_error(
                    agent=self.name,
                    error_type="MaxIterationsError",
                    error_message=error_msg
                )
                self.logger.log_agent_completed(
                    agent=self.name,
                    result=None,
                    iterations=iterations,
                    success=False
                )
            
            return {
                "success": False,
                "error": error_msg,
                "iterations": iterations
            }
        
        except Exception as e:
            # Unexpected error
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
    
    def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM with exponential backoff retry.
        
        Args:
            messages: Conversation messages
            max_retries: Maximum number of retries
        
        Returns:
            LLM response or None if all retries failed
        """
        for attempt in range(max_retries):
            start_time = time.time()
            
            try:
                response = self.llm_client.complete(
                    model=self.config.llm.model,
                    messages=messages,
                    tools=self.tool_schemas if self.tool_schemas else None,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful LLM call
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
                # Log error
                if self.logger:
                    self.logger.log_error(
                        agent=self.name,
                        error_type=type(e).__name__,
                        error_message=f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                
                # Retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    return None
        
        return None
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call.
        
        Args:
            tool_call: Tool call from LLM
        
        Returns:
            Dict with success, result, or error
        """
        tool_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]
        
        start_time = time.time()
        
        try:
            # Find tool
            tool = None
            for t in self.tools:
                if hasattr(t, '__name__') and t.__name__ == tool_name:
                    tool = t
                    break
            
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")
            
            # Execute tool
            result = tool(**arguments)
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful tool call
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
            
            return {
                "success": True,
                "result": result
            }
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # Log failed tool call
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
                    error_message=f"Tool {tool_name} failed: {e}"
                )
            
            return {
                "success": False,
                "error": str(e)
            }
