"""
Minimal agent implementation.
Just wraps LLM with tool execution - no complex workflows.
"""

from typing import List, Dict, Any, Optional
import json

from .config import config

class Agent:
    """Simple agent: LLM + tools."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm_client: Any,
        provider: str = None,
        model: str = None,
        tools: Dict[str, Dict[str, Any]] = None,
        temperature: float = None,
        max_tokens: int = None,
        logging_enabled: bool = True,
    ):
        """
        Args:
            name: Agent identifier
            system_prompt: System instructions
            llm_client: LLMClient instance
            tools: Dict of {name: {"function": callable, "schema": dict}}
        """
        self.name = name
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.tools = tools or {}
        self.history = []
        self.provider = provider if provider is not None else config.get("provider", None)
        self.model = model if model is not None else config.get("model", None)
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.0)
        self.max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 8192)
    
    def run(
        self,
        message: str,
        context: Dict[str, Any] = None,
        max_iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Execute agent on a message.
        
        Returns:
            {
                "content": str,
                "tool_calls": List[Dict],
                "history": List[Dict]
            }
        """
        messages = []
        
        # Add context if provided
        if context:
            context_text = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_text}\n\nTask: {message}"
            })
        else:
            messages.append({"role": "user", "content": message})
        
        # Get tool schemas
        tool_schemas = self._get_tool_schemas() if self.tools else None
        
        all_tool_calls = []
        iteration = 0
        
        # Agentic loop
        while iteration < max_iterations:
            iteration += 1
            
            # Call LLM
            response = self.llm_client.complete(
                messages=messages,
                system=self.system_prompt,
                tools=tool_schemas,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            self.history.append({
                "iteration": iteration,
                "response": response
            })

            # Print response for debugging
            print(f"=== Iteration {iteration} of agent {self.name} ===")
            print("Response Content:")
            print(response["content"])
            if response["tool_calls"]:
                print("Tool Calls:")
                for tc in response["tool_calls"]:
                    print(json.dumps(tc, indent=2))
            print("-----------------------\n")

            
            # No tool calls? Done
            if not response["tool_calls"]:
                return {
                    "content": response["content"],
                    "tool_calls": all_tool_calls,
                    "history": self.history
                }
            
            # Execute tools
            messages.append({
                "role": "assistant",
                "content": response["content"],
                "tool_calls": self._format_tool_calls(response["tool_calls"])
            })
            
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                all_tool_calls.append(tool_call)
                
                # Execute tool
                try:
                    result = self._execute_tool(tool_name, tool_call["arguments"])
                    result_content = json.dumps(result) if isinstance(result, dict) else str(result)
                except Exception as e:
                    result_content = f"Error: {str(e)}"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_name,
                    "content": result_content
                })

                # Print tool execution result
                print(f"Tool '{tool_name}' executed. Result:")
                print(result_content)
                print("-----------------------\n")

        # Max iterations reached
        return {
            "content": "Max iterations reached",
            "tool_calls": all_tool_calls,
            "history": self.history
        }
    
    def _get_tool_schemas(self) -> List[Dict]:
        """Get tool schemas for LLM."""
        return [tool["schema"] for tool in self.tools.values()]
    
    def _execute_tool(self, name: str, arguments: str) -> Any:
        """Execute a tool."""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        return self.tools[name]["function"](**args)
    
    def _format_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """Format tool calls for LLM API."""
        return [{
            "id": tc.get("id"),
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": tc["arguments"] if isinstance(tc["arguments"], str) 
                            else json.dumps(tc["arguments"])
            }
        } for tc in tool_calls]
