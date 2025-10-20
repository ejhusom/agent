"""
Simplified LLM client using LiteLLM.
"""

from typing import List, Dict, Optional, Any
import litellm


class LLMClient:
    """Unified LLM interface."""
    
    def __init__(self, provider: str = "anthropic", api_key: Optional[str] = None):
        """
        Args:
            provider: 'anthropic', 'openai', or 'ollama'
            api_key: API key (not needed for Ollama)
        """
        self.provider = provider
        
        if provider == "anthropic" and api_key:
            litellm.api_key = api_key
        elif provider == "openai" and api_key:
            litellm.openai_key = api_key
    
    def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 16384,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Make completion request.
        
        Returns:
            {
                "content": str,
                "tool_calls": List[Dict],
                "usage": Dict
            }
        """
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Add system prompt
        if system:
            kwargs["messages"] = [{"role": "system", "content": system}] + messages
        
        # Add tools
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        try:
            response = litellm.completion(**kwargs)
            return self._parse_response(response)
        
        except Exception as e:
            raise Exception(f"LLM completion failed: {str(e)}")
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse LiteLLM response."""
        content = ""
        tool_calls = []
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            if hasattr(message, 'content') and message.content:
                content = message.content
            
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id if hasattr(tc, 'id') else None,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    })

            finish_reason = choice.finish_reason if hasattr(choice, 'finish_reason') else None
            role = message.role if hasattr(message, 'role') else None

        
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        if hasattr(response, 'usage'):
            usage_obj = response.usage
            usage = {
                "input_tokens": getattr(usage_obj, 'prompt_tokens', 0),
                "output_tokens": getattr(usage_obj, 'completion_tokens', 0),
                "total_tokens": getattr(usage_obj, 'total_tokens', 0)
            }

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": usage,
            "created": getattr(response, 'created', None),
            "model": getattr(response, 'model', None),
            "role": role,
            "finish_reason": finish_reason
        }
