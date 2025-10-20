"""
Conversation logging system for iExplain.

Logs all agent interactions to JSONL files for debugging and analysis.
Supports:
- Real-time logging (survives crashes)
- Rich event types (LLM calls, tool calls, agent delegation)
- Hierarchical tracking (parent/child events)
- Metadata capture (timestamps, tokens, durations)
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


class LoggingError(Exception):
    """Logging-related errors."""
    pass


@dataclass
class LogEvent:
    """A single logged event."""
    run_id: str
    event_id: str
    timestamp: str
    event_type: str
    agent: str
    data: Dict[str, Any]
    parent_event_id: Optional[str] = None
    depth: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ConversationLogger:
    """
    Logger for agent conversations and tool executions.
    
    Writes events to JSONL file as they occur for crash-safety.
    """
    
    def __init__(
        self,
        log_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None
    ):
        """
        Initialize conversation logger.
        
        Args:
            log_dir: Directory to store log files
            config: Configuration dict to log
            user_query: Initial user query
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique run ID
        self.run_id = str(uuid.uuid4())
        
        # Create log file with timestamp + run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{timestamp}_{self.run_id[:8]}.jsonl"
        
        # Open file handle (keep open for real-time writing)
        try:
            self.file_handle = open(self.log_file, 'a', buffering=1)  # Line buffering
        except Exception as e:
            raise LoggingError(f"Failed to open log file {self.log_file}: {e}") from e
        
        # Track parent events for hierarchical logging
        self.event_stack: List[str] = []
        
        # Log run start
        self._log_run_start(config, user_query)
    
    def _log_run_start(
        self,
        config: Optional[Dict[str, Any]],
        user_query: Optional[str]
    ) -> None:
        """Log the start of a run."""
        self.log_event(
            event_type="run_start",
            agent="system",
            data={
                "config": config or {},
                "user_query": user_query,
                "timestamp_start": datetime.now().isoformat()
            }
        )
    
    def log_event(
        self,
        event_type: str,
        agent: str,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None
    ) -> str:
        """
        Log an event.
        
        Args:
            event_type: Type of event (llm_call, tool_call, etc.)
            agent: Name of agent that triggered event
            data: Event-specific data
            parent_event_id: Optional parent event ID for nesting
        
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        
        # Determine parent and depth
        if parent_event_id is None and self.event_stack:
            parent_event_id = self.event_stack[-1]
        
        depth = len(self.event_stack)
        
        event = LogEvent(
            run_id=self.run_id,
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            agent=agent,
            data=data,
            parent_event_id=parent_event_id,
            depth=depth
        )
        
        # Write to file immediately
        try:
            self.file_handle.write(json.dumps(event.to_dict()) + '\n')
            self.file_handle.flush()  # Ensure written to disk
        except Exception as e:
            # Don't crash on logging errors, just print
            print(f"Warning: Failed to log event: {e}")
        
        return event_id
    
    def push_context(self, event_id: str) -> None:
        """
        Push an event onto the stack (makes it parent of subsequent events).
        
        Use for hierarchical tracking (e.g., agent delegation).
        """
        self.event_stack.append(event_id)
    
    def pop_context(self) -> None:
        """Pop an event from the stack."""
        if self.event_stack:
            self.event_stack.pop()
    
    # ========================================================================
    # Convenience Methods for Common Events
    # ========================================================================
    
    def log_llm_call(
        self,
        agent: str,
        model: str,
        messages: List[Dict[str, Any]],
        response: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        include_messages: bool = True
    ) -> str:
        """
        Log an LLM API call.
        
        Args:
            agent: Agent making the call
            model: Model name
            messages: Input messages
            response: LLM response (if available)
            duration_ms: Call duration in milliseconds
            include_messages: Whether to include full messages (can be verbose)
        
        Returns:
            Event ID
        """
        data: Dict[str, Any] = {
            "model": model,
            "duration_ms": duration_ms
        }
        
        if include_messages:
            data["messages"] = messages
            data["response"] = response
        else:
            # Just log metadata
            data["message_count"] = len(messages)
            if response:
                data["response_summary"] = {
                    "role": response.get("role"),
                    "has_content": "content" in response,
                    "has_tool_calls": "tool_calls" in response
                }
        
        # Extract token usage if available
        if response and "usage" in response:
            data["tokens"] = response["usage"]
        
        return self.log_event(
            event_type="llm_call",
            agent=agent,
            data=data
        )
    
    def log_tool_call(
        self,
        agent: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Any] = None,
        success: bool = True,
        duration_ms: Optional[float] = None,
        include_result: bool = True
    ) -> str:
        """
        Log a tool execution.
        
        Args:
            agent: Agent using the tool
            tool_name: Name of tool
            arguments: Tool arguments
            result: Tool result
            success: Whether tool succeeded
            duration_ms: Execution duration in milliseconds
            include_result: Whether to include full result
        
        Returns:
            Event ID
        """
        data: Dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": success,
            "duration_ms": duration_ms
        }
        
        if include_result:
            data["result"] = result
        else:
            # Just log result type/size
            data["result_type"] = type(result).__name__
            if isinstance(result, (list, dict, str)):
                data["result_size"] = len(result)
        
        return self.log_event(
            event_type="tool_call",
            agent=agent,
            data=data
        )
    
    def log_tool_created(
        self,
        agent: str,
        tool_name: str,
        code: str,
        schema: Dict[str, Any]
    ) -> str:
        """
        Log tool creation.
        
        Args:
            agent: Agent creating the tool
            tool_name: Name of new tool
            code: Tool code
            schema: Tool schema
        
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="tool_created",
            agent=agent,
            data={
                "tool_name": tool_name,
                "code": code,
                "schema": schema,
                "code_lines": len(code.split('\n'))
            }
        )
    
    def log_agent_created(
        self,
        creator_agent: str,
        agent_name: str,
        system_prompt: str,
        tools: List[str]
    ) -> str:
        """
        Log agent creation.
        
        Args:
            creator_agent: Agent creating the new agent
            agent_name: Name of new agent
            system_prompt: System prompt for new agent
            tools: List of tool names
        
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="agent_created",
            agent=creator_agent,
            data={
                "agent_name": agent_name,
                "system_prompt": system_prompt,
                "tools": tools,
                "tool_count": len(tools)
            }
        )
    
    def log_agent_delegated(
        self,
        from_agent: str,
        to_agent: str,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log task delegation to another agent.
        
        Args:
            from_agent: Agent delegating the task
            to_agent: Agent receiving the task
            task: Task description
            context: Optional context data
        
        Returns:
            Event ID
        """
        event_id = self.log_event(
            event_type="agent_delegated",
            agent=from_agent,
            data={
                "to_agent": to_agent,
                "task": task,
                "context": context or {}
            }
        )
        
        # Push this event as context for child agent's events
        self.push_context(event_id)
        
        return event_id
    
    def log_agent_completed(
        self,
        agent: str,
        result: Any,
        iterations: int,
        success: bool = True
    ) -> str:
        """
        Log agent completion.
        
        Args:
            agent: Agent that completed
            result: Agent result
            iterations: Number of iterations
            success: Whether agent succeeded
        
        Returns:
            Event ID
        """
        # Pop context if this was a delegated agent
        if self.event_stack:
            self.pop_context()
        
        return self.log_event(
            event_type="agent_completed",
            agent=agent,
            data={
                "result": result,
                "iterations": iterations,
                "success": success
            }
        )
    
    def log_error(
        self,
        agent: str,
        error_type: str,
        error_message: str,
        traceback: Optional[str] = None
    ) -> str:
        """
        Log an error.
        
        Args:
            agent: Agent where error occurred
            error_type: Error type/class name
            error_message: Error message
            traceback: Optional full traceback
        
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="error",
            agent=agent,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "traceback": traceback
            }
        )
    
    def log_run_end(
        self,
        outcome: str,
        total_tokens: Optional[int] = None,
        total_duration_ms: Optional[float] = None
    ) -> str:
        """
        Log the end of a run.
        
        Args:
            outcome: Run outcome (success, error, interrupted)
            total_tokens: Total tokens used
            total_duration_ms: Total duration
        
        Returns:
            Event ID
        """
        return self.log_event(
            event_type="run_end",
            agent="system",
            data={
                "outcome": outcome,
                "total_tokens": total_tokens,
                "total_duration_ms": total_duration_ms,
                "timestamp_end": datetime.now().isoformat()
            }
        )
    
    def close(self) -> None:
        """Close the log file."""
        if hasattr(self, 'file_handle') and not self.file_handle.closed:
            self.file_handle.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Log error if exception occurred
        if exc_type is not None:
            self.log_error(
                agent="system",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                traceback=None  # Could add traceback.format_exc() here
            )
            self.log_run_end(outcome="error")
        else:
            self.log_run_end(outcome="success")
        
        self.close()
        return False  # Don't suppress exceptions
    
    def get_log_path(self) -> Path:
        """Get the path to the log file."""
        return self.log_file
