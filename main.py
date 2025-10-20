"""
iExplain: Self-Modifying Agentic System

Entry point for the supervisor-driven architecture.
"""
import os
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import config
from core.llm_client import LLMClient
from core.supervisor import Supervisor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry



def main():
    """Run the supervisor agent."""
    
    print("=" * 70)
    print("iExplain: Self-Modifying Agentic System")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing system...")
    llm_client = LLMClient(
        provider=config.get("provider", None),
        api_key=config.get("api_key", None),
        model=config.get("model", None)
    )
    tool_registry = ToolRegistry(persist_dir=config.get("tools_persist_dir"))
    agent_registry = AgentRegistry(persist_dir=config.get("agents_persist_dir"))
    
    supervisor = Supervisor(
        llm_client=llm_client,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        instructions_dir="instructions"
    )
    
    print(f"- [x] LLM client initialized (provider: {llm_client.provider}, model: {llm_client.model})")
    print(f"- [x] Tool registry initialized ({len(tool_registry.list_tools())} tools)")
    print(f"- [x] Agent registry initialized ({len(agent_registry.list_agents())} agents)")
    print(f"- [x] Supervisor ready")
    print()
    
    # Interactive mode
    if len(sys.argv) > 1:
        # Single task from command line
        task = " ".join(sys.argv[1:])
        print(f"Task: {task}")
        print("-" * 70)
        
        result = supervisor.run(task)
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        print(result["content"])
        print()
        
        if result["tool_calls"]:
            print(f"\nTools used: {len(result['tool_calls'])}")
            for tc in result["tool_calls"]:
                print(f"  - {tc['name']}")
    
    else:
        # Interactive REPL
        print("Interactive mode. Type '/exit' to quit.")
        print("Type '/help' for available commands.")
        print()
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/exit', '/quit', '/bye']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == '/help':
                    print_help()
                    continue
                
                if user_input.lower() == '/tools':
                    print(f"Available tools: {tool_registry.list_tools()}")
                    continue
                
                if user_input.lower() == '/agents':
                    print(f"Created agents: {agent_registry.list_agents()}")
                    continue
                
                # Execute task
                print()
                result = supervisor.run(user_input)
                
                print("\n" + "-" * 70)
                print(result["content"])
                print("-" * 70)
                print()
            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type '/exit' to quit.")
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def print_help():
    """Print help message."""
    print("""
Commands:
  <task>     Execute a task with the supervisor
  /tools      List available tools
  /agents     List created agents
  /help       Show this help
  /exit       Quit

Examples:
  >>> Create a tool to parse OpenStack logs
  >>> Analyze error.log for anomalies
  >>> Create an agent specialized in log analysis

The supervisor will create tools and agents as needed.
""")


if __name__ == "__main__":
    main()
