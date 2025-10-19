"""
iExplain v2: Self-Modifying Agentic System

Entry point for the supervisor-driven architecture.
"""

import os
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.llm_client import LLMClient
from core.supervisor import Supervisor
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry


def main():
    """Run the supervisor agent."""
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)
    
    print("=" * 70)
    print("iExplain v2: Self-Modifying Agentic System")
    print("=" * 70)
    print()
    
    # Initialize components
    print("Initializing system...")
    llm_client = LLMClient(provider="anthropic", api_key=api_key)
    tool_registry = ToolRegistry()
    agent_registry = AgentRegistry()
    
    supervisor = Supervisor(
        llm_client=llm_client,
        tool_registry=tool_registry,
        agent_registry=agent_registry,
        instructions_dir="instructions"
    )
    
    print(f"✓ LLM client initialized (Anthropic)")
    print(f"✓ Tool registry initialized ({len(tool_registry.list_tools())} tools)")
    print(f"✓ Agent registry initialized ({len(agent_registry.list_agents())} agents)")
    print(f"✓ Supervisor ready")
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
        print("Interactive mode. Type 'exit' to quit.")
        print("Type 'help' for available commands.")
        print()
        
        while True:
            try:
                user_input = input(">>> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print_help()
                    continue
                
                if user_input.lower() == 'tools':
                    print(f"Available tools: {tool_registry.list_tools()}")
                    continue
                
                if user_input.lower() == 'agents':
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
                print("\n\nInterrupted. Type 'exit' to quit.")
            
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()


def print_help():
    """Print help message."""
    print("""
Commands:
  <task>     Execute a task with the supervisor
  tools      List available tools
  agents     List created agents
  help       Show this help
  exit       Quit

Examples:
  >>> Create a tool to parse OpenStack logs
  >>> Analyze error.log for anomalies
  >>> Create an agent specialized in log analysis

The supervisor will create tools and agents as needed.
""")


if __name__ == "__main__":
    main()
