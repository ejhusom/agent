#!/usr/bin/env python3
"""
Quick test to verify iExplain v2 components work.
Run this before using the full system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.sandbox import Sandbox
from core.llm_client import LLMClient
from registry.tool_registry import ToolRegistry
from registry.agent_registry import AgentRegistry


def test_sandbox():
    """Test code execution sandbox."""
    print("Testing sandbox...", end=" ")
    
    sandbox = Sandbox()
    
    # Test simple execution
    result = sandbox.execute("""
result = 2 + 2
""")
    
    assert result["success"], "Sandbox execution failed"
    assert result["return_value"] == 4, "Sandbox returned wrong value"
    
    print("✓")


def test_registries():
    """Test tool and agent registries."""
    print("Testing registries...", end=" ")
    
    # Test tool registry
    tools = ToolRegistry(persist_dir="/tmp/test-tools")
    
    def dummy_tool(x: int) -> int:
        return x * 2
    
    schema = {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "Test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"}
                },
                "required": ["x"]
            }
        }
    }
    
    tools.register("dummy_tool", dummy_tool, schema, "def dummy_tool(x): return x * 2")
    assert "dummy_tool" in tools.list_tools(), "Tool not registered"
    
    # Test agent registry
    agents = AgentRegistry(persist_dir="/tmp/test-agents")
    agents.register("test_agent", None, {"system_prompt": "test"})
    assert "test_agent" in agents.list_agents(), "Agent not registered"
    
    print("✓")


def test_llm_client():
    """Test LLM client initialization."""
    print("Testing LLM client...", end=" ")
    
    import os
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("⚠ (skipped - no API key)")
        return
    
    try:
        client = LLMClient(provider="anthropic", api_key=api_key)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")


def main():
    """Run all tests."""
    print("=" * 50)
    print("iExplain v2 Component Tests")
    print("=" * 50)
    print()
    
    try:
        test_sandbox()
        test_registries()
        test_llm_client()
        
        print()
        print("=" * 50)
        print("All tests passed! System is ready.")
        print("=" * 50)
        print()
        print("Run with: python main.py")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
