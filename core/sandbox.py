"""
Safe code execution in isolated subprocess.
"""

import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


class Sandbox:
    """Execute Python code safely in subprocess."""
    
    def __init__(self, timeout: int = 100, workspace: str = "/tmp/iexplain-workspace"):
        """
        Args:
            timeout: Execution timeout in seconds
            workspace: Directory for code execution
        """
        self.timeout = timeout
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)
    
    def execute(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute Python code in isolated process.
        
        Args:
            code: Python code to execute
            context: Optional variables to make available (as JSON)
        
        Returns:
            {
                "success": bool,
                "output": str,
                "error": str,
                "return_value": Any (if code uses `result = ...`)
            }
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            dir=self.workspace,
            delete=False
        ) as f:
            # Wrap code to capture output
            wrapped_code = self._wrap_code(code, context)
            f.write(wrapped_code)
            f.flush()
            
            try:
                result = subprocess.run(
                    ['python', f.name],
                    timeout=self.timeout,
                    capture_output=True,
                    text=True,
                    cwd=self.workspace
                )
                
                return {
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr,
                    "return_value": self._extract_result(result.stdout)
                }
            
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"Execution timeout ({self.timeout}s)",
                    "return_value": None
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "output": "",
                    "error": str(e),
                    "return_value": None
                }
            
            finally:
                # Cleanup
                Path(f.name).unlink(missing_ok=True)
    
    def _wrap_code(self, code: str, context: Dict[str, Any] = None) -> str:
        """Wrap code to capture result."""
        wrapper = []
        
        # Add context if provided
        if context:
            wrapper.append("import json")
            wrapper.append(f"_context = {json.dumps(context)}")
            wrapper.append("")
        
        # Add user code
        wrapper.append(code)
        wrapper.append("")
        
        # Capture result if assigned
        wrapper.append("""
# Try to capture result
if 'result' in locals():
    print('__RESULT__:', result)
""")
        
        return "\n".join(wrapper)
    
    def _extract_result(self, output: str) -> Any:
        """Extract result from output."""
        for line in output.split('\n'):
            if line.startswith('__RESULT__:'):
                result_str = line.split('__RESULT__:', 1)[1].strip()
                try:
                    return json.loads(result_str)
                except:
                    return result_str
        return None
    
    def test_tool(self, tool_code: str, test_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test a tool function with inputs.
        
        Args:
            tool_code: Function definition
            test_inputs: Arguments to pass
        
        Returns:
            Execution result
        """
        test_code = f"""
{tool_code}

# Extract function name (assume first 'def')
import re
func_match = re.search(r'def\\s+(\\w+)\\s*\\(', '''{tool_code}''')
func_name = func_match.group(1) if func_match else None

if func_name:
    import json
    inputs = {json.dumps(test_inputs)}
    result = globals()[func_name](**inputs)
"""
        
        return self.execute(test_code)
