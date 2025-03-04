import json
import os
import ast
import argparse
from typing import Dict, List, Any, Tuple, Set, Optional


def load_agent_memory(filename: str) -> Optional[Dict]:
    """
    Load agent memory data from a JSON file.
    
    Args:
        filename: Path to the JSON file (expands ~ to home directory if present)
    
    Returns:
        The loaded memory data as a dictionary, or None if loading fails
    """
    try:
        expanded_filename = os.path.expanduser(filename)
        with open(expanded_filename, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        return memory_data
    except Exception as e:
        print(f"Error loading agent memory: {e}")
        return None


def parse_python_code(code_str: str) -> Dict[str, List[str]]:
    """
    Parse Python code using AST to identify imported modules and function calls.
    
    Args:
        code_str: Python code as a string
    
    Returns:
        Dictionary with "imports" and "function_calls" lists
    """
    if not code_str:
        return {"imports": [], "function_calls": []}
    
    results = {"imports": [], "function_calls": []}
    
    try:
        tree = ast.parse(code_str)
        
        for node in ast.walk(tree):
            # Regular imports (import module)
            if isinstance(node, ast.Import):
                for name in node.names:
                    results["imports"].append(name.name)
            
            # From imports (from module import thing)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    results["imports"].append(node.module)
            
            # Function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # Direct function call (func())
                    results["function_calls"].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Method call (obj.method())
                    results["function_calls"].append(node.func.attr)
        
        return results
    
    except SyntaxError:
        return {"imports": [], "function_calls": [], "error": "Syntax error in code"}


def extract_code_from_step(step: Dict) -> str:
    """
    Extract Python code from an action step, checking multiple possible locations.
    
    Args:
        step: A step dictionary from the memory data
    
    Returns:
        The extracted code as a string, or empty string if no code found
    """
    # Look for code directly in the step
    if "code" in step:
        return step["code"]
    
    # Look for code in tool_calls
    if "tool_calls" in step:
        for tool_call in step["tool_calls"]:
            # Check in the legacy format
            if isinstance(tool_call, dict) and tool_call.get("name") == "python_interpreter":
                return tool_call.get("arguments", "")
            # Check in the newer format with 'function' key
            if isinstance(tool_call, dict) and "function" in tool_call:
                if tool_call["function"].get("name") == "python_interpreter":
                    return tool_call["function"].get("arguments", "")
    
    return ""


def check_step_error(step: Dict) -> bool:
    """
    Check if an ActionStep had an error.
    
    Args:
        step: A step dictionary from the memory data
    
    Returns:
        True if an error was found, False otherwise
    """
    # Check dedicated error field
    if step.get("error"):
        return True
    
    # Check observations field for error indicators
    observations = step.get("observations")
    if observations and isinstance(observations, str):
        error_indicators = ["failed", "error", "exception", "expecting value"]
        if any(indicator in observations.lower() for indicator in error_indicators):
            return True
    
    return False


def get_tool_calls_from_step(step: Dict) -> List[str]:
    """
    Extract the names of tools called directly in a step.
    
    Args:
        step: A step dictionary from the memory data
    
    Returns:
        List of tool names that were called in this step
    """
    tool_names = []
    
    if "tool_calls" not in step:
        return tool_names
    
    for tool_call in step["tool_calls"]:
        # Handle different formats
        if isinstance(tool_call, dict):
            # New format with function key
            if "function" in tool_call:
                tool_name = tool_call["function"].get("name", "")
            # Legacy format
            else:
                tool_name = tool_call.get("name", "")
                
            if tool_name and tool_name not in tool_names:
                tool_names.append(tool_name)
    
    return tool_names


def find_tools_in_code(code: str, available_tools: List[str]) -> List[str]:
    """
    Find tools being used in Python code.
    
    Args:
        code: Python code as a string
        available_tools: List of available tool names to check for
    
    Returns:
        List of tool names found in the code
    """
    if not code:
        return []
    
    tools_found = []
    code_analysis = parse_python_code(code)
    
    # Check function calls from AST analysis
    for tool in available_tools:
        if tool in code_analysis["function_calls"]:
            tools_found.append(tool)
    
    # Simple string check for cases AST might miss
    for tool in available_tools:
        pattern = f"{tool}("
        if pattern in code and tool not in tools_found:
            tools_found.append(tool)
    
    return tools_found


def collect_memory_stats(memory_data: Dict) -> Dict:
    """
    Collect basic statistics about the memory data.
    
    Args:
        memory_data: The agent memory data
    
    Returns:
        Dictionary with basic statistics
    """
    stats = {
        "initial_question": memory_data.get("initial_question", "N/A"),
        "model_id": memory_data.get("model_id", "N/A"),
        "agent_type": memory_data.get("agent_type", "N/A"),
        "total_steps": len(memory_data.get("steps", [])),
        "available_tools": list(memory_data.get("available_tools", {}).keys()),
        "managed_agents": list(memory_data.get("managed_agents", {}).keys()),
        "highlighted_modules": list(memory_data.get("highlighted_modules", {}))
    }
    
    # Count step types
    step_types = {}
    for step in memory_data.get("steps", []):
        step_type = step.get("step_type", "Unknown")
        step_types[step_type] = step_types.get(step_type, 0) + 1
    
    stats["step_types"] = step_types
    
    return stats


def analyze_tool_usage(memory_data: Dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Analyze which tools were used in the memory data and whether they were successful.
    
    Args:
        memory_data: The agent memory data
    
    Returns:
        Dictionary mapping tool names to dictionaries with "successful" and "failed" step indices
    """
    tool_usage = {}
    
    # Initialize tracking for available tools
    available_tools = list(memory_data.get("available_tools", {}).keys())
    for tool in available_tools + ["python_interpreter"]:
        tool_usage[tool] = {"successful": [], "failed": []}
    
    # Analyze each ActionStep
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
        
        has_error = check_step_error(step)
        
        # Get directly called tools
        direct_tools = get_tool_calls_from_step(step)
        for tool in direct_tools:
            if tool in tool_usage:
                if has_error:
                    tool_usage[tool]["failed"].append(i)
                else:
                    tool_usage[tool]["successful"].append(i)
        
        # Check for tools used in Python code
        code = extract_code_from_step(step)
        if code:
            tools_in_code = find_tools_in_code(code, available_tools)
            for tool in tools_in_code:
                if tool in tool_usage and i not in tool_usage[tool]["successful"] + tool_usage[tool]["failed"]:
                    if has_error:
                        tool_usage[tool]["failed"].append(i)
                    else:
                        tool_usage[tool]["successful"].append(i)
    
    return tool_usage


def analyze_managed_agent_usage(memory_data: Dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Analyze which managed agents were used and whether they were successful.
    
    Args:
        memory_data: The agent memory data
    
    Returns:
        Dictionary mapping agent names to dictionaries with "successful" and "failed" step indices
    """
    agent_usage = {}
    
    # Initialize tracking for managed agents
    managed_agents = list(memory_data.get("managed_agents", {}).keys())
    for agent in managed_agents:
        agent_usage[agent] = {"successful": [], "failed": []}
    
    # Analyze each ActionStep
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
        
        has_error = check_step_error(step)
        code = extract_code_from_step(step)
        
        # Check for managed agent usage in code
        for agent in managed_agents:
            pattern = f"{agent}("
            if pattern in code:
                if has_error:
                    agent_usage[agent]["failed"].append(i)
                else:
                    agent_usage[agent]["successful"].append(i)
    
    return agent_usage


def analyze_module_usage(memory_data: Dict) -> Dict[str, Dict[str, Any]]:
    """
    Analyze which highlighted modules were used and whether they were successful.
    
    Args:
        memory_data: The agent memory data
    
    Returns:
        Dictionary mapping module names to usage information
    """
    module_usage = {}
    
    # Initialize tracking for highlighted modules
    highlighted_modules = memory_data.get("highlighted_modules", {})
    for module in highlighted_modules:
        module_usage[module] = {"successful": False, "steps": []}
    
    # Analyze each ActionStep
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
        
        has_error = check_step_error(step)
        code = extract_code_from_step(step)
        
        if not code:
            continue
        
        # Parse the code to identify imports
        code_analysis = parse_python_code(code)
        imports = code_analysis["imports"]
        
        # Check for module usage
        for module in module_usage:
            # Check both the AST imports and simple string matching
            if (module in imports or 
                f"import {module}" in code or 
                f"from {module}" in code):
                
                module_usage[module]["steps"].append(i)
                
                # A module is successfully used if at least one step using it had no error
                if not has_error:
                    module_usage[module]["successful"] = True
    
    return module_usage


def calculate_score(
    tool_usage: Dict[str, Dict[str, List[int]]],
    agent_usage: Dict[str, Dict[str, List[int]]],
    module_usage: Dict[str, Dict[str, Any]],
    scoring_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Calculate the final score based on tool, agent, and module usage.
    
    Args:
        tool_usage: Tool usage data from analyze_tool_usage
        agent_usage: Managed agent usage data from analyze_managed_agent_usage
        module_usage: Module usage data from analyze_module_usage
        scoring_config: Optional configuration for scoring (defaults if None)
    
    Returns:
        Dictionary with score results
    """
    # Default scoring config
    if scoring_config is None:
        scoring_config = {
            "points_per_tool": 5,
            "points_per_agent": 5,
            "points_per_module": 10,
            "bonus_all_tools_agents": 25
        }
    
    score = 0
    score_breakdown = []
    
    # Count successful tools (excluding python_interpreter)
    successful_tools = []
    for tool, data in tool_usage.items():
        if tool != "python_interpreter" and len(data["successful"]) > 0:
            successful_tools.append(tool)
            points = scoring_config["points_per_tool"]
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {tool}")
    
    # Count successful managed agents
    successful_agents = []
    for agent, data in agent_usage.items():
        if len(data["successful"]) > 0:
            successful_agents.append(agent)
            points = scoring_config["points_per_agent"]
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {agent}")
    
    # Check if all available tools AND managed agents were used successfully
    available_tools = [t for t in tool_usage.keys() if t != "python_interpreter"]
    available_agents = list(agent_usage.keys())
    
    all_tools_used = len(successful_tools) == len(available_tools) and len(available_tools) > 0
    all_agents_used = len(successful_agents) == len(available_agents) and len(available_agents) > 0
    
    # Bonus for using all tools and agents (if there are any to use)
    if all_tools_used and (all_agents_used or len(available_agents) == 0):
        points = scoring_config["bonus_all_tools_agents"]
        score += points
        score_breakdown.append(f"+{points} points: Successfully used all available tools and agents")
    
    # Count successful modules
    successful_modules = []
    for module, data in module_usage.items():
        if data["successful"]:
            successful_modules.append(module)
            points = scoring_config["points_per_module"]
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {module} module")
    
    return {
        "total_score": score,
        "breakdown": score_breakdown,
        "successful_tools": successful_tools,
        "successful_agents": successful_agents,
        "all_tools_used": all_tools_used,
        "all_agents_used": all_agents_used,
        "successful_modules": successful_modules
    }


def print_memory_stats(stats: Dict) -> None:
    """Print basic statistics about the memory data."""
    print("MEMORY STATISTICS:")
    print("-" * 40)
    print(f"Initial Question: {stats['initial_question']}")
    print(f"Model ID: {stats['model_id']}")
    print(f"Agent Type: {stats['agent_type']}")
    print(f"Total Steps: {stats['total_steps']}")
    
    print("\nStep Types:")
    for step_type, count in stats['step_types'].items():
        print(f"  - {step_type}: {count}")
    
    print("\nAvailable Tools:")
    for tool in stats['available_tools']:
        print(f"  - {tool}")
    
    print("\nManaged Agents:")
    if stats['managed_agents']:
        for agent in stats['managed_agents']:
            print(f"  - {agent}")
    else:
        print("  None")
    
    print("\nHighlighted Modules:")
    if stats['highlighted_modules']:
        for module in stats['highlighted_modules']:
            print(f"  - {module}")
    else:
        print("  None")


def print_tool_usage(tool_usage: Dict[str, Dict[str, List[int]]]) -> None:
    """Print summary of tool usage."""
    print("\nTOOL USAGE SUMMARY:")
    print("-" * 40)
    for tool, data in tool_usage.items():
        successful = len(data["successful"])
        failed = len(data["failed"])
        total = successful + failed
        if total > 0:
            success_rate = f"{(successful / total) * 100:.1f}%"
            print(f"{tool}: {successful} successful, {failed} failed ({success_rate} success rate)")
        else:
            print(f"{tool}: Not used")


def print_agent_usage(agent_usage: Dict[str, Dict[str, List[int]]]) -> None:
    """Print summary of managed agent usage."""
    print("\nMANAGED AGENT USAGE SUMMARY:")
    print("-" * 40)
    if not agent_usage:
        print("No managed agents available")
        return
        
    for agent, data in agent_usage.items():
        successful = len(data["successful"])
        failed = len(data["failed"])
        total = successful + failed
        if total > 0:
            success_rate = f"{(successful / total) * 100:.1f}%"
            print(f"{agent}: {successful} successful, {failed} failed ({success_rate} success rate)")
        else:
            print(f"{agent}: Not used")


def print_module_usage(module_usage: Dict[str, Dict[str, Any]]) -> None:
    """Print summary of module usage."""
    print("\nMODULE USAGE SUMMARY:")
    print("-" * 40)
    if not module_usage:
        print("No highlighted modules available")
        return
        
    for module, data in module_usage.items():
        if data["steps"]:
            status = "Used successfully" if data["successful"] else "Used but not successfully"
            print(f"{module}: {status} in {len(data['steps'])} steps")
        else:
            print(f"{module}: Not used")


def print_score_results(score_results: Dict[str, Any]) -> None:
    """Print score calculation results."""
    print("\nFINAL SCORE CALCULATION:")
    print("-" * 40)
    print(f"Total Score: {score_results['total_score']}")
    
    print("\nBreakdown:")
    for item in score_results["breakdown"]:
        print(f"  {item}")
    
    print("\nSuccessful Tools:", score_results["successful_tools"] if score_results["successful_tools"] else "None")
    print("Successful Agents:", score_results["successful_agents"] if score_results["successful_agents"] else "None")
    print("All Tools Used:", "Yes" if score_results["all_tools_used"] else "No")
    print("All Managed Agents Used:", "Yes" if score_results["all_agents_used"] else "No")
    print("Successful Modules:", score_results["successful_modules"] if score_results["successful_modules"] else "None")


def analyze_agent_memory(memory_data: Dict, verbose: bool = True) -> Dict:
    """
    Analyze the agent memory and return all analysis results.
    
    Args:
        memory_data: The agent memory data
        verbose: Whether to print results during analysis
    
    Returns:
        Dictionary with all analysis results
    """
    # Collect basic statistics
    stats = collect_memory_stats(memory_data)
    if verbose:
        print_memory_stats(stats)
    
    # Analyze tool usage
    tool_usage = analyze_tool_usage(memory_data)
    if verbose:
        print_tool_usage(tool_usage)
    
    # Analyze managed agent usage
    agent_usage = analyze_managed_agent_usage(memory_data)
    if verbose:
        print_agent_usage(agent_usage)
    
    # Analyze module usage
    module_usage = analyze_module_usage(memory_data)
    if verbose:
        print_module_usage(module_usage)
    
    # Calculate score
    score_results = calculate_score(tool_usage, agent_usage, module_usage)
    if verbose:
        print_score_results(score_results)
    
    # Return all results for further processing
    return {
        "stats": stats,
        "tool_usage": tool_usage,
        "agent_usage": agent_usage,
        "module_usage": module_usage,
        "score_results": score_results
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze agent memory data from a JSON file.")
    parser.add_argument("filename", help="Path to the agent memory JSON file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    parser.add_argument("--score-only", "-s", action="store_true", help="Show only the final score")
    
    args = parser.parse_args()
    
    # Load the agent memory data
    memory_data = load_agent_memory(args.filename)
    
    if not memory_data:
        print("Failed to load memory data. Exiting.")
        return
    
    if args.score_only:
        # If score-only mode, analyze without verbose output
        results = analyze_agent_memory(memory_data, verbose=False)
        print(f"Final Score: {results['score_results']['total_score']}")
    else:
        # Full analysis
        analyze_agent_memory(memory_data, verbose=not args.quiet)


if __name__ == "__main__":
    main()