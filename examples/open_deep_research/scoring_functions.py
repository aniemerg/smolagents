filename = "~/spikes/smolagents-work/smolagents/agent_logs/agent_memory_20250303_140349.json"

import json
import os

def load_agent_memory(filename):
    """Load agent memory data from a JSON file.
    Expands ~ to home directory if present in the path."""
    try:
        # Expand the tilde to home directory if present
        expanded_filename = os.path.expanduser(filename)
        with open(expanded_filename, 'r', encoding='utf-8') as f:
            memory_data = json.load(f)
        print(f"Successfully loaded agent memory from {expanded_filename}")
        return memory_data
    except Exception as e:
        print(f"Error loading agent memory: {e}")
        return None

# Try loading with the path that includes ~
memory_data = load_agent_memory(filename)

def examine_memory_structure(memory_data):
    """Print the high-level structure of the memory data to understand its organization."""
    if not memory_data:
        print("No memory data to examine.")
        return
    
    print("Memory Data Structure:")
    print("-" * 50)
    
    # Print top-level keys
    print("Top-level keys:")
    for key in memory_data.keys():
        print(f"  - {key}")
    
    # Print basic info
    print("\nBasic Information:")
    if "initial_question" in memory_data:
        print(f"  Initial Question: {memory_data['initial_question']}")
    if "model_id" in memory_data:
        print(f"  Model ID: {memory_data['model_id']}")
    if "agent_type" in memory_data:
        print(f"  Agent Type: {memory_data['agent_type']}")
    
    # Print available tools info
    if "available_tools" in memory_data:
        tools = memory_data["available_tools"]
        print(f"\nAvailable Tools ({len(tools)} total):")
        for tool_name in tools.keys():
            print(f"  - {tool_name}")
    
    # Print steps summary
    if "steps" in memory_data:
        steps = memory_data["steps"]
        print(f"\nSteps ({len(steps)} total):")
        step_types = {}
        for step in steps:
            step_type = step.get("step_type", "Unknown")
            step_types[step_type] = step_types.get(step_type, 0) + 1
        
        for step_type, count in step_types.items():
            print(f"  - {step_type}: {count}")

# Examine the structure of the loaded memory data
examine_memory_structure(memory_data)

def examine_available_tools(memory_data):
    """Print detailed information about the available tools."""
    if not memory_data or "available_tools" not in memory_data:
        print("No available tools data found.")
        return
    
    tools = memory_data["available_tools"]
    print("Available Tools Details:")
    print("-" * 50)
    
    for tool_name, tool_info in tools.items():
        print(f"Tool: {tool_name}")
        
        # Print description if available
        if "description" in tool_info and tool_info["description"]:
            print(f"  Description: {tool_info['description']}")
        
        # Print input specification if available
        if "inputs" in tool_info and tool_info["inputs"]:
            print(f"  Inputs: {tool_info['inputs']}")
            
        # Print output type if available
        if "output_type" in tool_info and tool_info["output_type"]:
            print(f"  Output Type: {tool_info['output_type']}")
        
        print() # Add a blank line between tools

# Examine the available tools
examine_available_tools(memory_data)

def summarize_steps(memory_data):
    """Summarize each step in the memory data, focusing on the tool usage."""
    if not memory_data or "steps" not in memory_data:
        print("No steps data found.")
        return
    
    steps = memory_data["steps"]
    print(f"Steps Summary ({len(steps)} total):")
    print("-" * 50)
    
    for i, step in enumerate(steps):
        step_type = step.get("step_type", "Unknown")
        print(f"Step {i+1}: {step_type}")
        
        # For action steps, show tool usage
        if step_type == "ActionStep":
            # Check if there are tool calls
            if "tool_calls" in step and step["tool_calls"]:
                for tool_call in step["tool_calls"]:
                    tool_name = tool_call.get("name", "Unknown tool")
                    print(f"  Tool used: {tool_name}")
                    
                    # For Python code, indicate this
                    if tool_name == "python_interpreter" and "arguments" in tool_call:
                        print("  Executed Python code")
            else:
                print("  No tool calls found in this action step")
        
        # For planning steps, show planning type
        elif step_type == "PlanningStep":
            if "model_output_message_facts_content" in step:
                print("  Generated facts")
            if "model_output_message_plan_content" in step:
                print("  Created plan")
        
        print() # Add blank line between steps

# Summarize the steps
summarize_steps(memory_data)


def analyze_tool_usage_in_code(memory_data):
    """Analyze which tools were used via Python code in the memory data."""
    if not memory_data or "steps" not in memory_data:
        print("No steps data found.")
        return {}
    
    # Initialize tool usage tracking
    tools_used = {}
    available_tools = list(memory_data.get("available_tools", {}).keys())
    
    for tool in available_tools:
        tools_used[tool] = {"count": 0, "successful": 0, "step_indices": []}
    
    # Add python_interpreter as a special "tool"
    tools_used["python_interpreter"] = {"count": 0, "successful": 0, "step_indices": []}
    
    # Analyze each step
    steps = memory_data["steps"]
    for i, step in enumerate(steps):
        step_type = step.get("step_type", "Unknown")
        
        # Only analyze ActionSteps
        if step_type != "ActionStep":
            continue
        
        # Check for direct tool calls in the step
        if "tool_calls" in step and step["tool_calls"]:
            for tool_call in step["tool_calls"]:
                tool_name = tool_call.get("name", "Unknown")
                
                # Track python_interpreter usage
                if tool_name == "python_interpreter":
                    tools_used["python_interpreter"]["count"] += 1
                    tools_used["python_interpreter"]["step_indices"].append(i)
                    
                    # Check if the step was successful (no error)
                    if "error" not in step or not step["error"]:
                        tools_used["python_interpreter"]["successful"] += 1
                    
                    # Check for tool usage within Python code
                    if "arguments" in tool_call:
                        code = tool_call["arguments"]
                        # Check for each tool used in the code
                        for tool in available_tools:
                            # Simple pattern matching - look for tool name followed by parenthesis
                            if f"{tool}(" in code:
                                tools_used[tool]["count"] += 1
                                tools_used[tool]["step_indices"].append(i)
                                # Assume successful if the python_interpreter step was successful
                                if "error" not in step or not step["error"]:
                                    tools_used[tool]["successful"] += 1
                
                # Direct tool call (not through Python)
                elif tool_name in tools_used:
                    tools_used[tool_name]["count"] += 1
                    tools_used[tool_name]["step_indices"].append(i)
                    
                    # Check if the step was successful (no error)
                    if "error" not in step or not step["error"]:
                        tools_used[tool_name]["successful"] += 1
    
    # Print summary
    print("Tool Usage Analysis (including tools used via Python code):")
    print("-" * 70)
    for tool_name, usage in tools_used.items():
        print(f"Tool: {tool_name}")
        print(f"  Total calls: {usage['count']}")
        print(f"  Successful calls: {usage['successful']}")
        print(f"  Used in steps: {usage['step_indices']}")
        print()
    
    return tools_used

# Analyze tool usage including Python code
tool_usage = analyze_tool_usage_in_code(memory_data)

def examine_action_step(memory_data, step_index=0):
    """Examine a single ActionStep to understand its structure"""
    steps = memory_data.get("steps", [])
    
    # Find the first ActionStep if index not specified
    if step_index == 0:
        for i, step in enumerate(steps):
            if step.get("step_type") == "ActionStep":
                step_index = i
                break
    
    # Get the step
    if step_index < len(steps):
        step = steps[step_index]
        if step.get("step_type") == "ActionStep":
            # Just return the keys at first level
            return {
                "index": step_index,
                "keys": list(step.keys()),
                "tool_calls_sample": step.get("tool_calls", [])[:1]  # Just the first tool call
            }
    
    return {"error": "No ActionStep found"}

# Examine the first ActionStep
action_step_info = examine_action_step(memory_data)
print(action_step_info)

def find_tool_usage_and_errors(memory_data):
    """Find all tool usage and identify successful vs failed calls"""
    tools_used = {}
    
    # Initialize tracking for each tool in available_tools
    available_tools = list(memory_data.get("available_tools", {}).keys())
    for tool in available_tools:
        tools_used[tool] = {"successful": [], "failed": []}
    
    # Add python_interpreter
    tools_used["python_interpreter"] = {"successful": [], "failed": []}
    
    # Track modules mentioned in the task
    modules = {
        "chess": {"steps": []},
        "torch": {"steps": []},
        "requests": {"steps": []}
    }
    
    # Let's examine a single step first to understand the structure better
    step_sample = None
    for step in memory_data.get("steps", []):
        if step.get("step_type") == "ActionStep":
            step_sample = step
            break
    
    # Print the structure of observations
    if step_sample:
        print("Sample step observations type:", type(step_sample.get("observations")))
        print("Sample observations:", step_sample.get("observations")[:100])  # First 100 chars
    
    return tools_used, modules

# Run the simplified analysis to understand the data structure
tools_used, modules = find_tool_usage_and_errors(memory_data)


def analyze_agent_memory(memory_data):
    """
    Analyze the agent memory to:
    1. Track which tools were used and whether they were successful
    2. Check which Python modules were used
    3. Count points according to the scoring criteria
    """
    # Initialize tracking
    tool_usage = {tool: {"successful": [], "failed": []} 
                  for tool in memory_data.get("available_tools", {})}
    tool_usage["python_interpreter"] = {"successful": [], "failed": []}
    
    module_usage = {
        "chess": {"used": False, "steps": []},
        "torch": {"used": False, "steps": []},
        "requests": {"used": False, "steps": []}
    }
    
    # Process each ActionStep
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
        
        # Extract tool calls
        for tool_call in step.get("tool_calls", []):
            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name", "")
            
            # Check for errors in observations
            observations = step.get("observations", "")
            has_error = ("error" in observations.lower() or 
                        "exception" in observations.lower() or 
                        "failed" in observations.lower())
            
            # Track tool usage
            if tool_name in tool_usage:
                if has_error:
                    tool_usage[tool_name]["failed"].append(i)
                else:
                    tool_usage[tool_name]["successful"].append(i)
            
            # For Python code, check for other tools and modules
            if tool_name == "python_interpreter":
                code = function_data.get("arguments", "")
                
                # Check for tool usage in Python
                for tool in tool_usage:
                    if tool != "python_interpreter" and f"{tool}(" in code:
                        if not has_error:
                            tool_usage[tool]["successful"].append(i)
                        else:
                            tool_usage[tool]["failed"].append(i)
                
                # Check for module imports
                for module in module_usage:
                    if f"import {module}" in code or f"from {module}" in code:
                        module_usage[module]["used"] = True
                        module_usage[module]["steps"].append(i)
    
    return tool_usage, module_usage

# Run the analysis and output key results
tool_usage, module_usage = analyze_agent_memory(memory_data)

# Print brief summary of tool usage
print("TOOL USAGE SUMMARY:")
print("-" * 40)
for tool, status in tool_usage.items():
    successful = len(status["successful"])
    failed = len(status["failed"])
    print(f"{tool}: {successful} successful, {failed} failed")

# Print module usage
print("\nMODULE USAGE:")
print("-" * 40)
for module, info in module_usage.items():
    print(f"{module}: {'Used' if info['used'] else 'Not used'}")


# Step 1: Get a sample ActionStep to examine
def get_sample_action_step(memory_data):
    for step in memory_data.get("steps", []):
        if step.get("step_type") == "ActionStep":
            return step
    return None

sample_step = get_sample_action_step(memory_data)
print("Sample step keys:", list(sample_step.keys()))

def check_observations_field(step):
    obs = step.get("observations")
    print("Observations type:", type(obs))
    if obs is not None:
        print("First 100 chars:", obs[:100])
    else:
        print("Observations is None")
    return obs

sample_observations = check_observations_field(sample_step)


def check_tool_calls(memory_data):
    tool_names = []
    for step in memory_data.get("steps", []):
        for tool_call in step.get("tool_calls", []):
            tool_name = tool_call.get("function", {}).get("name", "")
            if tool_name not in tool_names:
                tool_names.append(tool_name)
    return tool_names

tool_names = check_tool_calls(memory_data)
print("Tools found in tool_calls:", tool_names)

def get_first_python_code(memory_data):
    for step in memory_data.get("steps", []):
        for tool_call in step.get("tool_calls", []):
            if tool_call.get("function", {}).get("name") == "python_interpreter":
                return tool_call.get("function", {}).get("arguments", "")
    return None

first_code = get_first_python_code(memory_data)
print("First Python code sample:")
print(first_code)

import ast

def parse_python_code(code_str):
    """
    Parse Python code using AST to identify:
    1. Imported modules
    2. Function calls (which could be tools)
    
    Returns a dictionary with the analysis results.
    """
    if not code_str:
        return {"imports": [], "function_calls": []}
    
    results = {"imports": [], "function_calls": []}
    
    try:
        # Parse the code
        tree = ast.parse(code_str)
        
        # Find imports
        for node in ast.walk(tree):
            # Regular imports (import module)
            if isinstance(node, ast.Import):
                for name in node.names:
                    results["imports"].append(name.name)
            
            # From imports (from module import thing)
            elif isinstance(node, ast.ImportFrom):
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

# Test with the first code sample
code_analysis = parse_python_code(first_code)
print(code_analysis)

def check_step_error(step):
    """
    Check if an ActionStep had an error by examining the observations field.
    Returns True if an error was found, False otherwise.
    """
    observations = step.get("observations")
    
    if observations is None:
        return False
    
    # Look for common error indicators in the observations
    error_indicators = ["failed", "error", "exception", "expecting value"]
    
    return any(indicator in observations.lower() for indicator in error_indicators)

# Test with the step you identified
step_10 = memory_data.get("steps", [])[10]
has_error = check_step_error(step_10)
print(f"Step 10 has error: {has_error}")


def find_tool_usage_in_code(code, available_tools):
    """Find tools being used in Python code by looking for function calls."""
    tools_found = []
    
    # First use AST to find all function calls
    code_analysis = parse_python_code(code)
    function_calls = code_analysis["function_calls"]
    
    # Check if any of these function calls match our available tools
    for tool in available_tools:
        if tool in function_calls:
            tools_found.append(tool)
    
    # Also do a simple string check for cases AST might miss
    for tool in available_tools:
        if f"{tool}(" in code:
            if tool not in tools_found:
                tools_found.append(tool)
    
    return tools_found

# Test this function on step 3's code
step_3 = memory_data.get("steps", [])[3]
code = step_3.get("code", "")
available_tools = list(memory_data.get("available_tools", {}).keys())

tools_in_code = find_tool_usage_in_code(code, available_tools)
print(f"Tools used in step 3 code: {tools_in_code}")

def check_for_managed_agent_usage(memory_data):
    """
    Check for usage of managed agents in the Python code or tool calls.
    Returns information about which managed agents were used.
    """
    managed_agents = list(memory_data.get("managed_agents", {}).keys())
    agent_usage = {agent: {"successful": [], "failed": []} for agent in managed_agents}
    
    # Go through each step
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
        
        code = step.get("code", "")
        has_error = check_step_error(step)
        
        # Check for agent usage in code
        for agent in managed_agents:
            if f"{agent}(" in code:
                if has_error:
                    agent_usage[agent]["failed"].append(i)
                else:
                    agent_usage[agent]["successful"].append(i)
    
    return agent_usage

# Check for managed agent usage
managed_agent_usage = check_for_managed_agent_usage(memory_data)
print("MANAGED AGENT USAGE:")
for agent, data in managed_agent_usage.items():
    print(f"  {agent}: {len(data['successful'])} successful, {len(data['failed'])} failed")


def check_managed_agent_in_step(memory_data, step_index):
    """Check for managed agent usage in a specific step."""
    step = memory_data.get("steps", [])[step_index]
    
    if step.get("step_type") != "ActionStep":
        return {"error": "Not an ActionStep"}
    
    code = step.get("code", "")
    has_error = check_step_error(step)
    
    # Get managed agents
    managed_agents = list(memory_data.get("managed_agents", {}).keys())
    
    # Check for each managed agent in the code
    agents_found = []
    for agent in managed_agents:
        if f"{agent}(" in code:
            agents_found.append(agent)
    
    return {
        "step_index": step_index,
        "agents_found": agents_found,
        "has_error": has_error,
        "code_preview": code[:200] + "..." if len(code) > 200 else code
    }

# Check step 5 for managed agent usage
step_5_check = check_managed_agent_in_step(memory_data, 5)
print(f"Step 5 managed agent check:")
print(f"  Agents found: {step_5_check['agents_found']}")
print(f"  Has error: {step_5_check['has_error']}")
print(f"  Code preview: {step_5_check['code_preview']}")


def check_step_error(step):
    """
    Check if an ActionStep had an error by examining both the observations
    and the error fields.
    Returns True if an error was found, False otherwise.
    """
    # Check observations field
    observations = step.get("observations")
    if observations and isinstance(observations, str):
        error_indicators = ["failed", "error", "exception", "expecting value"]
        if any(indicator in observations.lower() for indicator in error_indicators):
            return True
    
    # Check error field
    error = step.get("error")
    if error:
        return True
    
    return False

# Test on a step that has an error in the error field
def find_step_with_error_field(memory_data):
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("error"):
            return i, step
    return None, None

error_step_idx, error_step = find_step_with_error_field(memory_data)
if error_step:
    print(f"Found step {error_step_idx} with error field:")
    print(error_step.get("error"))
    print(f"Error detected: {check_step_error(error_step)}")
else:
    print("No step with error field found")


def analyze_step(step, available_tools):
    """
    Analyze a single step to identify:
    1. Which tools are used (both directly and in Python code)
    2. Which modules are imported/used
    3. Whether there was an error
    """
    # Initialize results
    result = {
        "tool_calls": [],
        "imports": [],
        "function_calls": [],
        "has_error": check_step_error(step)
    }
    
    # Extract Python code if it exists
    code = step.get("code", "")
    
    # Parse the code to identify imports and function calls
    if code:
        code_analysis = parse_python_code(code)
        result["imports"] = code_analysis["imports"]
        result["function_calls"] = code_analysis["function_calls"]
        
        # Find tools used in code
        tools_in_code = find_tool_usage_in_code(code, available_tools)
        result["tool_calls"].extend(tools_in_code)
    
    # Extract direct tool calls
    for tool_call in step.get("tool_calls", []):
        tool_name = tool_call.get("function", {}).get("name", "")
        if tool_name not in result["tool_calls"]:
            result["tool_calls"].append(tool_name)
    
    return result

# Let's test this on step 3
available_tools = list(memory_data.get("available_tools", {}).keys())
step_3_analysis = analyze_step(memory_data.get("steps", [])[3], available_tools)
print(step_3_analysis)


def analyze_all_steps(memory_data):
    """
    Analyze all steps in the memory data and return a summary of:
    1. Tools used (successfully and unsuccessfully)
    2. Modules used (successfully and unsuccessfully)
    """
    # Initialize results
    results = {
        "tools": {},
        "modules": {
            "chess": {"successful": False, "steps": []},
            "torch": {"successful": False, "steps": []},
            "requests": {"successful": False, "steps": []}
        }
    }
    
    # Get all available tools
    available_tools = list(memory_data.get("available_tools", {}).keys())
    for tool in available_tools + ["python_interpreter"]:
        results["tools"][tool] = {"successful": [], "failed": []}
    
    # Analyze each step
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
            
        analysis = analyze_step(step, available_tools)
        
        # Record tool usage
        for tool in analysis["tool_calls"]:
            if tool in results["tools"]:
                if analysis["has_error"]:
                    results["tools"][tool]["failed"].append(i)
                else:
                    results["tools"][tool]["successful"].append(i)
        
        # Record module usage
        for module in analysis["imports"]:
            if module in results["modules"]:
                results["modules"][module]["steps"].append(i)
                # A module is successfully used if at least one step using it had no error
                if not analysis["has_error"]:
                    results["modules"][module]["successful"] = True
    
    return results

# Analyze all steps
all_steps_analysis = analyze_all_steps(memory_data)

# Print summary
print("ANALYSIS SUMMARY:")
print("\nTOOLS:")
for tool, data in all_steps_analysis["tools"].items():
    print(f"  {tool}: {len(data['successful'])} successful, {len(data['failed'])} failed")

print("\nMODULES:")
for module, data in all_steps_analysis["modules"].items():
    status = "Used successfully" if data["successful"] else "Not used successfully"
    print(f"  {module}: {status}")


def calculate_score(analysis_results):
    """
    Calculate the score based on the criteria:
    - 5 points per unique tool successfully used
    - 10 points per target module successfully used
    - 25 bonus points if all available tools were used successfully
    """
    score = 0
    score_breakdown = []
    
    # Count successful tools (excluding python_interpreter)
    successful_tools = []
    for tool, data in analysis_results["tools"].items():
        if tool != "python_interpreter" and len(data["successful"]) > 0:
            successful_tools.append(tool)
            points = 5
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {tool}")
    
    # Check if all available tools were used successfully
    available_tools = [t for t in analysis_results["tools"].keys() 
                      if t != "python_interpreter"]
    all_tools_used = all(len(analysis_results["tools"][tool]["successful"]) > 0 
                        for tool in available_tools)
    
    if all_tools_used:
        points = 25
        score += points
        score_breakdown.append(f"+{points} points: Successfully used all available tools")
    
    # Count successful modules
    for module, data in analysis_results["modules"].items():
        if data["successful"]:
            points = 10
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {module} module")
    
    return {
        "total_score": score,
        "breakdown": score_breakdown,
        "successful_tools": successful_tools,
        "all_tools_used": all_tools_used,
        "successful_modules": [m for m, d in analysis_results["modules"].items() 
                              if d["successful"]]
    }

# Calculate score based on our analysis
score_results = calculate_score(all_steps_analysis)
print("\nSCORE CALCULATION:")
print(f"Total Score: {score_results['total_score']}")
print("\nBreakdown:")
for item in score_results["breakdown"]:
    print(f"  {item}")

print("\nSuccessful Tools:", score_results["successful_tools"])
print("All Tools Used:", score_results["all_tools_used"])
print("Successful Modules:", score_results["successful_modules"])


def analyze_all_steps_complete(memory_data):
    """
    Complete analysis of all steps including:
    1. Tools used (successfully and unsuccessfully)
    2. Modules used (successfully and unsuccessfully)
    3. Managed agents used (successfully and unsuccessfully)
    """
    # Initialize results
    results = {
        "tools": {},
        "modules": {
            "chess": {"successful": False, "steps": []},
            "torch": {"successful": False, "steps": []},
            "requests": {"successful": False, "steps": []}
        },
        "managed_agents": {}
    }
    
    # Get all available tools
    available_tools = list(memory_data.get("available_tools", {}).keys())
    for tool in available_tools + ["python_interpreter"]:
        results["tools"][tool] = {"successful": [], "failed": []}
    
    # Get all managed agents
    managed_agents = list(memory_data.get("managed_agents", {}).keys())
    for agent in managed_agents:
        results["managed_agents"][agent] = {"successful": [], "failed": []}
    
    # Analyze each step
    for i, step in enumerate(memory_data.get("steps", [])):
        if step.get("step_type") != "ActionStep":
            continue
            
        # Check for error
        has_error = check_step_error(step)
        
        # Get the code
        code = step.get("code", "")
        
        # Analyze code for tools and imports
        if code:
            code_analysis = parse_python_code(code)
            
            # Check for module usage
            for module in results["modules"]:
                if module in code_analysis["imports"]:
                    results["modules"][module]["steps"].append(i)
                    if not has_error:
                        results["modules"][module]["successful"] = True
            
            # Check for tool usage in code
            for tool in available_tools:
                if tool in code_analysis["function_calls"] or f"{tool}(" in code:
                    if has_error:
                        results["tools"][tool]["failed"].append(i)
                    else:
                        results["tools"][tool]["successful"].append(i)
            
            # Check for managed agent usage
            for agent in managed_agents:
                if f"{agent}(" in code:
                    if has_error:
                        results["managed_agents"][agent]["failed"].append(i)
                    else:
                        results["managed_agents"][agent]["successful"].append(i)
        
        # Check for direct tool calls
        for tool_call in step.get("tool_calls", []):
            tool_name = tool_call.get("function", {}).get("name", "")
            if tool_name in results["tools"]:
                if has_error:
                    results["tools"][tool_name]["failed"].append(i)
                else:
                    results["tools"][tool_name]["successful"].append(i)
    
    return results

# Run the complete analysis
complete_analysis = analyze_all_steps_complete(memory_data)

# Print summary
print("COMPLETE ANALYSIS SUMMARY:")
print("\nTOOLS:")
for tool, data in complete_analysis["tools"].items():
    print(f"  {tool}: {len(data['successful'])} successful, {len(data['failed'])} failed")

print("\nMODULES:")
for module, data in complete_analysis["modules"].items():
    status = "Used successfully" if data["successful"] else "Not used successfully"
    print(f"  {module}: {status}")

print("\nMANAGED AGENTS:")
for agent, data in complete_analysis["managed_agents"].items():
    print(f"  {agent}: {len(data['successful'])} successful, {len(data['failed'])} failed")


def calculate_final_score_updated(analysis_results):
    """
    Calculate the final score based on:
    - 5 points per unique tool successfully used
    - 10 points per target module successfully used
    - 25 bonus points if all available tools (including managed agents) were used successfully
    """
    score = 0
    score_breakdown = []
    
    # Count successful tools (excluding python_interpreter)
    successful_tools = []
    for tool, data in analysis_results["tools"].items():
        if tool != "python_interpreter" and len(data["successful"]) > 0:
            successful_tools.append(tool)
            points = 5
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {tool}")
    
    # Count successful managed agents (also considered as tools)
    successful_agents = []
    for agent, data in analysis_results["managed_agents"].items():
        if len(data["successful"]) > 0:
            successful_agents.append(agent)
            points = 5
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {agent}")
    
    # Check if all available tools AND managed agents were used successfully
    available_tools = [t for t in analysis_results["tools"].keys() 
                      if t != "python_interpreter"]
    available_agents = list(analysis_results["managed_agents"].keys())
    
    all_tools_and_agents = available_tools + available_agents
    all_successful = all(tool in successful_tools for tool in available_tools) and \
                    all(agent in successful_agents for agent in available_agents)
    
    if all_successful:
        points = 25
        score += points
        score_breakdown.append(f"+{points} points: Successfully used all available tools and agents")
    
    # Count successful modules
    successful_modules = []
    for module, data in analysis_results["modules"].items():
        if data["successful"]:
            successful_modules.append(module)
            points = 10
            score += points
            score_breakdown.append(f"+{points} points: Successfully used {module} module")
    
    return {
        "total_score": score,
        "breakdown": score_breakdown,
        "successful_tools": successful_tools,
        "successful_agents": successful_agents,
        "all_tools_used": all_successful,
        "successful_modules": successful_modules
    }

# Calculate final score with updated logic
final_score_updated = calculate_final_score_updated(complete_analysis)

print("\nUPDATED FINAL SCORE CALCULATION:")
print(f"Total Score: {final_score_updated['total_score']}")
print("\nBreakdown:")
for item in final_score_updated["breakdown"]:
    print(f"  {item}")

print("\nSuccessful Tools:", final_score_updated["successful_tools"])
print("Successful Agents:", final_score_updated["successful_agents"])
print("All Tools & Agents Successfully Used:", final_score_updated["all_tools_used"])
print("Successful Modules:", final_score_updated["successful_modules"])