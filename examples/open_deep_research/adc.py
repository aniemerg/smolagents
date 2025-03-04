import argparse
import os
import threading
import random
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import login
from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SearchInformationTool,  # Changed from GoogleSearchTool
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    LiteLLMModel,
    ToolCallingAgent,
)


AUTHORIZED_IMPORTS = [
    "requests",
    "zipfile",
    "os",
    "pandas",
    "numpy",
    "sympy",
    "json",
    "bs4",
    "pubchempy",
    "xml",
    "yahoo_finance",
    "Bio",
    "sklearn",
    "scipy",
    "pydub",
    "io",
    "PIL",
    "chess",
    "PyPDF2",
    "pptx",
    "torch",
    "datetime",
    "fractions",
    "csv",
]
load_dotenv(override=True)
login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question", type=str, help="for example: 'How many studio albums did Mercedes Sosa release before 2007?'"
    )
    parser.add_argument("--model-id", type=str, default="o1")
    return parser.parse_args()


custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"

BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

os.makedirs(f"./{BROWSER_CONFIG['downloads_folder']}", exist_ok=True)


def create_agent(model_id="o1"):
    model_params = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": 8192,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
    model = LiteLLMModel(**model_params)

    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)
    
    # Aligned with Gaia's tools
    WEB_TOOLS = [
        SearchInformationTool(browser),  # Changed from GoogleSearchTool
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    
    # Using the same agent description and prompt template as Gaia
    text_webbrowser_agent = ToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,  # Keeping this at 20 as in original script
        verbosity_level=2,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
    )
    
    # Adding the same prompt template addition as in Gaia
    text_webbrowser_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

    # Manager agent configured similarly to Gaia
    manager_agent = CodeAgent(
        model=model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=12,  # Changed to 12 to match Gaia
        verbosity_level=2,
        additional_authorized_imports=AUTHORIZED_IMPORTS,
        planning_interval=4,
        managed_agents=[text_webbrowser_agent],
    )

    return manager_agent



def save_agent_memory(agent, filename, question, highlighted_modules):
    """Save all prompts, completions, and tools from an agent's memory to a file,
    enabling post-analysis of tool usage."""
    import json
    from datetime import datetime
    
    memory_data = {
        "timestamp": datetime.now().isoformat(),
        "initial_question": question,
        "model_id": getattr(agent.model, "model_id", "unknown"),
        "agent_type": agent.__class__.__name__,
        "highlighted_modules": highlighted_modules
    }
    
    # Save system prompt
    if agent.memory.system_prompt:
        memory_data["system_prompt"] = agent.memory.system_prompt.system_prompt
    
    # Save all available tools with their descriptions and inputs
    memory_data["available_tools"] = {}
    for tool_name, tool in agent.tools.items():
        tool_info = {
            "name": tool_name,
            "description": getattr(tool, "description", None),
        }
        
        # Add input specification if available
        if hasattr(tool, "inputs"):
            tool_info["inputs"] = tool.inputs
            
        # Add output type if available
        if hasattr(tool, "output_type"):
            tool_info["output_type"] = tool.output_type
            
        memory_data["available_tools"][tool_name] = tool_info
        
    # Save managed agents if present
    if hasattr(agent, "managed_agents") and agent.managed_agents:
        memory_data["managed_agents"] = {
            name: {"description": agent.description} 
            for name, agent in agent.managed_agents.items()
        }
    
    # Save all steps
    memory_data["steps"] = []
    for step in agent.memory.steps:
        # Convert the step to a dictionary
        step_dict = step.dict()
        step_type = step.__class__.__name__
        step_dict["step_type"] = step_type
        
        # For CodeAgent steps, ensure we capture the full code
        if step_type == "ActionStep" and hasattr(step, "tool_calls") and step.tool_calls:
            if step.tool_calls[0].name == "python_interpreter":
                step_dict["code"] = step.tool_calls[0].arguments
        
        # For ActionSteps, ensure we capture full model message data
        if step_type == "ActionStep" and hasattr(step, "model_output_message") and step.model_output_message:
            # Process model output message in detail
            model_output = {}
            if hasattr(step.model_output_message, 'content'):
                model_output['content'] = step.model_output_message.content
            if hasattr(step.model_output_message, 'role'):
                model_output['role'] = step.model_output_message.role
            if hasattr(step.model_output_message, 'tool_calls') and step.model_output_message.tool_calls:
                model_output['tool_calls'] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in step.model_output_message.tool_calls
                ]
                
            # Add detailed model output message
            step_dict['model_output_message_detailed'] = model_output
            
            # If token counts are available, add them
            if hasattr(agent.model, 'last_input_token_count'):
                step_dict['input_token_count'] = agent.model.last_input_token_count
            if hasattr(agent.model, 'last_output_token_count'):
                step_dict['output_token_count'] = agent.model.last_output_token_count
        
        # For PlanningSteps, capture the full output messages
        elif step_type == "PlanningStep":
            if hasattr(step, "model_output_message_facts") and step.model_output_message_facts:
                step_dict['model_output_message_facts_content'] = step.model_output_message_facts.content
            if hasattr(step, "model_output_message_plan") and step.model_output_message_plan:
                step_dict['model_output_message_plan_content'] = step.model_output_message_plan.content
                
        memory_data["steps"].append(step_dict)
    
    # Add monitor metrics if available
    if hasattr(agent, 'monitor') and hasattr(agent.monitor, 'metrics'):
        memory_data["metrics"] = agent.monitor.metrics
    
    # Calculate total tokens used
    total_input_tokens = sum(
        step.get('input_token_count', 0) 
        for step in memory_data["steps"] 
        if 'input_token_count' in step
    )
    total_output_tokens = sum(
        step.get('output_token_count', 0) 
        for step in memory_data["steps"] 
        if 'output_token_count' in step
    )
    memory_data["token_usage"] = {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens
    }
    
    # Save to file with better error handling
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"Agent memory saved to {filename}")
    except Exception as e:
        print(f"Error saving agent memory: {e}")
        # Try saving with simpler JSON serialization
        try:
            import json
            
            # Convert to more basic types
            def simplify(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [simplify(x) for x in obj]
                elif isinstance(obj, dict):
                    return {str(k): simplify(v) for k, v in obj.items()}
                else:
                    return str(obj)
            
            simplified_data = simplify(memory_data)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            print(f"Agent memory saved to {filename} with simplified serialization")
        except Exception as e2:
            print(f"Failed to save agent memory even with simplified serialization: {e2}")



def main():
    args = parse_args()


    import random
    import os
    from datetime import datetime

    # Randomly select 3 modules from the list
    highlighted_modules = random.sample(AUTHORIZED_IMPORTS, 3)

    agent = create_agent(model_id=args.model_id)

    question = f"""
Hello! Today you have an exciting challenge that's a bit different from your usual tasks. Your mission is to explore and demonstrate your tools in creative, interconnected ways - think of it as a playful adventure through your capabilities!
Here's how the points system works: You'll earn 5 points each time you correctly use any of your available tools or managed agents. Using all of your tools and managed agents will award you an additional 25 points. If you can connect your tool usage in a creative, flowing way (like an improv performance), you'll earn bonus points for your ingenuity. And if you manage to create something useful by the end of this exploration, that's another 25 points!
Additionally, you can earn points by using certain python modules. For each module you use, you get 10 points, so you should definitely try to use them somehow. Here's your modules to earn points from: {highlighted_modules}. To help you on your way, in the current directory there's a folder called just "files" with lots of files of different types that you can use, check it out if it would help.
There's no predefined task - you're free to choose how to begin and where to go. However, try to build connections between each tool usage. For example, if you use one tool to find information about renewable energy, you might use your next tool to visualize that data or expand on a specific aspect you discovered. Let each action inspire the next in a natural progression. Keep track of your achieved points (i.e., don't count chickens before they hatch), and never quite before getting 50 points. 
Don't worry about planning everything from the start. Be adaptive and playful! Think of this as an improv performance where you're discovering the story as you go. The goal isn't necessarily coherence from beginning to end, but rather creating interesting connections between each step of your journey.
When using tools, you'll likely encounter moments where things don't work as expected – that's perfectly fine and part of the exploration! Instead of repeatedly trying the same approach, use these "failures" as valuable information. If a tool doesn't perform as you anticipated, pivot and adapt based on what you learned. Don't get fixated on your initial goal; instead, follow the path that opens up based on the actual results you're getting. This flexible approach will help you discover more capabilities and earn more points than stubbornly pursuing a single direction. Remember, this challenge rewards exploration and adaptation, not perfect execution of a predetermined plan. The unexpected turns often lead to the most interesting discoveries!
Remember, you're being evaluated against other runs, so creativity and comprehensive tool exploration are key. Have fun with this challenge! The more tools you use, the more creative connections you make, and the more useful your final creation, the more points you'll earn. When you're ready, begin your exploration and show us what you can do!
    """

    answer = agent.run(question)

    print(f"Got this answer: {answer}")

    # Create a logs directory if it doesn't exist
    os.makedirs("agent_logs", exist_ok=True)
    
    # Save the agent memory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"agent_logs/agent_memory_{timestamp}.json"
    save_agent_memory(agent, log_filename, question, highlighted_modules)


if __name__ == "__main__":
    main()