from smolagents import CodeAgent, GradioUI,  LiteLLMModel
from smolagents.interactive_ui import InteractiveGradioUI
from smolagents.models import GradioInteractiveLLMModel  # Import from models if it's there

model_id = "gpt-4o-2024-08-06"
interactive_model = GradioInteractiveLLMModel(
    model_id,
    temperature=0.8
)

agent = CodeAgent(
    tools=[],
    model=interactive_model,
    max_steps=4,
    verbosity_level=1,
    name="example_agent",
    description="This is an example agent that has no tools and uses only code.",
)

# Launch with our custom UI
ui = InteractiveGradioUI(agent, file_upload_folder="./data")
ui.launch()