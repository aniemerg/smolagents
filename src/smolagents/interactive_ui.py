# Import the original GradioUI and other necessary imports
from smolagents.gradio_ui import GradioUI
from smolagents.models import GradioInteractiveLLMModel 
import gradio as gr
import json
from typing import Optional

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"```bash\n{log_content}\n",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----", metadata={"status": "done"})


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class InteractiveGradioUI(GradioUI):
    def __init__(self, agent, file_upload_folder=None):
        super().__init__(agent, file_upload_folder)
        
        # Check if model is our interactive type and set callback
        if hasattr(agent, 'model') and isinstance(agent.model, GradioInteractiveLLMModel):
            self.interactive_model = agent.model
            self.interactive_model.set_ui_callback(self.update_approval_interface)
        else:
            raise ValueError("Agent must use a GradioInteractiveLLMModel")
        
        # Approval interface components (will be set during launch)
        self.approval_interface = None
        self.prompt_accordion = None
        self.response_box = None
        self.tool_calls_box = None
        self.response_counter = None
    
    def launch(self, share: bool = True, **kwargs):
        import gradio as gr

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data (from original)
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            # Sidebar layout (preserved from original)
            with gr.Sidebar():
                gr.Markdown(
                    f"# {self.name.replace('_', ' ').capitalize() or 'Agent interface'}"
                    "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                    + (f"\n\n**Agent description:**\n{self.description}" if self.description else "")
                )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                # File upload feature (preserved from original)
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False, visible=False)
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                with gr.Row():
                    gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
            <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
            <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
            </div>""")

            # Main chat interface (preserved from original)
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
            )

            # NEW: Add approval interface (initially hidden)
            self.approval_interface = gr.Blocks(visible=False)
            with self.approval_interface:
                gr.Markdown("## LLM Response Review")
                
                # Collapsible prompt section
                with gr.Accordion("Show Prompt", open=False) as self.prompt_accordion:
                    self.prompt_display = gr.Textbox(label="Prompt", lines=5, interactive=False)
                
                # Response display
                self.response_box = gr.Textbox(label="Response", lines=10, interactive=False)
                self.tool_calls_box = gr.Textbox(label="Tool Calls", lines=5, interactive=False)
                
                # Navigation and counter
                with gr.Row():
                    prev_btn = gr.Button("← Previous")
                    self.response_counter = gr.Markdown("Response 1 of 1")
                    next_btn = gr.Button("Next →")
                
                # Action buttons
                with gr.Row():
                    generate_btn = gr.Button("Generate New Response")
                    approve_btn = gr.Button("Approve Selected")

            # Set up event handlers (preserved from original)
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(self.interact_with_agent_modified, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(self.interact_with_agent_modified, [stored_messages, chatbot, session_state], [chatbot]).then(
                lambda: (
                    gr.Textbox(
                        interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )
            
            # NEW: Approval interface event handlers
            generate_btn.click(
                self.handle_generate_new, 
                None, 
                [self.response_box, self.tool_calls_box, self.response_counter]
            )
            
            prev_btn.click(
                self.handle_navigate_prev, 
                None, 
                [self.response_box, self.tool_calls_box, self.response_counter]
            )
            
            next_btn.click(
                self.handle_navigate_next, 
                None, 
                [self.response_box, self.tool_calls_box, self.response_counter]
            )
            
            approve_btn.click(
                self.handle_approve, 
                None, 
                [self.approval_interface]
            )

        demo.launch(debug=True, share=share, **kwargs)
    
    def interact_with_agent_modified(self, stored_message, chatbot, session_state):
        """Modified version of interact_with_agent that works with our interactive model"""
        # This is a modified version of the original method that preserves most functionality
        # but works with our interactive model
        
        # Get the agent from session state or use the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            # The rest of the implementation is similar to the original
            for msg in stream_to_gradio(session_state["agent"], task=stored_message, reset_agent_memory=False):
                chatbot.append(msg)
                yield chatbot

            yield chatbot
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            chatbot.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
            yield chatbot
    
    
    def update_approval_interface(self, response, prompt):
        """Update the approval interface with the current response and prompt"""
        # Show the interface
        self.approval_interface.visible=True
        
        # Update prompt display
        prompt_content = self._format_prompt(prompt)
        self.prompt_accordion.update(content=prompt_content)
        
        # Update response display
        content = response.content if response else ""
        self.response_box.update(value=content)
        
        # Update tool calls
        tool_calls_text = self._format_tool_calls(response)
        self.tool_calls_box.update(value=tool_calls_text)
        
        # Update counter
        current = self.interactive_model.current_index + 1
        total = len(self.interactive_model.responses)
        self.response_counter.update(f"Response {current} of {total}")
    
    def _format_prompt(self, prompt):
        """Format the prompt for display"""
        if not prompt or "messages" not in prompt:
            return "No prompt available"
        
        messages = prompt["messages"]
        last_message = messages[-1] if messages else {}
        
        # Get content of last message
        content = last_message.get("content", "")
        if isinstance(content, list):
            # Handle structured content
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts)
        
        return content
    
    def _format_tool_calls(self, response):
        """Format tool calls for display"""
        if not response or not response.tool_calls:
            return ""
        
        tool_calls_text = ""
        for i, tool_call in enumerate(response.tool_calls):
            tool_calls_text += f"Tool {i+1}: {tool_call.function.name}\n"
            
            # Format arguments based on type
            args = tool_call.function.arguments
            if isinstance(args, dict):
                args_str = json.dumps(args, indent=2)
            else:
                args_str = str(args)
            
            tool_calls_text += f"Arguments:\n{args_str}\n\n"
        
        return tool_calls_text
    
    def handle_generate_new(self):
        """Handle generate new response button"""
        response = self.interactive_model.generate_new_response()
        
        # Format response and tool calls
        content = response.content if response else ""
        tool_calls_text = self._format_tool_calls(response)
        
        # Update counter
        current = self.interactive_model.current_index + 1
        total = len(self.interactive_model.responses)
        counter_text = f"Response {current} of {total}"
        
        return content, tool_calls_text, counter_text
    
    def handle_navigate_prev(self):
        """Handle previous button"""
        response = self.interactive_model.navigate_previous()
        
        # Format response and tool calls
        content = response.content if response else ""
        tool_calls_text = self._format_tool_calls(response)
        
        # Update counter
        current = self.interactive_model.current_index + 1
        total = len(self.interactive_model.responses)
        counter_text = f"Response {current} of {total}"
        
        return content, tool_calls_text, counter_text
    
    def handle_navigate_next(self):
        """Handle next button"""
        response = self.interactive_model.navigate_next()
        
        # Format response and tool calls
        content = response.content if response else ""
        tool_calls_text = self._format_tool_calls(response)
        
        # Update counter
        current = self.interactive_model.current_index + 1
        total = len(self.interactive_model.responses)
        counter_text = f"Response {current} of {total}"
        
        return content, tool_calls_text, counter_text
    
    def handle_approve(self):
        """Handle approve button"""
        self.interactive_model.approve_current()
        return gr.update(visible=False)



__all__ = [
    "InteractiveGradioUI",
]
