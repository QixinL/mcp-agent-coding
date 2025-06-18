import asyncio
import os
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm import RequestParams
from orchestrator import Orchestrator
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="mcp_basic_agent")

#TODO put into utils
def create_python_file(file_content, file_name):
    """
    Create a Python file with the given file_content and file_name.
    The file will be created in the current working directory.
    """
    with open(file_name, "w") as f:
        f.write(file_content)
    print(f"Python file '{file_name}' created successfully.")
    return file_name

#TODO put into utils
def execute_python_file(file_name):
    """
    Execute a Python file with the given file_name.
    The file should be in the current working directory.
    """
    import subprocess
    try:
        result = subprocess.run(["python", file_name], capture_output=True, text=True)
        print(f"Executed Python file '{file_name}' successfully.")
        print(f"Result: {result.stdout}")
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error executing file: {result.stderr}"
    except Exception as e:
        print(f"Error executing Python file '{file_name}': {e}")
        return str(e)

#TODO put into utils
#def obtain header


async def example_usage():
    async with app.run() as agent_app:
        context = agent_app.context

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem.
            Your job is to search the filesystem for the file that match a user's request.
            Return the location of the file relative to the current working directory.
            Ensure that:
            - The returned file path is relative to the current working directory.""",
            server_names=["filesystem"],
        )

        coder_agent = Agent(
            name="coder",
            instruction=""" Output a JSON object with the following fields:
            status: "success" or "error"
            result: the code that solves the user's request (you need to generate the code)
            summary: a concise summary of the code and its purpose
            Ensure that:
            - The code can be executed directly in the current working directory.
            - The code has no syntax or runtime errors.
            """,
        )

        summary_agent = Agent(
            name="summary",
            instruction="""You are an agent that summarizes the result of the execution.
            Your job is to summarize the result of the execution and return it in a JSON format.
            Ensure that:
            - The summary is concise and contains only the necessary information.
            Ensure that the output follows the JSON format that will be provided.""",
        )

        task = """Obtain the number of lines in the file short_story.md"""

        orchestrator = Orchestrator(
            augumented_llm=GoogleAugmentedLLM,
            available_agents=[
                finder_agent,
                # summary_agent,
                coder_agent,
            ],
            available_functions=[
                create_python_file,
                execute_python_file,
            ],
        )

        result = await orchestrator.generate_str(user_query=task)

if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
