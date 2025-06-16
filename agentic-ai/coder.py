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

async def example_usage():
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        # logger.info("Current config:", data=context.config.model_dump())

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
            instruction="""Your job is to determine if code is needed to solve the user's request.
            If code is needed, write python code to solve the user's request and return the code in the 
            code section of the JSON formatting that will be provided.
            Ensure that:
            - The code can be executed directly in the current working directory and does not have ```python headers and footers.
            - The code has no syntax or runtime errors.
            After, return the path to the python file relative to the current working directory.""",
            # server_names=[],
        )

        executor_agent = Agent(
            name="executor",
            instruction="""Determine if the code has been properly executed and return the output/error.""",
            server_names=["filesystem"],
        )

        task = """Obtain the number of lines in the file short_story.md"""

        orchestrator = Orchestrator(
            augumented_llm=GoogleAugmentedLLM,
            available_agents=[
                finder_agent,
                coder_agent,
                executor_agent,
            ],
        )

        result = await orchestrator.generate_str(user_query=task)
        # logger.info(f"{result}")

if __name__ == "__main__":
    start = time.time()
    asyncio.run(example_usage())
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")
