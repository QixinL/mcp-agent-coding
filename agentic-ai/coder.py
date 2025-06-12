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

        logger.info("Current config:", data=context.config.model_dump())

        # Add the current directory to the filesystem server's args
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        coder_agent = Agent(
            name="coder",
            instruction="""Your job is to write code that solves the user's request.""",
            server_names=["filesystem"],
        )

        executor_agent = Agent(
            name="executor",
            instruction=""""Your job is to execute code """,
            server_names=["fetch"],
        )

        task = """How many people upvoted the question in: https://stackoverflow.com/questions/76760600/how-to-fix-pydantics-deprecation-warning-about-using-model-dict-method
        Save the number to results.txt in the current working directory."""

        orchestrator = Orchestrator(
            augumented_llm=OpenAIAugmentedLLM,
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
