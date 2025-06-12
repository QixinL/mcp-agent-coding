from mcp_agent.agents.agent import Agent
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class PlanStep(BaseModel):
    agent: str = Field(..., description="Which agent performs this step")
    instruction: str = Field(..., description="The instruction for the agent to perform this step")

class Plan(BaseModel):
    steps: list[PlanStep]


class Orchestrator():
    def __init__(self, augumented_llm, available_agents):
        # The augmented_llm should be an instance of OpenAIAugmentedLLM or GoogleAugmentedLLM
        self.augmented_llm = augumented_llm
        # The available_agents should be a dictionary or list of Agent instances
        self.available_agents = {agent.name: agent for agent in available_agents}
    
        self.parsers = {
            "planner": PydanticOutputParser(pydantic_object=Plan),
            # "Other Agent Name": PydanticOutputParser(pydantic_object=OtherAgentOutput),
        }
    
    async def llm_factory(self, agent):
        return await agent.attach_llm(self.augmented_llm)


    #generate plan
    async def generate_plan(self, user_query):
        # This method would contain the logic to generate a plan using the available agents
        planning_agent=Agent(
            name="planner",
            instruction="""
            You are an expert planner. Given an objective task and a list of MCP servers (which are collections of tools)
            or Agents (which are collections of servers), your job is to break down the objective into a series of steps,
            which can be performed by LLMs with access to the servers or agents.
            """,
        )
        planning_llm = await self.llm_factory(planning_agent)
        parser = self.parsers["planner"]
        format_instructions = parser.get_format_instructions()

        # synthesizer_agent=Agent(
        #     name="synthesizer",
        #     instruction="You are an expert at synthesizing the results of a plan into a single coherent message.",
        # )
        # synthesizer_llm = await self.llm_factory(synthesizer_agent)


        prompt = f"""
        You are an expert planner.  Break down the task into JSON steps.
        {format_instructions}

        Task: {user_query}
        Available agents: {[agent for agent in self.available_agents.values()]
        }
        """
        
        raw = await planning_llm.generate_str(prompt)

        # 4) Let the parser validate & coerce into Plan()
        plan: Plan = parser.parse(raw)
        return plan



    async def generate_str(self, user_query):
        #test run generate_plan
        plan = await self.generate_plan(user_query)
        print(f"Generated plan: {plan.model_dump()}")
