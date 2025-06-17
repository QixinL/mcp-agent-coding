from mcp_agent.agents.agent import Agent
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Optional

#Planning agent
class PlanStep(BaseModel):
    agent: str = Field(..., description="Which agent performs this step")
    instruction: str = Field(..., description="The instruction for the agent to perform this step")

class Plan(BaseModel):
    steps: list[PlanStep]

#Schema for all agent outputs
class Schema(BaseModel):
    status: str = Field(..., description="The status of the finding step, e.g., 'pass' or 'error'")
    result: str = Field(..., description="The direct result of the agent's task, e.g., the file path or the output of the code execution")
    summary: str = Field(..., description="A short summary of what was accomplished and the key results of this step")

class OrchestratorSchema(BaseModel):
    agent:  Optional[str] = Field(..., description="The name of the agent that should perform the next step")
    instruction: Optional[str] = Field(..., description="The instruction for the agent to perform the next step")
    finished: bool = Field(..., description="True if the orchestration is complete or there is an error, False if the orchestration should continue")


class Orchestrator():
    def __init__(self, augumented_llm, available_agents):
        # The augmented_llm should be an instance of OpenAIAugmentedLLM or GoogleAugmentedLLM
        self.augmented_llm = augumented_llm
        # The available_agents should be a dictionary or list of Agent instances
        self.available_agents = {agent.name: agent for agent in available_agents}
    
        self.parsers = {
            "planner": PydanticOutputParser(pydantic_object=Plan),
            "orchestrator": PydanticOutputParser(pydantic_object=PlanStep),
            "general": PydanticOutputParser(pydantic_object=Schema),
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

        prompt = f"""
        You are an expert planner. Break down the task into JSON steps. Provide sufficient context for each instruction.
        {format_instructions}

        Task: {user_query}
        Available agents: {[agent for agent in self.available_agents.values()]
        }
        """
        
        raw = await planning_llm.generate_str(prompt)

        plan: Plan = parser.parse(raw)
        return plan.model_dump()

    async def orchestrate_plan(self, user_query, plan):
        #create an orchestrator agent that will review the output of the agents and update the instruction if needed
        orchestrator_agent = Agent(
            name="orchestrator",
            instruction="""
            You are an orchestrator agent. Your job is to oversee the execution of the plan.
            Review the output of each agent and update the instructions for the next agent if needed.
            """,
        )

        result_json = None
        previous_agent = None
        previous_instruction = None

        while True:
            # have the orchestrator agent review the plan and update the instructions if needed
            orchestrator_llm = await self.llm_factory(orchestrator_agent)
            format_instructions = self.parsers["orchestrator"].get_format_instructions()

            prompt = f"""
            current task: {user_query}
            current plan: {plan}

            previous agent: {previous_agent}
            previous instruction: {previous_instruction}

            previous agent status: {result_json.get("status") if result_json else "N/A"}
            previous agent result: {result_json.get("result") if result_json else "N/A"}
            previous agent summary: {result_json.get("summary") if result_json else "N/A"}
            
            Return in a JSON object with the following fields:
            - finished: True if the orchestration is complete or there is an error, False if the orchestration should continue
            - agent: the name of the agent that should perform the next step (None if the orchestration is finished)
            - instruction: the instruction for the agent to perform the next step (None if the orchestration is finished)
            {format_instructions}

            Ensure that:
            - The instruction has all the necessary context to perform its task.
            - The instruction is as concise as possible.
            - Provide the exact paths obtained from the previous agent's output if it is relevant to the current agent.
            """
            
            raw = await orchestrator_llm.generate_str(prompt)
            print(f"\nOrchestrator response: {raw}")
            next_step = self.parsers["orchestrator"].parse(raw)
            next_step_json = next_step.model_dump()

            #Exit condition
            if next_step_json.get("finished"):
                print("Orchestration finished.")
                break

            agent_name = next_step_json.get("agent")
            instruction = next_step_json.get("instruction")

            agent_llm = await self.llm_factory(self.available_agents[agent_name])
            parser = self.parsers["general"]
            format_instructions = parser.get_format_instructions()

            prompt = f"""
            Produce one JSON object matching this schema:
            {format_instructions}
            Task: {instruction}
            """
            raw = await agent_llm.generate_str(prompt)
            print(f"\nRaw response from agent {agent_name}: {raw}")
            result = parser.parse(raw)
            result_json = result.model_dump()



    async def generate_str(self, user_query):
        print(f"User query: {user_query}")
        print(f"Available agents: {self.available_agents.keys()}")
        print("Starting orchestration...\n")
        #test run generate_plan
        plan = await self.generate_plan(user_query)
        print(f"Generated plan: {plan}")
        #test run orchestrate_plan
        await self.orchestrate_plan(user_query, plan)
        print("Orchestration complete.")