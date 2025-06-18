from mcp_agent.agents.agent import Agent
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Optional, List
import inspect

#Planning agent
class PlanStep(BaseModel):
    agent: str = Field(..., description="Which agent performs this step")
    instruction: str = Field(..., description="The instruction for the agent to perform this step")

class Plan(BaseModel):
    steps: list[PlanStep]

#Schema for all agent outputs
class Schema(BaseModel):
    result: str = Field(..., description="The direct result of the agent's task")
    summary: str = Field(..., description="A short summary of what was accomplished")


class OrchestratorSchema(BaseModel):
    finished: bool = Field(..., description="True if the orchestration is complete or there is an error, False if the orchestration should continue")
    agent:  Optional[str] = Field(None, description="The name of the agent that should perform the next step")
    instruction: Optional[str] = Field(None, description="The instruction for the agent to perform the next step")
    context: Optional[str] = Field(None, description="A summary of the most important information gathered from the last agent's output. Use less than 10 words")


class FunctionCall(BaseModel):
    function_name: str = Field(..., description="The name of the function to call")
    arguments: dict = Field(..., description="The arguments to pass to the function")



class Orchestrator():
    def __init__(self, augumented_llm, available_agents=[], available_functions=[]):
        # The augmented_llm should be an instance of OpenAIAugmentedLLM or GoogleAugmentedLLM
        self.augmented_llm = augumented_llm
        # The available_agents should be a dictionary or list of Agent instances
        
        self.available_functions = available_functions
        exact_params = [f"{func.__name__}: {inspect.signature(func)}" for func in self.available_functions]

        #create the function calling agent
        function_calling_agent = Agent(
            name="function_calling_agent",
            instruction=f"""You are a function calling agent. 
            Your job is to return a JSON object following your schema containing
            the exact function name and the arguments.
            Here is a list of the functions that can be called:
            {self.available_functions}
            Here are the exact parameters for each function:
            {exact_params}
            """,
        )

        available_agents.append(function_calling_agent)
        self.available_agents = {agent.name: agent for agent in available_agents}


        self.parsers = {
            "planner": PydanticOutputParser(pydantic_object=Plan),
            "orchestrator": PydanticOutputParser(pydantic_object=OrchestratorSchema),
            "general": PydanticOutputParser(pydantic_object=Schema),
            "function_call": PydanticOutputParser(pydantic_object=FunctionCall),
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
            Review the functions available to the function_calling_agent and use it when appropiate.
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
            Determine if the plan is complete. If not, review the output of each agent and update the instructions for the next agent if needed.

            """,
        )

        result_json = {}
        previous_agent = None
        previous_instruction = None

        # Plan loop
        max_steps = 20  # Limit the number of steps to prevent infinite loops

        for _ in range(max_steps):
            # have the orchestrator agent review the plan and update the instructions if needed
            orchestrator_llm = await self.llm_factory(orchestrator_agent)
            format_instructions = self.parsers["orchestrator"].get_format_instructions()


            prompt = f"""
            current task: {user_query}
            current plan: {plan}

            previous agent: {previous_agent}
            previous instruction: {previous_instruction}

            previous agent result: {result_json.get("result") if result_json else "N/A"}
            previous agent summary: {result_json.get("summary") if result_json else "N/A"}
            
            Return in a JSON object with the following fields:
            - finished: True if the orchestration is complete, False if the orchestration should continue
            - agent: the name of the agent that should perform the next step (review the plan when deciding this agent)
            here is the list of available agents: {self.available_agents.keys()} and the functions available to the function calling agent: {self.available_functions}
            - instruction: the instruction for the agent to perform the next step (Answer to the user query if finished)

            {format_instructions}

            Ensure that:
            - The instruction has all the necessary context to perform its task as the agents do not have access to previous steps or any history.
            - The instruction is as concise as possible.
            - Provide the exact paths obtained from the previous agent's output if it is relevant to the current agent.
            - The user query has been fully addressed before finishing the orchestration.
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

            # If the agent is the function calling agent, we need to handle it differently
            if agent_name == "function_calling_agent":
                parser = self.parsers["function_call"]
                format_instructions = parser.get_format_instructions()

                # Call the function calling agent to get the function name and arguments
                prompt = f"""
                You are a function calling agent. Your job is to return a JSON object with the exact function name and the arguments.
                Task: {instruction}
                Available functions: {self.available_functions}

                Return in a JSON object with the following fields:
                {format_instructions}

                Ensure that:
                - The function name is exactly as defined in the available functions.
                - The arguments are in the correct format and match the function's signature.
                """
                raw = await agent_llm.generate_str(prompt)
                print(f"\nRaw response from agent {agent_name}: {raw}")
                result = self.parsers["function_call"].parse(raw)
                function_call = result.model_dump()

                # Call the function with the provided arguments
                function_name = function_call["function_name"]
                arguments = function_call["arguments"]
                
                # Find the function in the available functions
                for func in self.available_functions:
                    if func.__name__ == function_name:
                        result_json["result"] = func(**arguments)
                        result_json["summary"] = f"Executed function {function_name} with arguments {arguments}"
                        print(f"Function {function_name} executed with result: {result_json['result']}")
                        break
            else:
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