"""
Assistant Manager - ast_main
"""

import os
import importlib

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

# from tools_lib import tools_list
from source.utils import model, get_memory, fetch_prompt

from dotenv import load_dotenv

# load_dotenv("/home/bhat/ALQ/ALQ-Projects/mowasalat_bot/.env")
load_dotenv()


class AgentManager:
    def __init__(self, settings: dict):
        """
        Initializes the AgentManager with the given settings.

        Args:
            settings (dict): Dictionary containing settings for the agent.
        """

        self.settings= settings

    def initialize_agent(self) -> RunnableWithMessageHistory:
        """
        Initializes and configures the agent for execution.

        Returns:
            RunnableWithMessageHistory: An agent runnable with message history.
        """

        # Fetch Prpompt/ AGent prompt from settings
        prompt_id= self.settings["parent_settings"]["agent_id"]
        prompt= fetch_prompt(prompt_id)

        tool_id = self.settings["parent_settings"]["tool_id"]
        tools_name = f'tools_list_{tool_id}'
        tools_module = importlib.import_module("tools_lib")
        tools_list = getattr(tools_module, tools_name)
        
        agent= create_openai_tools_agent(model, tools_list, prompt)



        # Creating AgentExecuter for agent execution
        agent_executer= AgentExecutor(
            agent=agent,
            tools=tools_list,
            verbose=True,
            return_intermediate_steps=True,
            early_stopping_method="generate"
        )

        agent_with_history= RunnableWithMessageHistory(
            agent_executer,
            lambda session_id: get_memory(session_id),
            input_messages_key="input",
            history_messages_key="chat_history",

        )

        return agent_with_history

# def execute_agent(in_params: dict, settings: dict):
#     """
#     Executes the agent using the provided input parameters and settings.

#     Args:
#         in_params (dict): Input parameters for the agent execution.
#         settings (dict): Settings required for agent initialization and execution.

#     Returns:
#         str: Result of the agent execution.
#     """
#     session_id = in_params["session_id"]

#     # Initialize Agent manager with settings
#     agent_manager= AgentManager(settings=settings)
#     try:
#         # initialize and configure the agent
#         agent= agent_manager.initialize_agent()

#         # invoke the agent with input params
#         result= agent.invoke({
#             "input": in_params["query"]
#         }, {
#             "configurable":{
#                 "session_id": session_id
#             }
#         })
#         result = result["output"]
#     except Exception as e:
#         print(f"Error during agent execution: {e}")
#         # Return an error message in case of exception
#         result = "Internal Error, If the issue persists please call admin"
    
#     return result

async def execute_agent(in_params: dict, settings: dict):
    """
    Executes the agent using the provided input parameters and settings.

    Args:
        in_params (dict): Input parameters for the agent execution.
        settings (dict): Settings required for agent initialization and execution.

    Yields:
        str: Incremental results of the agent execution.
    """
    session_id = in_params["session_id"]

    # Initialize Agent manager with settings
    agent_manager = AgentManager(settings=settings)
    try:
        # Initialize and configure the agent
        agent = agent_manager.initialize_agent()
        
        async def agent_stream_async():
            # Use the agent's async stream method if it exists
            async for chunk in agent.astream(
                {
                    "input": in_params["query"]
                },
                {
                    "configurable": {
                        "session_id": session_id
                    }
                }
            ):
                yield chunk

        async for chunk in agent_stream_async():
            content = chunk.get('output', {})
            if content:
                # Log the chunk being sent
                # logging.info("Sending chunk: %s", content)
                yield f"{content}\n\n"
    except Exception as e:
        # Return an error message in case of exception
        result = "Internal Error, If the issue persists please call admin"
        yield result

