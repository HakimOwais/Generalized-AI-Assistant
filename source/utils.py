# Import the necessary libraries
import json
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI

# Import local modules
from logger import logging
from exception import CustomException

os.environ.clear()

# Here we load the environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')
load_dotenv(dotenv_path)


def setup_model() -> object:
    """
    Initialize the large language model.

    Returns:
        object: A Large Language Model instance.
    """
    # Load the JSON string from the environment variable
    model_settings_str = os.environ.get("MODEL_SETTINGS")
    
    logging.info("Retrieved model settings string sucessfully: %s",
                 model_settings_str)
    

    # Parse the JSON string into a dictionary
    model_settings = json.loads(model_settings_str)
    
    # log the settings
    logging.info("Model settings loaded successfully", model_settings)

    # Create and configure the ChatOpenAI model instance
    llm_model = ChatOpenAI(
        model_name=model_settings.get('model_name'),
        streaming=model_settings.get('streaming'),
        callbacks=[StreamingStdOutCallbackHandler()],
        verbose=True
    )
    
    return llm_model


# Setup the model
model = setup_model()

def get_memory(session_id) -> object:
    """
    Retrieve conversational memory for a given session ID.

    This function retrieves the conversational memory associated with a specific session ID.
    The conversational memory contains the message history for the session stored in a Redis database.

    Args:
        session_id (str): The session ID for which memory is requested.

    Returns:
        object: Conversational Memory object.
    """
    logging.info("Retrieving conversational memory for session ID: ", session_id)

    # Create a RedisChatMessageHistory instance for storing message history
    redis_url = os.environ.get("REDIS_URL")
    message_history = RedisChatMessageHistory(url=redis_url, ttl=None, session_id=session_id)

    logging.info("Conversational memory retrieved successfully for session ID: %s", session_id)
    
    return message_history

def fetch_prompt(prompt_id) -> str:
    """
    Fetch a prompt from the Langchain hub.

    Args:
        prompt_id (str): The ID of the prompt to fetch.
    Returns:
        str: The fetched prompt.
    """
    prompt_hub= hub.pull(prompt_id)
    logging.info(f"fetched the prompt sucessfully {prompt_hub}")   
    return prompt_hub

# summarize("123")


# Testing the code
# if __name__ == "__main__":
#     fetch_prompt()
    