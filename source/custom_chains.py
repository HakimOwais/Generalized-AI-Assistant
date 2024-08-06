from utils import model, fetch_prompt

from logger import logging

class ChainHandler:
    def __init__(self):
        """
        Initializes the ChainHandler.
        """
        pass

    def create_chain(self, prompt_id: str):
        """
        Creates a chain using a prompt fetched based on the provided prompt ID.

        Args:
            prompt_id (str): The ID of the prompt used to create the chain.

        Returns:
            chain: A chain formed by concatenating the fetched prompt with the model.
        """
         # Fetch prompt based on the provided prompt ID
        logging.info("Fetching prompt")
        prompt = fetch_prompt(prompt_id)
        logging.info("Prompt fetched successfully.", prompt_id)

        # Concatenate the fetched prompt with the model
        logging.info("Creating chain by concatenating prompt with model.")
        chain = prompt | model
        logging.info("Chain created successfully.")
        
        return chain
    
    
#  test the code
# if __name__ == "__main__":
#     chain_handler = ChainHandler()
#     chain = chain_handler.create_chain('owais/langsmith_test')
#     print(chain)