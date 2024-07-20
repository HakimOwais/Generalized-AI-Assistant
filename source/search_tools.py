import os
import re

from langchain_community.utilities import GoogleSerperAPIWrapper

from custom_chains import ChainHandler
from logger import logging

#Fetching the API from environment Variables
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

class SearchTool:
    def __init__(self, settings: dict):
        """
        Initializes the SearchTool with the provided settings.

        Args:
            settings (dict): Dictionary containing settings for the SearchTool.
        """
        self.settings= settings
        self.serper_api = GoogleSerperAPIWrapper()
        
    def perform_search(self, query: str):
        """
        Performs a search using the configured search API.

        Args:
            query (str): The search query.

        Returns:
            tuple: A tuple containing the search response and a list of extracted URLs.
        """
        website_url = self.settings.get("website_url", "")  # Extracting the website URL from settings
        modified_query = f"{website_url} {query}"  # Modifying query to include website URL
        logging.info(f"Performing search for query: {modified_query}")  # Logging search query
        response = self.serper_api.run(query=modified_query)  # Running the search query
        results = self.serper_api.results(modified_query)  # Retrieving search results
        logging.info("Search completed successfully.")  # Logging search completion
        return response, self._extract_urls(results)  # Returning search response and extracted URLs
    
    
    @staticmethod
    def _extract_urls(results: dict):
        """
        Extracts URLs from search results.

        Args:
            results (dict): The search results.

        Returns:
            list: A list of extracted URLs.
        """
        url_list = set(result['link'] for result in results.get('organic', []))
        return list(url_list)

    @staticmethod
    def clean_text(text: str):
        """
        Cleans the text by removing Markdown links and extra whitespaces.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        logging.info("Cleaning text...")  # Logging text cleaning process
        text_without_md_links = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Removing Markdown links
        cleaned_text = re.sub(r"\s+", " ", text_without_md_links).strip()  # Removing extra whitespaces
        logging.info("Text cleaning completed successfully.")  # Logging text cleaning completion
        return cleaned_text  # Returning cleaned text
    
    

class WebSearchTool:
    def __init__(self, settings: dict):
        """
        Initializes the WebSearchTool with the provided settings.

        Args:
            settings (dict): Dictionary containing settings for the WebSearchTool.
        """
        self.settings = settings
        self.search_tool = SearchTool(settings)
        self.chain_handler = ChainHandler()

    def wb_tool(self, query: str):
        """
        Executes the web search tool.

        Args:
            query (str): The search query.

        Returns:
            object: Results of the web search.
        """
        logging.info(f"Executing web search tool for query: {query}")  # Logging web search execution
        search_response, urls = self.search_tool.perform_search(query)  # Performing search using SearchTool
        search_response = self.search_tool.clean_text(search_response)  # Cleaning the search response
        logging.info("Search response cleaned successfully.")  # Logging search response cleaning
        # chain = self.chain_handler.create_chain(prompt_id=self.settings.get("prompt_id"))  # Creating a chain
        chain = self.chain_handler.create_chain(prompt_id=self.settings["Tools"]["wb_tool"]["prompt_id"])
        results = chain.invoke({"question": query, "context": (search_response, urls)})  # Invoking the chain
        logging.info("Web search tool execution completed.")  # Logging web search tool completion
        return results  # Returning the results of the web search
    

# if __name__ =="__main__":
    # settings = {
    #         "api_key": "your_api_key",
    #         "other_setting": "other_value"
    #     }
    # SearchTool(settings=settings)
    # WebSearchTool(settings=settings)