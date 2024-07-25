import sys

# Adding paths to import custom modules
sys.path.insert(1, "source")
sys.path.insert(2, "application")
sys.path.insert(3, "configuration")

from langchain_core.tools import tool
from source.search_tools import WebSearchTool
from application.settings_manager import fetch_settings


@tool
def search_tool_wildfloc(question: str):
    """This is a websearch tool that you can use for question about any website url"""
    settings = fetch_settings("wildfloc")
    web_search = WebSearchTool(settings)
    response = web_search.wb_tool(question)
    return response


@tool
def search_tool_azal(question: str):
    """This is a websearch tool that you can use for question about any website url"""
    settings = fetch_settings("azal")
    web_search = WebSearchTool(settings)
    response = web_search.wb_tool(question)
    return response


# Putting all tools together
tools_list_wildfloc = [search_tool_wildfloc, search_tool_azal]
