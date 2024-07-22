import streamlit as st
import asyncio
from streamlit import session_state as state
import sys


# Adding paths to import custom modules
sys.path.insert(1, "source")
sys.path.insert(2, "application")
sys.path.insert(3, "configuration")


from source.ast_main import execute_agent
from application.settings_manager import fetch_settings, insert_settings


async def run_agent(in_params, settings, state):
    async for result in execute_agent(in_params, settings):
        state.results.append({"query": in_params["query"], "response": result})
    return state.results


# Define the Streamlit UI
def main():
    st.title("Azal Tourism AI Agent")
    
    # Hidden field for app_id
    app_id = "wildfloc"
    session_id = 'abc123'

    # Initialize the session state
    if 'results' not in st.session_state:
        st.session_state.results = []

    # Display previous queries and responses
    for interaction in st.session_state.results:
        st.write(f"**Query:** {interaction['query']}")
        st.write(f"**Response:** {interaction['response']}")

    # Chat box input at the bottom
    query = st.text_input("Query")
    
    if query:
        in_params = {"app_name": app_id, "session_id": session_id, "query": query}
        settings = fetch_settings(app_id)

        if st.button("Run Agent"):
            async def get_results():
                return await run_agent(in_params, settings, st.session_state)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(get_results())
            st.session_state.results = results
            st.rerun()

if __name__ == "__main__":
    main()