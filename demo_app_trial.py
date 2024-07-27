import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv
from streamlit import session_state as state

# Load environment variables
load_dotenv()

# Adding paths to import custom modules
sys.path.insert(1, "source")
sys.path.insert(2, "application")
sys.path.insert(3, "configuration")

from source.ast_main import execute_agent
from application.settings_manager import fetch_settings, insert_settings
from source.summarization import summarize
from automate_sends import send

# Get the Telegram token and chat ID from environment variables
my_token = os.getenv("TELEGRAM_TOKEN")
my_chat_id = 5654807603  # Replace with your chat ID

async def run_agent(in_params, settings, state):
    async for result in execute_agent(in_params, settings):
        state.results.append({"query": in_params["query"], "response": result})
    return state.results

# Define the main UI
def main():
    st.sidebar.title("Azal Tourism AI")
    page = st.sidebar.selectbox("Choose a page", ["AI Agent", "Telegram Summarizer"])

    if page == "AI Agent":
        ai_agent_page()
    elif page == "Telegram Summarizer":
        telegram_summarizer_page()


def ai_agent_page():
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
            

def telegram_summarizer_page():
    st.title("Telegram Message Summarizer")

    input_text = st.text_input("Enter the chat id to generate report")

    if st.button("Generate Report"):
        # Get the summarized text
        summarized_text = summarize(input_text)

        # Display the summarized text
        st.write("Report:")
        st.write(summarized_text)

        # Send the summarized text via Telegram
        if summarized_text:
            asyncio.run(send(msg=summarized_text, chat_id=my_chat_id, token=my_token))
            st.success("Message sent to Telegram!")
        else:
            st.error("Summarization failed. Please try again.")

if __name__ == "__main__":
    main()
