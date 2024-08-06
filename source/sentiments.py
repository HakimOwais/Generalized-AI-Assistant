import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



llm = ChatOpenAI(model="gpt-3.5-turbo")
redis_url = os.getenv("REDIS_URL")

def sentiment(id:str)-> str:
    message_history = RedisChatMessageHistory(
        url=redis_url,
        session_id=id
    )
    message_history = str(message_history)


    # messages = [
    #     SystemMessage(
    #         content = """
    # You will be given a document of a chat session between a Human and an AI, Go through the document 
   
    # Please extract aspect expressions, related segments and related sentiments from the following text and format output in JSON, strictly follow the below format:

    # This product is good but the battery doesn't last. It's lightweight and very easy to use. Well worth the money.

    # [
    #   {{ "aspect": "Overall satisfaction", "sentiment": "positive" }},
    #   {{ "aspect": "Battery", "sentiment": "negative" }},
    #   {{ "aspect": "Weight", "sentiment": "positive" }},
    #   {{ "aspect": "Usability", "sentiment": "positive" }},
    #   {{ "aspect": "Value for money", "sentiment": "positive" }}
    # ]

    # I don't like this product, it's very noisy. Anyway, it's very cheap. The other one I had was better.

    # [
    #   {{ "aspect": "Overall satisfaction", "sentiment": "negative" }},
    #   {{ "aspect": "Noise", "sentiment": "negative" }},
    #   {{ "aspect": "Price", "sentiment": "positive" }},
    #   {{ "aspect": "Comparison", "sentiment": "negative" }}
    # ]

    # ```{text}```
    # ABSA: 
    

    # """
    #     ),

    # HumanMessage(
    #     content=message_history
    # )
    # ]

    # print(llm.invoke(messages))
    # return
    print(message_history)


if __name__ == "__main":
    sentiment("abc123")


