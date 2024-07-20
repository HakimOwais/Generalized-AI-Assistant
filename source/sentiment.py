import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

import numpy as np
from langchain_core.prompts import PromptTemplate

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain



from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0,
                 model='gpt-3.5-turbo'
                )

# Get the Redis URL from the environment variable
redis_url = os.environ.get("REDIS_URL")

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')


def sentiment_analysis(id):


    message_history = RedisChatMessageHistory(
        url=redis_url,
        session_id=id
        )
        
    message_history = str(message_history)  
    text = ''


    for page in message_history:
        text += page
        
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=50)

    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()

    vectors = embeddings.embed_documents([x.page_content for x in docs])
    
    num_clusters = 1

    map_prompt = """
        You will be given a chat session.
        Your goal is to analyze all the human questions and do a sentimental analysis on the responses and provide a dictionatr.
        Your response should be precise and according to what was said in the passage.

        ```{text}```
        FULL SUMMARY:
        """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    llm_chain = LLMChain(llm=llm, prompt=map_prompt_template)
    
    map_chain = StuffDocumentsChain(llm_chain=llm_chain,
                             document_variable_name="text",
                             )
    

    selected_docs = [docs[doc] for doc in selected_indices]
    
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        
        # Go get a summary of the chunk
        chunk_summary = map_chain.invoke([doc])
        
        # Append that summary to your list
        summary_list.append(chunk_summary)
        
    return chunk_summary['output_text']
