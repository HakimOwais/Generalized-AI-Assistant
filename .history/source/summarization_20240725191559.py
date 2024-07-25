import os
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sklearn.cluster import KMeans
import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

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


def summarize(id:str)->str:
    """
    Summarize the  chat uing the session id.
    """
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

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto').fit(vectors)
    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        
        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)
        
        # Append that position to your closest indices list
        closest_indices.append(closest_index)
        
    selected_indices = sorted(closest_indices)
    
    map_prompt = """
        You will be given a document.
        Your goal is to extract all the human questions and give a summary of this human questions so that a reader will have a full understanding of what happened.
        Your response should be precise and according to what was said in the passage also do sentimental analysis on it and categorize the human questions in the following three
        categories such as negative, positive and neutral assing.

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