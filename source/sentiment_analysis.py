import os
from typing import List
import openai
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sklearn.cluster import KMeans
import numpy as np

from langchain.chains.combine_documents.stuff import StuffDocumentsChain

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
redis_url = os.getenv("REDIS_URL")

llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

def sentiment(id:str)-> str:
    message_history =  RedisChatMessageHistory(
        url=redis_url,
        session_id=id
    )

    message_history = str(message_history)
    text = ""

    for page in message_history:
        text+=page

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=50)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()

    vectors = embeddings.embed_documents([_.page_content for _ in docs])

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
    You will be given a document of a chat session between a Human and an AI, Go through the document 
   
    Please extract aspect expressions, related segments and related sentiments from the following text and format output in JSON, strictly follow the below format:

    This product is good but the battery doesn't last. It's lightweight and very easy to use. Well worth the money.

    [
      {{ "aspect": "Overall satisfaction", "sentiment": "positive" }},
      {{ "aspect": "Battery", "sentiment": "negative" }},
      {{ "aspect": "Weight", "sentiment": "positive" }},
      {{ "aspect": "Usability", "sentiment": "positive" }},
      {{ "aspect": "Value for money", "sentiment": "positive" }}
    ]

    I don't like this product, it's very noisy. Anyway, it's very cheap. The other one I had was better.

    [
      {{ "aspect": "Overall satisfaction", "sentiment": "negative" }},
      {{ "aspect": "Noise", "sentiment": "negative" }},
      {{ "aspect": "Price", "sentiment": "positive" }},
      {{ "aspect": "Comparison", "sentiment": "negative" }}
    ]

    ```{text}```
    ABSA: 
    """ 

    map_prompt_template = PromptTemplate(template=map_prompt, input_variables = ["text"])

    llm_chain = LLMChain(llm=llm, prompt=map_prompt_template)

    map_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    selected_docs = [docs[doc] for doc in selected_indices]
    
    sentiments_list = []

    for i, doc in enumerate(selected_docs):

        chunk_sentiments = map_chain.invoke([doc])

        sentiments_list.append(chunk_sentiments)
    
    print(chunk_sentiments['output_text'])

    # return chunk_sentiments['output_text']
    return 


if __name__  == "__main__":
    sentiment("abc123")
