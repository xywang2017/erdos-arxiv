import os 
import torch 
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb 
import requests
import json

import google.generativeai as genai

genai.configure(api_key="AIzaSyDpCp8WUjjaE3mJsOXcdxdlAihxuGjJf7E")

# ------------------------------------- Get documents ------------------------------------- #
documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]

# above is a placeholder, need to implement proper documents 

# ------------------------------------- Document embeddings using SentenceTransformer ------------------------------------- #

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# doc_embeddings = embed_model.encode(documents)  # type: np.array

# create a vector database of documents, code to be added 
client = chromadb.PersistentClient('./database')
collection = client.get_or_create_collection(name="arxiv",metadata={"hnsw:space": "cosine"})  # if not given, embedding_fn defaults to SentenceTransformer 

collection.add(documents = documents, ids = [f"{j}" for j in range(len(documents))])

# ------------------------------------- Document retrieval ------------------------------------- #

user_input = "recommend something to do for a nature-loving person"

query_rag = collection.query(query_texts=[user_input],n_results = 5)
                  # where={"metadata_field": "is_equal_to_this"},
                  # where_document={"$contains":"search_string"})

prompt = """
    You are a question-answer bot that provides answers in the scientific domain. 
    Given the provided context: {rag_context}
    Answer user's question: {user_input} \n 
"""
if query_rag: 
    prompt_rag = prompt.format(rag_context=query_rag,user_input=user_input)
else: 
    prompt_rag = prompt.format(rag_context=" ",user_input=user_input)

# ------------------------------------- Connecting to Commerical ChatBot ------------------------------------- #
model = genai.GenerativeModel(model_name="gemini-1.5-pro") 
response = model.generate_content([prompt_rag]) 
print(response.text) 