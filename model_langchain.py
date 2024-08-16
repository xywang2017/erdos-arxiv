from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os 
import getpass

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

_MAX_DOCS = 10 
_CHUNK_SIZE = 5120 
_CHUNK_OVERLAP = 20

### Getting relevant paper from arxiv #####

# query = "recent progress on theoretical studies of twisted MoTe2"
query = input("Tell me a topic you are interested in investigating:")

print("Loading relevant documents and creating vector database ...",end=" ")
arxiv_docs = ArxivLoader(query=query, load_max_docs=_MAX_DOCS).load() 


##### splitting data ####
## added metadatas = [doc.metadata] to each chunk of data
pdf_data = []
for doc in arxiv_docs:
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=_CHUNK_SIZE,
                    chunk_overlap=_CHUNK_OVERLAP)
    texts = text_splitter.create_documents(texts=[doc.page_content],metadatas=[doc.metadata])
    for j in range(len(texts)):
        pdf_data.append(texts[j])

### embeddings 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
db = Chroma.from_documents(documents = pdf_data, embedding = embeddings) 

print("Done")

##
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                             temperature=0,
                             max_tokens=None,
                             timeout=None,
                             max_retries=2,)

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=db.as_retriever())


# question = "What are the topological and correlated phenomena in twisted MoTe2?"
question = input("Please ask your question:")
result = qa({"query": question})
print(result)