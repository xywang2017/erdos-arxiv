import chromadb 
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai

genai.configure(api_key="AIzaSyDpCp8WUjjaE3mJsOXcdxdlAihxuGjJf7E")

_MAX_DOCS = 100
_CHUNK_SIZE = 5120
_CHUNK_OVERLAP = 20

# ------------------------------------- Get documents and vector database------------------------------------- #
user_input = input("Tell me a topic you are interested in \n")

print("Building a database of the topic based on arXiv.org ...")

# create_collection has a embedding_fn parameter. If not given, embedding_fn defaults to SentenceTransformer 
client = chromadb.Client()
collection = client.create_collection(name="arxiv",metadata={"hnsw:space": "cosine"})  

# using LangChain arXivLoader to load document summaries (i.e. no pdfs)
arxiv_docs = ArxivLoader(query=user_input, top_k_results=_MAX_DOCS, load_all_available_meta=True).get_summaries_as_docs()

_TOTAL_DOCS = len(arxiv_docs)

page_content, metadata, doc_identifiers = [], [], []
for doc in arxiv_docs:
    arxiv_specifier = doc.metadata['Entry ID'].split('/')[-1]
    arxiv_title = doc.metadata['Title']
    arxiv_authors = doc.metadata['Authors']

    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=_CHUNK_SIZE,
                    chunk_overlap=_CHUNK_OVERLAP)
    texts = text_splitter.create_documents(texts=[doc.page_content],metadatas=[doc.metadata])
    
    for j in range(len(texts)):
        page_content.append(texts[j].page_content)
        metadata.append({'arxiv_specifier':arxiv_specifier,'Title':arxiv_title,'Authors':arxiv_authors})
        doc_identifiers.append(str(hash(arxiv_specifier + f' {j}')))

collection.add(documents=page_content,ids=doc_identifiers,metadatas=metadata)


print(f"Retrieved and encoded {_TOTAL_DOCS} documents.") 
# print("Here are a few examples of retrieved papers:")
# tmp = collection.query(query_texts=[user_input],n_results = min(_TOTAL_DOCS,5))
# for j in range(len(tmp['metadatas'][0])):
#     print(tmp['metadatas'][0][j])
print("\n")

# ------------------------------------- Question and Answer with Commerical ChatBot ------------------------------------- #
try:
    model = genai.GenerativeModel(model_name="gemini-1.5-pro");
    while True:
        query = input(f'What do you want to know regarding "{user_input}"?\n')
        query_rag = collection.query(query_texts=[query],n_results = min(5,_TOTAL_DOCS))
        print("Here are the papers retrieved based on your question:")
        for j in range(len(query_rag['metadatas'][0])):
            print(query_rag['metadatas'][0][j])
        print("\n")

        prompt = """
            You are a question-answer bot that provides answers in the scientific domain. 
            Given the provided context: {rag_context}
            Answer user's question "{query}" on the topic of {user_input}. 
            When answering the question, try to make use of arxiv_specifiers provided in the context.\n 
        """

        rag_context = []
        for j in range(len(query_rag['metadatas'][0])): 
            context = query_rag['metadatas'][0][j]
            context['documents'] =  query_rag['documents'][0][j]
            rag_context.append(context)

        prompt_rag = prompt.format(rag_context=rag_context,user_input=user_input,query=query)

        response = model.generate_content([prompt_rag]);

        print(response.text) 
except KeyboardInterrupt:
    pass