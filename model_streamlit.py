import streamlit as st 

import chromadb

from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import google.generativeai as genai


# ctrl_button = st.button('Reset',type='primary')
# if ctrl_button:
#     for key in st.session_state.keys():
#         del st.session_state[key]

st.sidebar.title("arXiv ChatBot")
google_api_key = st.sidebar.text_input(label="Enter your google API key:",type='password')
genai.configure(api_key=google_api_key)

### create databse
client = chromadb.Client()
collection = client.get_or_create_collection(name="arxiv",metadata={"hnsw:space": "cosine"}) 

if google_api_key:
    _MAX_DOCS = 100
    _CHUNK_SIZE = 5120
    _CHUNK_OVERLAP = 20

    # ------------------------------------- Get documents and vector database------------------------------------- #
    user_input = st.sidebar.text_input("Tell me a topic you are interested in",key="topic")

    if user_input:
        st.sidebar.write("Building a database of the topic based on arXiv.org ...")

        # create_collection has a embedding_fn parameter. If not given, embedding_fn defaults to SentenceTransformer 
        

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

        collection.upsert(documents=page_content,ids=doc_identifiers,metadatas=metadata)


        # st.write(f"Retrieved and encoded {_TOTAL_DOCS} documents.") 
        st.sidebar.write("Here are a few examples of retrieved papers:")
        tmp = collection.query(query_texts=[user_input],n_results = min(_TOTAL_DOCS,3))
        for j in range(len(tmp['metadatas'][0])):
            data = tmp['metadatas'][0][j]
            st.sidebar.write(f"Title: {data['Title']} (arXiv:{data['arxiv_specifier']})")

        # ------------------------------------- Question and Answer with Commerical ChatBot ------------------------------------- #

        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        query = st.chat_input(f'What do you want to know regarding "{user_input}"?',key="question_answer")
        if query:
            query_rag = collection.query(query_texts=[query],n_results = min(5,_TOTAL_DOCS))
            st.write("Here are the papers retrieved based on your question:")
            for j in range(len(query_rag['metadatas'][0])):
                data = query_rag['metadatas'][0][j]
                st.write(f"Title: {data['Title']} (arXiv:{data['arxiv_specifier']})")
                st.write(f"Authors: {data['Authors']}")
                # st.write(query_rag['metadatas'][0][j])

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

            st.write(response.text) 