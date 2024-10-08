import streamlit as st 
import numpy as np 
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import ArxivLoader

import google.generativeai as genai

st.sidebar.title("arXiv.org ChatBot")
st.sidebar.write('A RAG-based chatbot to query any paper on arXiv. The chatbot uses Google Gemini LLM under the hood and was develop as part of the Erdos Institute Deep Learning Bootcamp. The Team includes: Tantrik Mukherji, Ketan Sand, Xiaoyu Wang, Tajudeen Mamadou Yacoubou, Guoqing Zhang.')
url = "https://aistudio.google.com/app/apikey"
st.sidebar.write(f"[Click here to get an API key if you don't have one]({url})")
st.sidebar.write('Click "Create API key" -> Done!')
google_api_key = st.sidebar.text_input(label="Enter your google API key:",type='password')
genai.configure(api_key=google_api_key)

if google_api_key:
    _MAX_DOCS = 100

    model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

    # ------------------------------------- Get documents and vector database------------------------------------- #

    user_input = st.sidebar.text_input("Tell me a topic you are interested in",key="topic")
    
    if user_input:
        user_input_embedding = model_sbert.encode([user_input])
        st.sidebar.write(f'Building a database on the topic "{user_input}" from arXiv.org ...')

        # using LangChain arXivLoader to load document summaries (i.e. no pdfs)
        arxiv_docs = ArxivLoader(query=f'"{user_input}"', top_k_results=_MAX_DOCS, load_all_available_meta=True).get_summaries_as_docs()

        if len(arxiv_docs) == 0: 
            st.sidebar.write('No documents matching the topic is found, try again!')
        else:
            page_content, metadata, doc_identifiers = [], [], []
            for doc in arxiv_docs:
                arxiv_specifier = doc.metadata['Entry ID'].split('/')[-1]
                arxiv_title = doc.metadata['Title']
                arxiv_authors = doc.metadata['Authors']
                doc_str = 'Title: '+ arxiv_title + 'Authors: ' + arxiv_authors + 'Summary: ' + doc.page_content
                
                page_content.append(doc_str)
                metadata.append({'arxiv_specifier':arxiv_specifier,'Title':arxiv_title,'Authors':arxiv_authors})
                doc_identifiers.append(str(hash(arxiv_specifier)))

            arxiv_embedding = model_sbert.encode(page_content)

            sim_scores = model_sbert.similarity(user_input_embedding, arxiv_embedding)  # 2d [0][j]

            idx_random_scores = np.random.choice(np.arange(0,len(sim_scores[0])),min(5,len(sim_scores[0])))

            if len(sim_scores[0])<6: 
                st.sidebar.write(f"Few papers found, consider lessening the keyword constraints in topic search.")
            st.sidebar.write(f"Here are a few examples ({len(idx_random_scores)} of {len(sim_scores[0])}) of retrieved papers:")
            cnt = 1
            for j in idx_random_scores:
                data = metadata[j]
                st.sidebar.write(f"[{cnt}] *{data['Title']}*, arXiv:{data['arxiv_specifier']} (relevance score: {sim_scores[0][j]:.2f})")
                cnt = cnt + 1

            # ------------------------------------- Question and Answer with Commerical ChatBot ------------------------------------- #

            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            query = st.chat_input(f'What do you want to know regarding "{user_input}"?',key="question_answer")
        
            if query:
                query_embedding = model_sbert.encode([user_input + ' ' + query])
                sim_scores = model_sbert.similarity(query_embedding, arxiv_embedding)  # 2d [0][j]
                idx_max_scores = np.argsort(np.array(sim_scores[0]))[-5:]

                st.write(f'Here are the papers retrieved based on your question "{query}":')
                cnt = 1
                for j in reversed(idx_max_scores):
                    data = metadata[j]
                    st.write(f"[{cnt}] {data['Authors']}, *{data['Title']}*, arXiv:{data['arxiv_specifier']}")
                    st.write(f"Paper Relevance Score (0-1): {sim_scores[0][j]:.2f}")
                    cnt = cnt + 1

                prompt = """
                    You are a question-answer bot that provides answers in the scientific domain. 
                    Given the provided context: {rag_context}
                    Answer user's question "{query}" on the topic of {user_input}. 
                    When answering the question, try to make use of arxiv_specifiers provided in the context.\n 
                """

                rag_context = []
                for j in idx_max_scores: 
                    context = metadata[j]['arxiv_specifier'] + ': ' + page_content[j]
                    rag_context.append(context)

                prompt_rag = prompt.format(rag_context=rag_context,user_input=user_input,query=query)

                response = model.generate_content([prompt_rag])

                st.write(response.text) 
