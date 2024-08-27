# FAQ, TODO, IDEAS

* cannot share api keys. each individual has to have his own.
* openai does not allow sharing keys
* becareful with api keys when pushing into git
* can use keybase or lastpass to share api keys among team members??


## Q. In the Towardsdatascience article the pipeline does not have a precomputed embeddings database. For each query there is a call to arxiv database and then relevant documents are preprocessed. Why do this?
I thought that we could generate embeddings for all documents in arxiv and store it in Chroma DB before any query. Then once we have the query, we can retrieve it.

Answer from Perplexity

You're absolutely right. The ingestion process can be optimized by pre-computing and storing the embeddings for all the documents in the arXiv database, before any specific user query comes in.Here's how the optimized ingestion process would look like:

1. **Retrieve all papers from arXiv**: Use the ArxivLoader to retrieve all the research papers from the arXiv API, without filtering by any specific query.
2. **Split papers into chunks**: Split the retrieved papers into smaller chunks using the RecursiveTextSplitter, as mentioned earlier.
3. **Generate embeddings for all chunks**: Create embeddings for each chunk using sentence-transformers from HuggingFace. This step can be done in advance, independent of any user query.
4. **Ingest embeddings into Chroma DB**: Store the document chunks along with their pre-computed embeddings into a Chroma DB database. This is a one-time setup process.

Now, whenever a user query comes in:

1. **Generate query embedding**: Create an embedding for the user's query using the same sentence-transformers model.
2. **Retrieve relevant chunks from Chroma DB**: Use the query embedding to perform a nearest neighbor search in Chroma DB. This will retrieve the most semantically similar document chunks to the query.
3. **Return relevant chunks to user**: Present the retrieved chunks to the user as the most relevant excerpts from research papers.

By pre-computing the embeddings and storing them in Chroma DB, the system can respond to user queries much faster, without having to generate embeddings on-the-fly. This optimization is crucial for building a scalable and responsive semantic search engine.


## Q. What if the user asks question from a particular paper?
Answer from Perplexity.

If the user asks a question that is specific to a particular research paper, the retrieval process can be optimized further. Here's how the system can handle such queries:

1. **Identify the specific paper**: When the user asks a question, first check if they have mentioned the title or author of a specific paper. This indicates that the user is looking for information within that paper.
2. **Retrieve relevant chunks from that paper**: If a specific paper is identified, use the paper's metadata (title, author, abstract, etc.) to search for and retrieve only the chunks from that paper, ignoring other papers.
3. **Rank retrieved chunks**: Rank the retrieved chunks based on their relevance to the user's question. Factors like the chunk's position in the paper (e.g. abstract, introduction, conclusion), the number of matching keywords, and the semantic similarity can be used for ranking.
4. **Generate answer from top chunks**: Pass the top-ranked chunks to the language model to generate the final answer. The language model can combine information from multiple chunks to provide a comprehensive response.

This targeted retrieval approach has several advantages:

- It reduces the search space to only the relevant paper, making retrieval faster.
- It ensures that the generated answer is focused and specific to the user's question within the context of the mentioned paper.
- It provides a better user experience by directly addressing the user's intent to get information from a particular paper.

However, this approach assumes that the user always specifies the paper they are asking about. If the user asks a more general question that could be answered by multiple papers, the system should fall back to the original retrieval process across all papers.By combining both targeted and general retrieval strategies, the system can handle a wide range of user questions efficiently and provide relevant answers from the ingested research papers.


## How can I host a Chainlit server?
Answer from perplexity:
Based on the Chainlit documentation, there are several options for hosting a Chainlit server:

Hosting Options

1. **Self-hosting**: You can self-host the Chainlit server on your own infrastructure. This gives you full control but requires managing the hosting environment yourself.
2. **Hosting services**: Chainlit provides guides for deploying on popular cloud platforms like:
    
    - AWS
    - Azure Container
    - Google Cloud Run
    - Google App Engine
    - Replit
    - Render
    - Fly.io
    - HuggingFace Spaces
    
3. **Literal AI Server**: If your organization requires self-hosting for data privacy reasons, Chainlit offers a self-hostable Literal AI Docker image under a commercial license. You can request access by contacting them.

Deployment Steps

1. **Set up the Chainlit application locally** and test it.
2. **Deploy the application to your chosen hosting service**. Chainlit is built on websockets, so the hosting platform must support them. Enable sticky sessions for auto-scaling.
3. **If deploying to a subpath** (e.g. `https://myapp.com/chainlit`), use the `--root-path` flag when running `chainlit run` to serve the app from the correct path.
4. **Always use the `-h` (headless) flag** when running `chainlit run` in production to prevent opening a browser window server-side.
5. **Configure the `LITERAL_API_URL` environment variable** if using a self-hosted Literal AI instance for data persistence.

The Chainlit documentation provides detailed guides for setting up the local development environment and deploying to various hosting platforms. With these options, you can host your Chainlit server based on your specific requirements and infrastructure.









## TODO
1. Decide on whether to precompute all the embeddings vs compute embeddings for each query.

## IDEAS
1. Arxiv search is not that good. Maybe we can tweak it.

## Links
1. [experience training a local llm](https://peterfalkingham.com/2024/04/26/my-experience-training-a-local-llm-ai-chatbot-on-local-data/)
2. [Towards DataScience article](https://towardsdatascience.com/building-an-observable-arxiv-rag-chatbot-with-langchain-chainlit-and-literal-ai-9c345fcd1cd8)
3. 
