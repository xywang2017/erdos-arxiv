# erdos-arxiv

arXiv chatbot, Erdos Institute Deep Learning Boot Camp

Standard search methods on arXiv.org are outdated, and based on keyword matching. ChatGPT appears to do better, with a few examples shown in the pictures below. 

In this project proposal, We would do something similar to chatGPT, but better in the academic domain by incorporating retrieval augmented generation (RAG). Specifically we are interested to address questions like: 

- I want to research XYZ, could you provide a summary of the research results in the past month? 
- Are there common topics both domain A and domain B are working on, but researchers are too lazy to spot it (by hopping over to a different domain and suffering from a different set of jargons)? 
- Could you provide a summary of the research results in a given paper arXiv:0123.45678?
- Can you give me a select of recent 5 papers on topic XXX, and give a summary of their main results?
- Possible multimodal NLP task combining text and figures

Project skills: web scraping, NLP, NLP fine tuning methods such as RAG, model evaluation metrics, and deployment

How to evaluate model performance? 
- Is generated summary similar to the actual abstract?  
— Plagiarism detection via GAN

Stakeholders
- arXiv.org
- General online archiving databases (e.g. Google scholar)
- Field-based online archives
- Professional Associations

July 1 Rough Task Division: 
- Data preprocessing
  - Getting arXiv metadata and paper content (text, figure) from the past year, create a RAG vector database (word embedding)
  - For a figure,a separate task to translate it to a texual description is needed to create a word embedding
  - Main contributors: XXX
- Core model
  - Create RAG + NLP pipeline
  - Main contributors: XXX
- Model evaluation
  - Evaluate model for various tasks
  - Main contributors: XXX
- Deployment
  - web deployment
  - Main contributors: XXX
- Exploratory Tasks
  - multimodel NLP
  - customized transformer architecture (poor man's version of chatGPT)
  - create a Transformer-GAN for plagarism detection.
