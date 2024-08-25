import pandas as pd
import chromadb 

# ------------------------------------- Get document metadata ------------------------------------- #
arxiv_dataset = './arxiv-metadata-oai-snapshot.json'
print("Loading arxiv dataset ...")
df = pd.read_json(arxiv_dataset, lines=True)
df = df[df['update_date']>'2016-01-01']  # build database from recent year publications
df.drop(columns=['submitter','comments',
                 'journal-ref','doi','report-no','license','versions','update_date','authors_parsed'],inplace=True)
df.drop_duplicates(subset=['id'],inplace=True)
df.dropna(inplace=True)
print("Done")

# reformat the dafaframe
def get_entries(df):
    ids, metadatas, documents = [], [], []
    column_names = list(df.columns)

    print('Begin arxiv-metadata reformatting ...')

    for i in range(len(df)):
        entry = df.iloc[i]
        document = entry['abstract']
        metadata = {column_names[j]:entry[column_names[j]] for j in range(1,len(column_names)-1)}
        id = entry[column_names[0]]  # arXiv:1313.1244 
        ids.append(id)
        metadatas.append(metadata)
        documents.append(document)
    print('Done')
    return ids, metadatas, documents


# ------------------------------------- Document embeddings using SentenceTransformer ------------------------------------- #

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# doc_embeddings = embed_model.encode(documents)  # type: np.array

# create a vector database of documents, code to be added 
client = chromadb.PersistentClient(path="./database")


# if an embedding function is not given, embedding_fn parameter defaults to SentenceTransformer 'all-MiniLM-L6-v2'

collection = client.create_collection(name="arxiv-metadata",metadata={"hnsw:space": "cosine"})  

ids, metadatas, documents = get_entries(df)

print("Adding arxiv metadata to vector database ...")
for j in range(0,len(ids),1000):
    print(f"Adding data to collection, part {j//1000}")
    if 1000*(j+1) < len(ids):
        collection.add(documents = documents[j:(j+1000)], ids = ids[j:(j+1000)], metadatas = metadatas[j:(j+1000)])
    else:
        collection.add(documents = documents[j:], ids = ids[j:], metadatas = metadatas[j:])
print("Done")
