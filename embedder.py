from sentence_transformers import SentenceTransformer

model = None

def embed_docs(sentences):
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    sentence_embeddings = model.encode(sentences)

    return sentence_embeddings


