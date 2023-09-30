from sentence_transformers import SentenceTransformer


def embed_docs(sentences, embedding_model):
    global model
    model = SentenceTransformer(embedding_model)

    sentence_embeddings = model.encode(sentences, show_progress_bar=True)

    return sentence_embeddings


