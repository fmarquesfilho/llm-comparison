# Sugestão: FAISS como backend
import faiss
import numpy as np

class VectorIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add_embeddings(self, embeds):
        # embeds: np.ndarray shape (N, dim)
        self.index.add(embeds)

    def search(self, embed_query, k=5):
        D, I = self.index.search(np.array([embed_query]), k)
        return I[0], D[0]
