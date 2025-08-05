import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre vetores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding_model(model_name: str = None):
    """Obtém modelo de embeddings"""
    default_model = 'all-MiniLM-L6-v2'
    return SentenceTransformer(model_name or default_model)