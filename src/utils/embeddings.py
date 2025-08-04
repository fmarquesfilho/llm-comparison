import numpy as np
from sentence_transformers import SentenceTransformer
from ..config import settings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre vetores"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding_model(model_name: str = None):
    """Obtém modelo de embeddings"""
    return SentenceTransformer(model_name or settings.EMBEDDING_MODEL)