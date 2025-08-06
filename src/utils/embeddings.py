import numpy as np
import hashlib
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similaridade de cosseno entre vetores"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

class SimpleTextEmbedder:
    """
    Embedder simples baseado em hash e features de texto
    Para uso quando sentence-transformers não está disponível
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        logger.info(f"Initialized SimpleTextEmbedder with dim={dim}")
    
    def encode(self, text: str) -> np.ndarray:
        """
        Gera embedding simples baseado em características do texto
        """
        if isinstance(text, list):
            return np.array([self._encode_single(t) for t in text])
        else:
            return self._encode_single(text)
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Gera embedding para um único texto"""
        text_lower = text.lower().strip()
        
        # Feature 1: Hash-based features
        text_hash = hashlib.md5(text_lower.encode()).hexdigest()
        hash_features = np.array([int(c, 16) for c in text_hash[:32]]) / 15.0
        
        # Feature 2: Character frequency features
        char_features = np.zeros(26)
        for char in text_lower:
            if 'a' <= char <= 'z':
                char_features[ord(char) - ord('a')] += 1
        char_features = char_features / max(len(text_lower), 1)
        
        # Feature 3: Word-based features
        words = text_lower.split()
        word_features = np.array([
            len(words),  # number of words
            np.mean([len(w) for w in words]) if words else 0,  # avg word length
            sum(1 for w in words if len(w) > 6),  # long words count
            sum(1 for w in words if w in ['ruído', 'noise', 'sound', 'equipamento', 'construction']),  # domain words
        ]) / 10.0  # normalize
        
        # Feature 4: Equipment type features
        equipment_types = ['martelo', 'serra', 'betoneira', 'britadeira', 'guindaste', 'caminhao']
        equipment_features = np.array([1.0 if eq in text_lower else 0.0 for eq in equipment_types])
        
        # Feature 5: Semantic category features
        categories = {
            'intensity': ['alto', 'baixo', 'forte', 'fraco', 'loud', 'quiet'],
            'time': ['manhã', 'tarde', 'noite', 'morning', 'afternoon', 'night'],
            'violation': ['violação', 'problema', 'violation', 'problem'],
            'analysis': ['análise', 'padrão', 'pattern', 'analysis']
        }
        
        category_features = []
        for category, terms in categories.items():
            score = sum(1 for term in terms if term in text_lower)
            category_features.append(min(score / 3.0, 1.0))  # normalize and cap
        category_features = np.array(category_features)
        
        # Combine all features
        all_features = np.concatenate([
            hash_features[:20],  # 20 dimensions
            char_features[:20],  # 20 dimensions  
            word_features,       # 4 dimensions
            equipment_features,  # 6 dimensions
            category_features    # 4 dimensions
        ])
        
        # Pad or truncate to desired dimension
        if len(all_features) < self.dim:
            # Pad with noise based on text content
            np.random.seed(abs(hash(text_lower)) % 2**32)
            padding = np.random.normal(0, 0.1, self.dim - len(all_features))
            all_features = np.concatenate([all_features, padding])
        else:
            all_features = all_features[:self.dim]
        
        # Normalize
        norm = np.linalg.norm(all_features)
        if norm > 0:
            all_features = all_features / norm
        
        return all_features

def get_embedding_model(model_name: Optional[str] = None):
    """
    Obtém modelo de embeddings com fallback para versão simples
    """
    try:
        # Tenta importar sentence-transformers
        from sentence_transformers import SentenceTransformer
        default_model = 'all-MiniLM-L6-v2'
        logger.info(f"Using SentenceTransformer: {model_name or default_model}")
        return SentenceTransformer(model_name or default_model)
    except ImportError as e:
        logger.warning(f"SentenceTransformers not available ({e}), using SimpleTextEmbedder fallback")
        return SimpleTextEmbedder()
    except Exception as e:
        logger.warning(f"Error loading SentenceTransformer ({e}), using SimpleTextEmbedder fallback")
        return SimpleTextEmbedder()