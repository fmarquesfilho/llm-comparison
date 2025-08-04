from sentence_transformers import SentenceTransformer
from src.utils.time import fourier_time_embedding

class DEUBuilder:
    def __init__(self, model_name='all-MiniLM-L6-v2', lambda_time=0.5):
        self.model = SentenceTransformer(model_name)
        self.lambda_time = lambda_time

    def build_deu(self, event):
        embedding_txt = self.model.encode(event.tipo)
        embedding_time = fourier_time_embedding(event.timestamp)
        full_emb = np.concatenate([embedding_txt, self.lambda_time*embedding_time])
        return {
            **event.to_dict(),
            "embedding": full_emb
        }
