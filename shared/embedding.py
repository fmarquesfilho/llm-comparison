import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def fourier_time_embedding(timestamp, base_period=86400):
    t_sec = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    return np.array([
        np.sin(2 * np.pi * t_sec / base_period),
        np.cos(2 * np.pi * t_sec / base_period)
    ])

def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

def embed_deu(deu):
    base_text = f"{deu.tipo_som} {deu.texto or ''}"
    text_embed = model.encode(base_text)
    time_embed = fourier_time_embedding(deu.timestamp)
    return np.concatenate([text_embed, time_embed])
