import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models():
    """Baixa modelos base e de embedding necessários"""
    base_models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "microsoft/DialoGPT-medium"  # fallback menor
    ]
    embedding_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    base_dir = Path("models/base")
    embed_dir = Path("models/embeddings")
    base_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    # Baixa modelos base (tokenizer + pesos completos)
    for model_id in base_models:
        try:
            logger.info(f"Baixando modelo base: {model_id}...")
            # Sempre baixa pesos completos para CPU (torch.float32)
            AutoTokenizer.from_pretrained(model_id, cache_dir=str(base_dir))
            AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=None,  # Compatível com CPU, evita float16
                cache_dir=str(base_dir)
            )
            logger.info(f"✓ Modelo base {model_id} baixado com sucesso")
        except Exception as e:
            logger.error(f"Falha ao baixar {model_id}: {e}")

    # Baixa modelos de embedding
    for embed_id in embedding_models:
        try:
            logger.info(f"Baixando modelo embedding: {embed_id}...")
            SentenceTransformer(
                embed_id, 
                cache_folder=str(embed_dir)
            )
            logger.info(f"✓ Embedding {embed_id} baixado com sucesso")
        except Exception as e:
            logger.error(f"Falha ao baixar embedding {embed_id}: {e}")

def main():
    logger.info("Iniciando download dos modelos...")
    download_models()
    logger.info("Download concluído!")

if __name__ == "__main__":
    main()
