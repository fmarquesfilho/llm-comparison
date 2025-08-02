# src/config.py
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    name: str
    model_id: str
    max_length: int
    temperature: float
    device: str = "auto"

@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    # Modelo leve para MVP
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_type: str = "faiss"

@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"

class Config:
    # Caminhos
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Modelos simplificados para MVP
    MODELS = {
        # Modelo pequeno para testes locais
        "small-local": ModelConfig(
            name="DistilBERT-Base",
            model_id="distilbert-base-uncased",
            max_length=512,
            temperature=0.7
        ),
        # Backup ainda menor se necessário
        "tiny-local": ModelConfig(
            name="TinyBERT",
            model_id="huawei-noah/TinyBERT_General_4L_312D",
            max_length=256,
            temperature=0.7
        ),
        # API externa como baseline
        "openai-api": ModelConfig(
            name="OpenAI-GPT-3.5-Turbo",
            model_id="gpt-3.5-turbo",
            max_length=4096,
            temperature=0.7
        )
    }
    
    # Configurações padrão
    RAG = RAGConfig()
    LORA = LoRAConfig()
    
    # Métricas simplificadas para MVP
    EVALUATION_METRICS = ["rouge_l", "response_time", "cost_per_query", "relevance_score"]
    
    # API Keys (via .env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    @classmethod
    def setup_directories(cls):
        """Cria diretórios necessários"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "embeddings",
            cls.DATA_DIR / "evaluation",
            cls.MODELS_DIR / "embeddings",
            Path("logs")
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Detecta melhor device disponível para M1 Mac"""
        import torch
        
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    @classmethod
    def validate_setup(cls):
        """Valida se o ambiente está configurado corretamente"""
        import torch
        from sentence_transformers import SentenceTransformer
        
        issues = []
        
        # Verifica PyTorch
        try:
            device = cls.get_device()
            print(f"✅ PyTorch {torch.__version__} - Device: {device}")
        except Exception as e:
            issues.append(f"❌ PyTorch: {e}")
        
        # Verifica sentence-transformers
        try:
            model = SentenceTransformer(cls.RAG.embedding_model)
            print(f"✅ Embedding model: {cls.RAG.embedding_model}")
        except Exception as e:
            issues.append(f"❌ Sentence Transformers: {e}")
        
        # Verifica API keys se necessário
        if not cls.OPENAI_API_KEY:
            issues.append("⚠️ OPENAI_API_KEY não configurada (opcional para MVP)")
        
        # Verifica diretórios
        cls.setup_directories()
        print("✅ Diretórios criados")
        
        if issues:
            print("\n⚠️ Problemas encontrados:")
            for issue in issues:
                print(f"  {issue}")
            return False
        
        print("\n🎉 Setup validado com sucesso!")
        return True
    