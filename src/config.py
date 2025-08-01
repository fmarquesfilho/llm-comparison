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
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    index_type: str = "faiss"

@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    #target_modules: List[str] = None
    target_modules: List[str] = field(default_factory=list)  # Corrigido aqui
    bias: str = "none"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class Config:
    # Caminhos
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Modelos disponíveis
    MODELS = {
        "mistral-7b": ModelConfig(
            name="Mistral-7B-Instruct",
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_length=4096,
            temperature=0.7
        ),
        "llama2-7b": ModelConfig(
            name="Llama-2-7B-Chat",
            model_id="meta-llama/Llama-2-7b-chat-hf",
            max_length=4096,
            temperature=0.7
        )
    }
    
    # Configurações padrão
    RAG = RAGConfig()
    LORA = LoRAConfig()
    
    # Avaliação
    EVALUATION_METRICS = ["rouge_l", "bleu", "bert_score", "factual_accuracy"]
    
    @classmethod
    def setup_directories(cls):
        """Cria diretórios necessários"""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR]:
            for subdir in ["raw", "processed", "evaluation"]:
                (dir_path / subdir).mkdir(parents=True, exist_ok=True)
                