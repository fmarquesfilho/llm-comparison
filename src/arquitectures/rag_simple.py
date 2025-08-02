# src/architectures/rag_simple.py
import logging
from typing import List, Dict, Any, Optional
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

from ..config import Config

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    def __init__(self, config: Config, model_name: str = "phi-2"):
        self.config = config
        self.model_config = config.MODELS[model_name]
        
        # Detecta melhor device para M1 Mac
        self.device = self._get_optimal_device()
        logger.info(f"Inicializando RAG com device: {self.device}")
        
        # Carrega componentes
        self.embedding_model = SentenceTransformer(
            config.RAG.embedding_model,
            device=self.device
        )
        self.index = None
        self.documents = []
        
        # Inicializa modelo de geração
        self._setup_generator()
    
    def _get_optimal_device(self):
        """Detecta o melhor device disponível, priorizando MPS para M1 Mac"""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _setup_generator(self):
        """Configura o modelo gerador otimizado para M1"""
        try:
            logger.info(f"Carregando modelo: {self.model_config.model_id}")
            
            # Carrega tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configuração de dtype baseada no device
            if self.device == "mps":
                # M1 Mac funciona melhor com float16
                torch_dtype = torch.float16
                device_map = None  # MPS não suporta device_map="auto"
            elif self.device == "cuda":
                torch_dtype = torch.float16
                device_map = "auto"
            else:
                torch_dtype = torch.float32
                device_map = None
            
            # Carrega modelo
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move para device se necessário
            if self.device == "mps":
                self.model = self.model.to(self.device)
            
            # Configura pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device in ["cuda", "mps"] else -1,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            logger.info(f"✅ Modelo {self.model_config.name} carregado com sucesso no {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo {self.model_config.model_id}: {e}")
            self._setup_fallback_generator()
    
    def _setup_fallback_generator(self):
        """Configura modelo fallback para casos de erro"""
        logger.info("Configurando modelo fallback...")
        try:
            fallback_model = "microsoft/DialoGPT-small"
            self.generator = pipeline(
                "text-generation",
                model=fallback_model,
                device=-1,  # CPU apenas
                torch_dtype=torch.float32
            )
            logger.info(f"✅ Modelo fallback configurado: {fallback_model}")
        except Exception as e:
            logger.error(f"❌ Erro até no fallback: {e}")
            # Último recurso: usar modelo dummy
            self.generator = None
    
    def load_index(self, index_dir: Path):
        """Carrega índice FAISS e documentos"""
        try:
            # Carrega índice FAISS
            index_file = index_dir / "index.faiss"
            if not index_file.exists():
                raise FileNotFoundError(f"Índice não encontrado: {index_file}")
            
            self.index = faiss.read_index(str(index_file))
            
            # Carrega documentos
            docs_file = index_dir / "documents.json"
            if not docs_file.exists():
                raise FileNotFoundError(f"Documentos não encontrados: {docs_file}")
            
            self.documents = pd.read_json(docs_file, orient='records').to_dict('records')
            
            logger.info(f"✅ Índice carregado: {self.index.ntotal} documentos")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar índice: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Recupera documentos relevantes usando busca vetorial"""
        if self.index is None:
            raise ValueError("Índice não carregado. Execute load_index() primeiro.")
        
        if top_k is None:
            top_k = self.config.RAG.top_k
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode([query])
            
            # Busca no índice FAISS
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Recupera documentos correspondentes
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Índice válido
                    doc = self.documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    retrieved_docs.append(doc)
            
            logger.debug(f"Recuperados {len(retrieved_docs)} documentos para query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"❌ Erro na recuperação: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Gera resposta baseada nos documentos recuperados"""
        if self.generator is None:
            return "❌ Modelo de geração não disponível. Verifique a configuração."
        
        if not retrieved_docs:
            return "❌ Nenhum documento relevante encontrado para sua pergunta."
        
        try:
            # Constrói contexto com os documentos mais relevantes
            context_docs = retrieved_docs[:3]  # Top 3 documentos
            context = "\n\n".join([
                f"Documento {i+1}:\n{doc['content'][:500]}..." if len(doc['content']) > 500 else f"Documento {i+1}:\n{doc['content']}"
                for i, doc in enumerate(context_docs)
            ])
            
            # Template de prompt otimizado
            prompt = f"""Com base nos documentos fornecidos, responda a pergunta de forma clara e precisa.

Contexto:
{context}

Pergunta: {query}

Resposta detalhada:"""
            
            # Parâmetros de geração otimizados para M1
            generation_params = {
                "max_new_tokens": 200,
                "temperature": self.model_config.temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None
            }
            
            # Gera resposta
            response = self.generator(prompt, **generation_params)[0]['generated_text']
            
            # Extrai apenas a parte da resposta
            if "Resposta detalhada:" in response:
                response = response.split("Resposta detalhada:")[-1].strip()
            elif "Resposta:" in response:
                response = response.split("Resposta:")[-1].strip()
            else:
                # Se não encontrou marcador, pega depois do prompt
                response = response[len(prompt):].strip()
            
            # Limpa resposta
            response = response.replace(self.tokenizer.eos_token if hasattr(self, 'tokenizer') else "", "").strip()
            
            if not response or len(response) < 10:
                return "❌ Não consegui gerar uma respuesta adequada. Tente reformular sua pergunta."
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erro na geração: {e}")
            return f"❌ Erro ao processar sua pergunta: {str(e)}"
    
    def query(self, question: str) -> Dict[str, Any]:
        """Interface principal para consultas RAG"""
        try:
            # 1. Recupera documentos relevantes
            retrieved_docs = self.retrieve(question)
            
            # 2. Gera resposta
            response = self.generate_response(question, retrieved_docs)
            
            # 3. Retorna resultado estruturado
            return {
                'question': question,
                'answer': response,
                'retrieved_documents': retrieved_docs,
                'metadata': {
                    'num_retrieved': len(retrieved_docs),
                    'model': self.model_config.name,
                    'device': self.device,
                    'embedding_model': self.config.RAG.embedding_model
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta: {e}")
            return {
                'question': question,
                'answer': f"❌ Erro ao processar consulta: {str(e)}",
                'retrieved_documents': [],
                'metadata': {
                    'num_retrieved': 0,
                    'model': self.model_config.name,
                    'device': self.device,
                    'error': str(e)
                }
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Processa múltiplas consultas em lote"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retorna informações do sistema"""
        return {
            'device': self.device,
            'model_name': self.model_config.name,
            'model_id': self.model_config.model_id,
            'embedding_model': self.config.RAG.embedding_model,
            'index_loaded': self.index is not None,
            'num_documents': len(self.documents) if self.documents else 0,
            'torch_version': torch.__version__,
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        }
