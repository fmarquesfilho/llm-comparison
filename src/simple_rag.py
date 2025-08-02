# src/simple_rag.py
import logging
import time
from typing import List, Dict, Any
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    """Sistema RAG simplificado para MVP - foco em embeddings + retrieval"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Detecta melhor device para M1 Mac
        self.device = self._get_optimal_device()
        logger.info(f"Inicializando RAG com device: {self.device}")
        
        # Carrega apenas modelo de embedding (sem geração para MVP)
        self.embedding_model = SentenceTransformer(
            self.config.RAG.embedding_model,
            device=self.device
        )
        
        self.index = None
        self.documents = []
        
        logger.info("✅ RAG System inicializado (modo embedding apenas)")
    
    def _get_optimal_device(self):
        """Detecta o melhor device disponível, priorizando MPS para M1 Mac"""
        try:
            import torch
            
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        except:
            return "cpu"
    
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
            start_time = time.time()
            
            # Encode query 
            query_embedding = self.embedding_model.encode([query])
            
            # Busca no índice FAISS
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            retrieval_time = time.time() - start_time
            
            # Recupera documentos correspondentes
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # Índice válido
                    doc = self.documents[idx].copy()
                    # FAISS retorna distância L2, converter para similaridade
                    doc['relevance_score'] = 1.0 / (1.0 + float(score))
                    retrieved_docs.append(doc)
            
            logger.debug(f"Recuperados {len(retrieved_docs)} documentos em {retrieval_time:.3f}s")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"❌ Erro na recuperação: {e}")
            return []
    
    def generate_simple_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Gera resposta simples baseada nos documentos recuperados (sem LLM)"""
        
        if not retrieved_docs:
            return "❌ Nenhum documento relevante encontrado para sua pergunta."
        
        # Para o MVP, retorna conteúdo do documento mais relevante
        best_doc = retrieved_docs[0]
        content = best_doc.get('content', '')
        
        # Resposta estruturada simples
        response = f"Baseado no documento mais relevante:\n\n{content}"
        
        # Adiciona informação de contexto se disponível
        if len(retrieved_docs) > 1:
            response += f"\n\n(Encontrados {len(retrieved_docs)} documentos relacionados)"
        
        return response
    
    def query(self, question: str) -> Dict[str, Any]:
        """Interface principal para consultas RAG"""
        try:
            start_time = time.time()
            
            # 1. Recupera documentos relevantes
            retrieved_docs = self.retrieve(question)
            
            # 2. Gera resposta simples
            response = self.generate_simple_response(question, retrieved_docs)
            
            total_time = time.time() - start_time
            
            # 3. Retorna resultado estruturado
            return {
                'question': question,
                'answer': response,
                'retrieved_documents': retrieved_docs,
                'response_time': total_time,
                'relevance_score': retrieved_docs[0]['relevance_score'] if retrieved_docs else 0,
                'retrieved_docs': len(retrieved_docs),
                'metadata': {
                    'num_retrieved': len(retrieved_docs),
                    'device': self.device,
                    'embedding_model': self.config.RAG.embedding_model,
                    'mode': 'simple_retrieval'
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta: {e}")
            return {
                'question': question,
                'answer': f"❌ Erro ao processar consulta: {str(e)}",
                'retrieved_documents': [],
                'response_time': 0,
                'relevance_score': 0,
                'retrieved_docs': 0,
                'metadata': {
                    'num_retrieved': 0,
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
        try:
            import torch
            torch_version = torch.__version__
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        except:
            torch_version = "N/A"
            mps_available = False
        
        return {
            'device': self.device,
            'embedding_model': self.config.RAG.embedding_model,
            'index_loaded': self.index is not None,
            'num_documents': len(self.documents) if self.documents else 0,
            'torch_version': torch_version,
            'mps_available': mps_available,
            'mode': 'simple_rag_mvp'
        }


def test_simple_rag():
    """Função de teste para RAG simples"""
    logger.info("🧪 Testando RAG simples...")
    
    try:
        # Inicializa sistema
        config = Config()
        rag_system = SimpleRAGSystem(config)
        
        # Carrega índice
        embeddings_dir = Path("data/embeddings")
        if not embeddings_dir.exists():
            logger.error("❌ Diretório de embeddings não encontrado")
            return []
        
        rag_system.load_index(embeddings_dir)
        
        # Carrega perguntas de teste
        questions_file = Path("data/evaluation/test_questions.json")
        if not questions_file.exists():
            logger.error("❌ Perguntas de teste não encontradas")
            return []
        
        import json
        with open(questions_file, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        # Executa testes
        results = []
        for q in test_questions:
            result = rag_system.query(q['question'])
            
            # Adiciona informações da pergunta
            result.update({
                'question_id': q['id'],
                'expected_concepts': q['expected_concepts'],
                'category': q['category']
            })
            
            results.append(result)
            
            logger.info(f"RAG {q['id']}: {result['response_time']:.2f}s, "
                       f"relevância: {result['relevance_score']:.3f}")
        
        # Salva resultados
        results_file = Path("data/evaluation/rag_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ RAG testado, resultados em {results_file}")
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no teste RAG: {e}")
        return []


if __name__ == "__main__":
    # Teste standalone
    logging.basicConfig(level=logging.INFO)
    test_simple_rag()
