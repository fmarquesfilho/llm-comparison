from typing import Union
from datetime import datetime
from ..utils.embeddings import cosine_similarity
from ..config import settings
from .events import DynamicEventUnit
from .temporal_rag import TemporalGraphRAG
from .vector_rag import TemporalVectorRAG

class QueryRouter:
    def __init__(self, llm, vector_rag: TemporalVectorRAG, graph_rag: TemporalGraphRAG):
        self.llm = llm
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        
    def route(self, query: str) -> str:
        """Determina a abordagem mais adequada para a consulta"""
        complexity = self._assess_complexity(query)
        return 'graph' if complexity == 'complex' else 'vector'
    
    def answer(self, query: str, query_time: datetime = None) -> str:
        """Roteia e responde à consulta"""
        approach = self.route(query)
        
        if approach == 'vector':
            results = self.vector_rag.retrieve(query, query_time)
            return self._format_vector_response(query, results)
        else:
            results = self.graph_rag.temporal_retrieval(query, query_time)
            return self.graph_rag.generate_time_cot_response(query, results)
    
    def _assess_complexity(self, query: str) -> str:
        """Classifica a consulta como simples ou complexa"""
        simple_keywords = ['occurred', 'happened', 'was there', 'any']
        complex_keywords = ['why', 'how', 'pattern', 'trend', 'cause', 'effect']
        
        if any(kw in query.lower() for kw in complex_keywords):
            return 'complex'
        return 'simple'
    
    def _format_vector_response(self, query: str, results) -> str:
        """Formata resposta para consultas simples"""
        events_str = "\n".join(
            f"- {e.timestamp}: {e.event_type} ({e.loudness}dB)"
            for e, _ in results
        )
        return f"Based on the following events:\n{events_str}\n\nAnswer: {query}"