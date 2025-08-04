from datetime import datetime
from typing import Optional
from ..core.router import QueryRouter
from ..core.temporal_rag import TemporalGraphRAG
from ..core.vector_rag import TemporalVectorRAG
from ..config import settings

class QueryProcessingPipeline:
    def __init__(self, llm, embedding_model):
        self.vector_rag = TemporalVectorRAG(embedding_model)
        self.graph_rag = TemporalGraphRAG(embedding_model)
        self.router = QueryRouter(llm, self.vector_rag, self.graph_rag)
    
    def process_query(self, query: str, query_time: Optional[datetime] = None):
        """Processa uma consulta do usuário"""
        return self.router.answer(query, query_time)
    
    def generate_report(self, start_time: datetime, end_time: datetime):
        """Gera um relatório temporal"""
        query = f"""
        Analyze noise patterns between {start_time} and {end_time}:
        1. Most frequent events
        2. Peak periods
        3. Potential norm violations
        4. Recommendations
        """
        return self.process_query(query, end_time)