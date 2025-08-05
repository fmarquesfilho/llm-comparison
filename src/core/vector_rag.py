import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from .events import DynamicEventUnit
from ..utils.embeddings import cosine_similarity, get_embedding_model
from ..utils.time_utils import FourierTimeEncoder, normalize_timestamp
from ..config import settings

logger = logging.getLogger(__name__)

class TemporalVectorRAG:
    """
    Implementação de RAG vetorial com consciência temporal.
    Sistema baseline para comparação com o DyG-RAG.
    """
    
    def __init__(self, embedding_model_name: str = None, time_weight: float = 0.3, time_dim: int = 64):
        """
        Inicializa o sistema TemporalVectorRAG.
        
        Args:
            embedding_model_name: Nome do modelo de embeddings
            time_weight: Peso da componente temporal na busca (0-1)
            time_dim: Dimensionalidade do encoding temporal
        """
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.time_weight = time_weight
        self.time_encoder = FourierTimeEncoder(dim=time_dim)
        
        # Armazenamento dos eventos e embeddings
        self.events: List[DynamicEventUnit] = []
        self.text_embeddings: List[np.ndarray] = []
        self.temporal_embeddings: List[np.ndarray] = []
        self.combined_embeddings: List[np.ndarray] = []
        
        # Cache para otimização
        self._embedding_cache = {}
        self._min_time = None
        self._max_time = None
        
        logger.info(f"Initialized TemporalVectorRAG with time_weight={time_weight}")
    
    def add_event(self, event: DynamicEventUnit):
        """
        Adiciona um evento ao índice vetorial.
        Gera embeddings semântico e temporal.
        """
        try:
            # Atualiza limites temporais
            if self._min_time is None or event.timestamp < self._min_time:
                self._min_time = event.timestamp
            if self._max_time is None or event.timestamp > self._max_time:
                self._max_time = event.timestamp
            
            # Gera embedding semântico
            text_embedding = self._get_text_embedding(event)
            
            # Gera embedding temporal
            temporal_embedding = self.time_encoder.encode(event.timestamp)
            
            # Combina embeddings
            combined_embedding = np.concatenate([
                (1 - self.time_weight) * text_embedding,
                self.time_weight * temporal_embedding
            ])
            
            # Armazena tudo
            self.events.append(event)
            self.text_embeddings.append(text_embedding)
            self.temporal_embeddings.append(temporal_embedding)
            self.combined_embeddings.append(combined_embedding)
            
            logger.debug(f"Added event {event.event_id} to vector index")
            
        except Exception as e:
            logger.error(f"Error adding event {event.event_id}: {str(e)}")
            raise
    
    def _get_text_embedding(self, event: DynamicEventUnit) -> np.ndarray:
        """Gera ou recupera embedding semântico do cache"""
        text = event.to_embedding_text()
        
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.embedding_model.encode(text)
        
        return self._embedding_cache[text]
    
    def retrieve(self, query: str, query_time: datetime = None, top_k: int = 5, 
                temporal_radius: Optional[int] = None) -> List[Tuple[DynamicEventUnit, float]]:
        """
        Recupera eventos mais relevantes para a consulta.
        
        Args:
            query: Consulta em linguagem natural
            query_time: Timestamp da consulta para busca temporal
            top_k: Número de eventos a retornar
            temporal_radius: Raio temporal em segundos (None = sem filtro)
        
        Returns:
            Lista de tuplas (evento, score de similaridade)
        """
        if not self.events:
            logger.warning("No events in index for retrieval")
            return []
        
        try:
            # Gera embedding da consulta
            query_text_embedding = self.embedding_model.encode(query)
            
            # Preparar embedding temporal da consulta se fornecido
            if query_time:
                query_temporal_embedding = self.time_encoder.encode(query_time)
                query_combined = np.concatenate([
                    (1 - self.time_weight) * query_text_embedding,
                    self.time_weight * query_temporal_embedding
                ])
            else:
                # Sem componente temporal
                query_combined = query_text_embedding
            
            # Calcula similaridades
            similarities = []
            for i, event in enumerate(self.events):
                
                # Filtro temporal se especificado
                if temporal_radius and query_time:
                    time_diff = abs((event.timestamp - query_time).total_seconds())
                    if time_diff > temporal_radius:
                        continue
                
                # Calcula similaridade
                if query_time:
                    # Usa embedding combinado
                    similarity = cosine_similarity(query_combined, self.combined_embeddings[i])
                else:
                    # Usa apenas embedding semântico
                    similarity = cosine_similarity(query_text_embedding, self.text_embeddings[i])
                
                similarities.append((event, similarity, i))
            
            # Ordena por similaridade
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Retorna top_k resultados
            results = [(event, score) for event, score, _ in similarities[:top_k]]
            
            logger.info(f"Retrieved {len(results)} events for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def temporal_range_search(self, start_time: datetime, end_time: datetime, 
                            event_types: Optional[List[str]] = None) -> List[DynamicEventUnit]:
        """
        Busca eventos em um intervalo temporal específico.
        
        Args:
            start_time: Início do intervalo
            end_time: Fim do intervalo
            event_types: Tipos de eventos para filtrar (opcional)
        
        Returns:
            Lista de eventos no intervalo
        """
        filtered_events = []
        
        for event in self.events:
            # Filtro temporal
            if not (start_time <= event.timestamp <= end_time):
                continue
            
            # Filtro por tipo se especificado
            if event_types and event.event_type not in event_types:
                continue
            
            filtered_events.append(event)
        
        # Ordena cronologicamente
        filtered_events.sort(key=lambda e: e.timestamp)
        
        logger.info(f"Found {len(filtered_events)} events in range {start_time} - {end_time}")
        return filtered_events
    
    def find_similar_events(self, reference_event: DynamicEventUnit, 
                          similarity_threshold: float = 0.7, 
                          exclude_self: bool = True) -> List[Tuple[DynamicEventUnit, float]]:
        """
        Encontra eventos similares a um evento de referência.
        
        Args:
            reference_event: Evento de referência
            similarity_threshold: Threshold de similaridade
            exclude_self: Se deve excluir o próprio evento
        
        Returns:
            Lista de eventos similares com scores
        """
        ref_embedding = self._get_text_embedding(reference_event)
        similar_events = []
        
        for i, event in enumerate(self.events):
            if exclude_self and event.event_id == reference_event.event_id:
                continue
            
            similarity = cosine_similarity(ref_embedding, self.text_embeddings[i])
            
            if