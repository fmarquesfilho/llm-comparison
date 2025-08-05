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
            
            if similarity >= similarity_threshold:
                similar_events.append((event, similarity))
        
        # Ordena por similaridade
        similar_events.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Found {len(similar_events)} similar events to {reference_event.event_type}")
        return similar_events
    
    def generate_summary_report(self, query: str, retrieved_events: List[Tuple[DynamicEventUnit, float]]) -> str:
        """
        Gera relatório baseado nos eventos recuperados.
        Versão simplificada comparada ao Time-CoT do DyG-RAG.
        """
        if not retrieved_events:
            return "Não foram encontrados eventos relevantes para a consulta."
        
        events = [event for event, _ in retrieved_events]
        
        # Estatísticas básicas
        total_events = len(events)
        avg_loudness = np.mean([e.loudness for e in events])
        peak_event = max(events, key=lambda e: e.loudness)
        
        # Violações
        violations = [e for e in events if e.violates_noise_regulations()]
        
        # Tipos de eventos
        event_types = [e.event_type for e in events]
        type_counts = {t: event_types.count(t) for t in set(event_types)}
        most_frequent = max(type_counts.items(), key=lambda x: x[1]) if type_counts else ("N/A", 0)
        
        # Monta resposta
        report_parts = [
            f"## Análise de {total_events} Eventos Sonoros\n",
            f"**Intensidade média:** {avg_loudness:.1f} dB",
            f"**Pico de ruído:** {peak_event.loudness:.1f} dB ({peak_event.event_type} em {peak_event.timestamp.strftime('%d/%m %H:%M')})",
            f"**Evento mais frequente:** {most_frequent[0]} ({most_frequent[1]} ocorrências)"
        ]
        
        if violations:
            report_parts.append(f"\n⚠️ **{len(violations)} violações** de normas detectadas:")
            for v in violations[:3]:  # Mostra apenas as 3 primeiras
                report_parts.append(f"- {v.event_type} ({v.loudness:.1f} dB) em {v.timestamp.strftime('%d/%m %H:%M')}")
        
        if violations or avg_loudness > 75:
            report_parts.append("\n**Recomendação:** Revisar procedimentos operacionais e implementar controles acústicos.")
        
        return "\n".join(report_parts)
    
    def get_noise_pattern_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analisa padrões de ruído nas últimas X horas.
        """
        if not self.events:
            return {"error": "No events available"}
        
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        if not recent_events:
            return {"error": f"No events in last {hours} hours"}
        
        # Análise por hora
        hourly_stats = {}
        for event in recent_events:
            hour = event.timestamp.hour
            if hour not in hourly_stats:
                hourly_stats[hour] = {"count": 0, "avg_loudness": [], "violations": 0}
            
            hourly_stats[hour]["count"] += 1
            hourly_stats[hour]["avg_loudness"].append(event.loudness)
            if event.violates_noise_regulations():
                hourly_stats[hour]["violations"] += 1
        
        # Processa médias
        for hour_data in hourly_stats.values():
            hour_data["avg_loudness"] = np.mean(hour_data["avg_loudness"])
        
        return {
            "total_events": len(recent_events),
            "timespan_hours": hours,
            "hourly_breakdown": hourly_stats,
            "peak_hour": max(hourly_stats.items(), key=lambda x: x[1]["avg_loudness"])[0] if hourly_stats else None,
            "total_violations": sum(stats["violations"] for stats in hourly_stats.values())
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema para monitoramento"""
        if not self.events:
            return {"total_events": 0, "status": "empty"}
        
        return {
            "total_events": len(self.events),
            "time_range": {
                "start": self._min_time.isoformat() if self._min_time else None,
                "end": self._max_time.isoformat() if self._max_time else None
            },
            "embedding_dimensions": {
                "text": len(self.text_embeddings[0]) if self.text_embeddings else 0,
                "temporal": len(self.temporal_embeddings[0]) if self.temporal_embeddings else 0,
                "combined": len(self.combined_embeddings[0]) if self.combined_embeddings else 0
            },
            "cache_size": len(self._embedding_cache),
            "time_weight": self.time_weight,
            "avg_loudness": np.mean([e.loudness for e in self.events]),
            "violation_rate": sum(1 for e in self.events if e.violates_noise_regulations()) / len(self.events)
        }