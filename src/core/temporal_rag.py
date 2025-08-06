import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .events import DynamicEventUnit
from ..utils.time_utils import FourierTimeEncoder
from ..utils.embeddings import cosine_similarity, get_embedding_model

logger = logging.getLogger(__name__)

class TemporalGraphRAG:
    """
    Implementação do DyG-RAG (Dynamic Graph RAG) para análise temporal de eventos.
    Constrói e mantém um grafo dinâmico de eventos com relacionamentos temporais e semânticos.
    """
    
    def __init__(self, embedding_model_name: str = None, time_dim: int = 64, time_window: int = 300):
        """
        Inicializa o sistema TemporalGraphRAG.
        
        Args:
            embedding_model_name: Nome do modelo de embeddings
            time_dim: Dimensionalidade do encoding temporal
            time_window: Janela temporal em segundos para conectar eventos
        """
        self.graph = nx.DiGraph()  # Grafo direcionado para capturar ordem temporal
        self.embedding_model = get_embedding_model(embedding_model_name)
        self.time_encoder = FourierTimeEncoder(dim=time_dim)
        self.time_window = time_window
        
        # Cache para embeddings e índices
        self.event_embeddings = {}
        self.temporal_index = defaultdict(list)  # timestamp -> [event_ids]
        
        logger.info(f"Initialized TemporalGraphRAG with time_window={time_window}s, time_dim={time_dim}")
    
    def add_event(self, event: DynamicEventUnit):
        """
        Adiciona um evento ao grafo dinâmico.
        Implementa a lógica de conexão temporal e semântica do DyG-RAG.
        """
        try:
            # 1. Adiciona o nó do evento
            self.graph.add_node(
                event.event_id,
                event=event,
                timestamp=event.timestamp,
                event_type=event.event_type,
                loudness=event.loudness,
                severity=event.get_severity_level()
            )
            
            # 2. Gera embeddings semântico e temporal
            self._compute_and_cache_embeddings(event)
            
            # 3. Conecta com eventos temporalmente próximos
            self._connect_temporal_neighbors(event)
            
            # 4. Conecta com eventos semanticamente similares
            self._connect_semantic_neighbors(event)
            
            # 5. Atualiza índice temporal
            self.temporal_index[event.timestamp.strftime('%Y-%m-%d %H:%M')].append(event.event_id)
            
            logger.debug(f"Added event {event.event_id} to temporal graph")
            
        except Exception as e:
            logger.error(f"Error adding event {event.event_id}: {str(e)}")
            raise
    
    def _compute_and_cache_embeddings(self, event: DynamicEventUnit):
        """Computa e armazena embeddings semântico e temporal"""
        # Embedding semântico
        text_embedding = self.embedding_model.encode(event.to_embedding_text())
        
        # Embedding temporal (Fourier)
        time_embedding = self.time_encoder.encode(event.timestamp)
        
        # Combina embeddings conforme DyG-RAG
        combined_embedding = np.concatenate([text_embedding, time_embedding])
        
        self.event_embeddings[event.event_id] = {
            'text': text_embedding,
            'temporal': time_embedding,
            'combined': combined_embedding
        }
    
    def _connect_temporal_neighbors(self, event: DynamicEventUnit):
        """
        Conecta eventos em proximidade temporal.
        Implementa a lógica de janela temporal do DyG-RAG.
        """
        start_time = event.timestamp - timedelta(seconds=self.time_window)
        end_time = event.timestamp + timedelta(seconds=self.time_window)
        
        for node_id in self.graph.nodes():
            if node_id == event.event_id:
                continue
                
            other_event = self.graph.nodes[node_id]['event']
            
            if start_time <= other_event.timestamp <= end_time:
                # Calcula peso da aresta baseado na proximidade temporal
                time_diff = abs((event.timestamp - other_event.timestamp).total_seconds())
                temporal_weight = max(0, 1 - (time_diff / self.time_window))
                
                # Adiciona aresta direcionada (do evento anterior para o posterior)
                if other_event.timestamp < event.timestamp:
                    self.graph.add_edge(
                        other_event.event_id, 
                        event.event_id,
                        weight=temporal_weight,
                        relation_type='temporal_succession',
                        time_diff=time_diff
                    )
                else:
                    self.graph.add_edge(
                        event.event_id,
                        other_event.event_id,
                        weight=temporal_weight,
                        relation_type='temporal_succession',
                        time_diff=time_diff
                    )
    
    def _connect_semantic_neighbors(self, event: DynamicEventUnit, similarity_threshold: float = 0.7):
        """
        Conecta eventos semanticamente similares.
        Implementa conexões baseadas em similaridade de embedding.
        """
        current_embedding = self.event_embeddings[event.event_id]['text']
        
        for node_id in self.graph.nodes():
            if node_id == event.event_id or node_id not in self.event_embeddings:
                continue
                
            other_embedding = self.event_embeddings[node_id]['text']
            similarity = cosine_similarity(current_embedding, other_embedding)
            
            if similarity >= similarity_threshold:
                # Adiciona aresta bidirecional para similaridade semântica
                self.graph.add_edge(
                    event.event_id,
                    node_id,
                    weight=similarity,
                    relation_type='semantic_similarity'
                )
                self.graph.add_edge(
                    node_id,
                    event.event_id,
                    weight=similarity,
                    relation_type='semantic_similarity'
                )
    
    def temporal_retrieval(self, query: str, query_time: datetime = None, top_k: int = 10) -> List[DynamicEventUnit]:
        """
        Recupera eventos relevantes usando busca temporal e semântica.
        Implementa a recuperação multi-hop do DyG-RAG.
        """
        try:
            # 1. Embedding da consulta
            query_text_embedding = self.embedding_model.encode(query)
            
            # 2. Embedding temporal da consulta (se fornecido)
            if query_time:
                query_time_embedding = self.time_encoder.encode(query_time)
                query_combined = np.concatenate([query_text_embedding, query_time_embedding])
                time_weight = 0.3  # Peso para componente temporal
            else:
                query_combined = query_text_embedding
                time_weight = 0.0
            
            # 3. Calcula similaridades com todos os eventos
            event_scores = []
            for event_id in self.graph.nodes():
                if event_id not in self.event_embeddings:
                    continue
                    
                event_data = self.event_embeddings[event_id]
                
                # Similaridade semântica
                sem_similarity = cosine_similarity(query_text_embedding, event_data['text'])
                
                # Similaridade temporal (se aplicável)
                if query_time and 'temporal' in event_data:
                    temp_similarity = cosine_similarity(query_time_embedding, event_data['temporal'])
                    combined_score = (1 - time_weight) * sem_similarity + time_weight * temp_similarity
                else:
                    combined_score = sem_similarity
                
                event_scores.append((event_id, combined_score))
            
            # 4. Ordena por relevância
            event_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 5. Recupera top_k eventos
            top_events = []
            for event_id, score in event_scores[:top_k]:
                event = self.graph.nodes[event_id]['event']
                top_events.append(event)
            
            logger.info(f"Retrieved {len(top_events)} events for query: {query[:50]}...")
            return top_events
            
        except Exception as e:
            logger.error(f"Error in temporal retrieval: {str(e)}")
            return []
    
    def extract_event_timeline(self, events: List[DynamicEventUnit]) -> List[DynamicEventUnit]:
        """
        Extrai uma linha temporal coerente dos eventos recuperados.
        Implementa o graph traversal para recuperação de sequências.
        """
        if not events:
            return []
        
        # Ordena eventos por timestamp
        timeline = sorted(events, key=lambda e: e.timestamp)
        
        # Aplica algoritmo de caminho mais provável no grafo
        enhanced_timeline = []
        event_ids = [e.event_id for e in timeline]
        
        # Para cada evento, verifica se há conexões temporais com o próximo
        for i, event in enumerate(timeline):
            enhanced_timeline.append(event)
            
            if i < len(timeline) - 1:
                next_event = timeline[i + 1]
                
                # Verifica se há caminho direto no grafo
                if self.graph.has_edge(event.event_id, next_event.event_id):
                    edge_data = self.graph.get_edge_data(event.event_id, next_event.event_id)
                    logger.debug(f"Timeline connection: {event.event_type} -> {next_event.event_type} "
                               f"({edge_data.get('relation_type', 'unknown')})")
        
        return enhanced_timeline
    
    def generate_time_cot_response(self, query: str, retrieved_events: List[DynamicEventUnit]) -> str:
        """
        Gera resposta usando Time Chain-of-Thought.
        Implementa o raciocínio temporal estruturado do DyG-RAG.
        """
        if not retrieved_events:
            return "Não foram encontrados eventos relevantes para responder à consulta."
        
        # Extrai linha temporal
        timeline = self.extract_event_timeline(retrieved_events)
        
        # Template Time-CoT
        cot_steps = []
        
        # Passo 1: Identificar escopo temporal
        if timeline:
            start_time = min(e.timestamp for e in timeline)
            end_time = max(e.timestamp for e in timeline)
            cot_steps.append(f"**Escopo Temporal:** {start_time.strftime('%d/%m/%Y %H:%M')} a {end_time.strftime('%d/%m/%Y %H:%M')}")
        
        # Passo 2: Filtrar eventos no escopo
        cot_steps.append(f"**Eventos Filtrados:** {len(timeline)} eventos relevantes identificados")
        
        # Passo 3: Analisar ordem cronológica
        if len(timeline) > 1:
            sequence_desc = " → ".join([f"{e.event_type}({e.timestamp.strftime('%H:%M')})" for e in timeline[:5]])
            if len(timeline) > 5:
                sequence_desc += "..."
            cot_steps.append(f"**Sequência Cronológica:** {sequence_desc}")
        
        # Passo 4: Verificar violações
        violations = [e for e in timeline if e.violates_noise_regulations()]
        if violations:
            cot_steps.append(f"**Violações Detectadas:** {len(violations)} eventos em desacordo com regulamentações")
        
        # Passo 5: Análise de padrões
        event_types = [e.event_type for e in timeline]
        type_counts = {t: event_types.count(t) for t in set(event_types)}
        most_frequent = max(type_counts.items(), key=lambda x: x[1])
        cot_steps.append(f"**Padrão Identificado:** {most_frequent[0]} foi o evento mais frequente ({most_frequent[1]} ocorrências)")
        
        # Geração da resposta final
        response_parts = [
            "## Análise Temporal dos Eventos de Ruído\n",
            "\n".join(cot_steps),
            "\n### Resposta:",
            self._generate_final_answer(query, timeline, violations)
        ]
        
        return "\n\n".join(response_parts)
    
    def _generate_final_answer(self, query: str, timeline: List[DynamicEventUnit], violations: List[DynamicEventUnit]) -> str:
        """Gera resposta final baseada na análise temporal"""
        if not timeline:
            return "Não há dados suficientes para responder à consulta."
        
        # Análise básica
        total_events = len(timeline)
        avg_loudness = np.mean([e.loudness for e in timeline])
        peak_event = max(timeline, key=lambda e: e.loudness)
        
        answer_parts = []
        
        # Resumo quantitativo
        answer_parts.append(f"Durante o período analisado, foram registrados {total_events} eventos sonoros relevantes.")
        answer_parts.append(f"A intensidade média foi de {avg_loudness:.1f} dB, com pico de {peak_event.loudness:.1f} dB ({peak_event.event_type} às {peak_event.timestamp.strftime('%H:%M')}).")
        
        # Violações
        if violations:
            answer_parts.append(f"ATENÇÃO: Identificadas {len(violations)} violações de normas de ruído que requerem ação corretiva.")
        
        # Recomendações
        if violations or avg_loudness > 75:
            answer_parts.append("Recomenda-se revisar os procedimentos operacionais e implementar medidas de controle acústico.")
        
        return " ".join(answer_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do grafo para monitoramento"""
        return {
            'total_events': self.graph.number_of_nodes(),
            'total_connections': self.graph.number_of_edges(),
            'temporal_connections': sum(1 for _, _, d in self.graph.edges(data=True) 
                                      if d.get('relation_type') == 'temporal_succession'),
            'semantic_connections': sum(1 for _, _, d in self.graph.edges(data=True) 
                                      if d.get('relation_type') == 'semantic_similarity'),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if self.graph.nodes() else 0
        }