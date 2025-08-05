"""
Integração com Kùzu Graph Database para o Cenário B (Hybrid RAG)
Implementa o backend de grafo usando Kùzu conforme especificado no README
"""

import kuzu
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json

from src.core.events import DynamicEventUnit
from src.utils.time_utils import FourierTimeEncoder

logger = logging.getLogger(__name__)

class KuzuTemporalGraphRAG:
    """
    Implementação do DyG-RAG usando Kùzu Graph Database.
    Substitui NetworkX por uma solução de grafo mais robusta e escalável.
    """
    
    def __init__(self, db_path: str = "data/kuzu_graph", embedding_model=None, time_dim: int = 64):
        """
        Inicializa a conexão com Kùzu e cria o schema do grafo temporal.
        
        Args:
            db_path: Caminho para o banco de dados Kùzu
            embedding_model: Modelo para embeddings semânticos
            time_dim: Dimensionalidade do encoding temporal
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Conexão com Kùzu
        self.database = kuzu.Database(str(self.db_path))
        self.connection = kuzu.Connection(self.database)
        
        self.embedding_model = embedding_model
        self.time_encoder = FourierTimeEncoder(dim=time_dim)
        
        # Inicializa schema do grafo
        self._create_schema()
        
        logger.info(f"Initialized KuzuTemporalGraphRAG with database at {db_path}")
    
    def _create_schema(self):
        """Cria o schema do grafo temporal no Kùzu"""
        try:
            # Tabela de nós: Eventos Dinâmicos
            self.connection.execute("""
                CREATE NODE TABLE IF NOT EXISTS Event(
                    event_id STRING,
                    timestamp TIMESTAMP,
                    event_type STRING,
                    loudness DOUBLE,
                    sensor_id STRING,
                    description STRING,
                    severity_level STRING,
                    hour INT32,
                    day_of_week INT32,
                    is_weekend BOOLEAN,
                    is_work_hours BOOLEAN,
                    is_noise_restricted BOOLEAN,
                    violates_regulations BOOLEAN,
                    embedding_text STRING,
                    metadata_json STRING,
                    PRIMARY KEY(event_id)
                )
            """)
            
            # Tabela de arestas: Relacionamentos Temporais
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS TemporalRelation(
                    FROM Event TO Event,
                    relation_type STRING,
                    weight DOUBLE,
                    time_diff_seconds DOUBLE,
                    semantic_similarity DOUBLE
                )
            """)
            
            # Tabela de arestas: Relacionamentos Semânticos
            self.connection.execute("""
                CREATE REL TABLE IF NOT EXISTS SemanticRelation(
                    FROM Event TO Event,
                    similarity_score DOUBLE,
                    shared_context STRING
                )
            """)
            
            logger.info("Kùzu schema created successfully")
            
        except Exception as e:
            logger.error(f"Error creating Kùzu schema: {str(e)}")
            raise
    
    def add_event(self, event: DynamicEventUnit):
        """
        Adiciona um evento ao grafo Kùzu.
        Implementa a lógica de DEU (Dynamic Event Unit) do DyG-RAG.
        """
        try:
            # Extrai contexto temporal
            temporal_context = event.to_temporal_context()
            
            # Prepara dados para inserção
            event_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event.event_type,
                'loudness': event.loudness,
                'sensor_id': event.sensor_id,
                'description': event.description or '',
                'severity_level': event.get_severity_level(),
                'hour': temporal_context['hour'],
                'day_of_week': temporal_context['day_of_week'],
                'is_weekend': temporal_context['is_weekend'],
                'is_work_hours': temporal_context['is_work_hours'],
                'is_noise_restricted': temporal_context['is_noise_restricted'],
                'violates_regulations': event.violates_noise_regulations(),
                'embedding_text': event.to_embedding_text(),
                'metadata_json': json.dumps(event.metadata or {})
            }
            
            # Insere o nó do evento
            self.connection.execute("""
                CREATE (e:Event {
                    event_id: $event_id,
                    timestamp: datetime($timestamp),
                    event_type: $event_type,
                    loudness: $loudness,
                    sensor_id: $sensor_id,
                    description: $description,
                    severity_level: $severity_level,
                    hour: $hour,
                    day_of_week: $day_of_week,
                    is_weekend: $is_weekend,
                    is_work_hours: $is_work_hours,
                    is_noise_restricted: $is_noise_restricted,
                    violates_regulations: $violates_regulations,
                    embedding_text: $embedding_text,
                    metadata_json: $metadata_json
                })
            """, parameters=event_data)
            
            # Cria relacionamentos temporais
            self._create_temporal_relationships(event)
            
            # Cria relacionamentos semânticos
            self._create_semantic_relationships(event)
            
            logger.debug(f"Added event {event.event_id} to Kùzu graph")
            
        except Exception as e:
            logger.error(f"Error adding event {event.event_id} to Kùzu: {str(e)}")
            raise
    
    def _create_temporal_relationships(self, event: DynamicEventUnit, time_window: int = 300):
        """
        Cria relacionamentos temporais no grafo Kùzu.
        Conecta eventos em proximidade temporal conforme DyG-RAG.
        """
        try:
            # Define janela temporal
            start_time = (event.timestamp - timedelta(seconds=time_window)).strftime('%Y-%m-%d %H:%M:%S')
            end_time = (event.timestamp + timedelta(seconds=time_window)).strftime('%Y-%m-%d %H:%M:%S')
            
            # Busca eventos na janela temporal
            result = self.connection.execute("""
                MATCH (other:Event)
                WHERE other.event_id <> $current_event_id
                  AND other.timestamp >= datetime($start_time)
                  AND other.timestamp <= datetime($end_time)
                RETURN other.event_id, other.timestamp
            """, parameters={
                'current_event_id': event.event_id,
                'start_time': start_time,
                'end_time': end_time
            })
            
            # Cria relacionamentos direcionais
            for row in result.get_next():
                other_event_id = row[0]
                other_timestamp = datetime.fromisoformat(str(row[1]))
                
                # Calcula peso temporal
                time_diff = abs((event.timestamp - other_timestamp).total_seconds())
                temporal_weight = max(0, 1 - (time_diff / time_window))
                
                # Determina direção (do anterior para o posterior)
                if other_timestamp < event.timestamp:
                    from_event, to_event = other_event_id, event.event_id
                else:
                    from_event, to_event = event.event_id, other_event_id
                
                # Cria aresta temporal
                self.connection.execute("""
                    MATCH (from:Event {event_id: $from_event})
                    MATCH (to:Event {event_id: $to_event})
                    CREATE (from)-[r:TemporalRelation {
                        relation_type: 'temporal_succession',
                        weight: $weight,
                        time_diff_seconds: $time_diff,
                        semantic_similarity: 0.0
                    }]->(to)
                """, parameters={
                    'from_event': from_event,
                    'to_event': to_event,
                    'weight': temporal_weight,
                    'time_diff': time_diff
                })
            
        except Exception as e:
            logger.error(f"Error creating temporal relationships for {event.event_id}: {str(e)}")
    
    def _create_semantic_relationships(self, event: DynamicEventUnit, similarity_threshold: float = 0.7):
        """
        Cria relacionamentos semânticos usando embeddings.
        Implementa a similaridade semântica do DyG-RAG.
        """
        if not self.embedding_model:
            return
        
        try:
            # Gera embedding do evento atual
            current_embedding = self.embedding_model.encode(event.to_embedding_text())
            
            # Busca outros eventos do mesmo tipo ou similar
            result = self.connection.execute("""
                MATCH (other:Event)
                WHERE other.event_id <> $current_event_id
                  AND (other.event_type = $event_type OR other.sensor_id = $sensor_id)
                RETURN other.event_id, other.embedding_text
                LIMIT 50
            """, parameters={
                'current_event_id': event.event_id,
                'event_type': event.event_type,
                'sensor_id': event.sensor_id
            })
            
            # Calcula similaridades e cria relacionamentos
            for row in result.get_next():
                other_event_id = row[0]
                other_embedding_text = row[1]
                
                if other_embedding_text:
                    other_embedding = self.embedding_model.encode(other_embedding_text)
                    similarity = np.dot(current_embedding, other_embedding) / (
                        np.linalg.norm(current_embedding) * np.linalg.norm(other_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        # Cria relacionamento semântico bidirecional
                        for from_id, to_id in [(event.event_id, other_event_id), (other_event_id, event.event_id)]:
                            self.connection.execute("""
                                MATCH (from:Event {event_id: $from_event})
                                MATCH (to:Event {event_id: $to_event})
                                CREATE (from)-[r:SemanticRelation {
                                    similarity_score: $similarity,
                                    shared_context: $context
                                }]->(to)
                            """, parameters={
                                'from_event': from_id,
                                'to_event': to_id,
                                'similarity': similarity,
                                'context': f"semantic_similarity_{event.event_type}"
                            })
            
        except Exception as e:
            logger.error(f"Error creating semantic relationships for {event.event_id}: {str(e)}")
    
    def temporal_retrieval(self, query: str, query_time: datetime = None, top_k: int = 10) -> List[DynamicEventUnit]:
        """
        Recuperação temporal usando consultas openCypher no Kùzu.
        Implementa recuperação multi-hop do DyG-RAG.
        """
        if not self.embedding_model:
            logger.warning("No embedding model available for semantic search")
            return self._simple_temporal_retrieval(query, query_time, top_k)
        
        try:
            # Busca semântica inicial
            query_embedding = self.embedding_model.encode(query)
            
            # Consulta complexa com relacionamentos temporais e semânticos
            cypher_query = """
            MATCH (e:Event)
            OPTIONAL MATCH (e)-[tr:TemporalRelation]-(related:Event)
            OPTIONAL MATCH (e)-[sr:SemanticRelation]-(semantic:Event)
            WITH e, 
                 COUNT(DISTINCT tr) as temporal_connections,
                 COUNT(DISTINCT sr) as semantic_connections,
                 AVG(tr.weight) as avg_temporal_weight,
                 AVG(sr.similarity_score) as avg_semantic_score
            """
            
            # Filtros baseados na consulta
            if query_time:
                time_range_start = (query_time - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
                time_range_end = (query_time + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
                cypher_query += f"""
                WHERE e.timestamp >= datetime('{time_range_start}')
                  AND e.timestamp <= datetime('{time_range_end}')
                """
            
            # Adiciona filtros por palavras-chave
            keywords = query.lower().split()
            if any(kw in ['ruído', 'noise', 'loud'] for kw in keywords):
                cypher_query += " AND e.loudness > 60"
            
            if any(kw in ['violação', 'violation'] for kw in keywords):
                cypher_query += " AND e.violates_regulations = true"
            
            # Ordena por relevância (combinação de conexões e contexto)
            cypher_query += """
            RETURN e.event_id, e.timestamp, e.event_type, e.loudness, e.sensor_id,
                   e.description, e.metadata_json, e.severity_level,
                   (temporal_connections * 0.3 + semantic_connections * 0.7) as relevance_score
            ORDER BY relevance_score DESC, e.loudness DESC
            LIMIT $top_k
            """
            
            result = self.connection.execute(cypher_query, parameters={'top_k': top_k})
            
            # Converte resultados para DynamicEventUnit
            events = []
            for row in result.get_next():
                event_data = {
                    'event_id': row[0],
                    'timestamp': datetime.fromisoformat(str(row[1])),
                    'event_type': row[2],
                    'loudness': row[3],
                    'sensor_id': row[4],
                    'description': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {}
                }
                
                event = DynamicEventUnit(**event_data)
                events.append(event)
            
            logger.info(f"Retrieved {len(events)} events using Kùzu temporal retrieval")
            return events
            
        except Exception as e:
            logger.error(f"Error in Kùzu temporal retrieval: {str(e)}")
            return []
    
    def _simple_temporal_retrieval(self, query: str, query_time: datetime, top_k: int) -> List[DynamicEventUnit]:
        """Recuperação simples sem embeddings semânticos"""
        try:
            cypher_query = """
            MATCH (e:Event)
            WHERE 1=1
            """
            
            parameters = {'top_k': top_k}
            
            # Filtros temporais
            if query_time:
                time_range_start = (query_time - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                time_range_end = (query_time + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
                cypher_query += """
                AND e.timestamp >= datetime($start_time)
                AND e.timestamp <= datetime($end_time)
                """
                parameters.update({
                    'start_time': time_range_start,
                    'end_time': time_range_end
                })
            
            # Ordena por timestamp recente e intensidade
            cypher_query += """
            RETURN e.event_id, e.timestamp, e.event_type, e.loudness, 
                   e.sensor_id, e.description, e.metadata_json
            ORDER BY e.timestamp DESC, e.loudness DESC
            LIMIT $top_k
            """
            
            result = self.connection.execute(cypher_query, parameters=parameters)
            
            events = []
            for row in result.get_next():
                event_data = {
                    'event_id': row[0],
                    'timestamp': datetime.fromisoformat(str(row[1])),
                    'event_type': row[2],
                    'loudness': row[3],
                    'sensor_id': row[4],
                    'description': row[5],
                    'metadata': json.loads(row[6]) if row[6] else {}
                }
                
                event = DynamicEventUnit(**event_data)
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error in simple temporal retrieval: {str(e)}")
            return []
    
    def extract_temporal_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Extrai padrões temporais usando análise de grafo.
        Implementa análise de padrões do DyG-RAG.
        """
        try:
            patterns = {}
            
            # Padrão 1: Sequências temporais mais comuns
            sequence_query = """
            MATCH (e1:Event)-[r:TemporalRelation]->(e2:Event)
            WHERE e1.timestamp >= datetime($start_time)
              AND e2.timestamp <= datetime($end_time)
              AND r.time_diff_seconds <= 1800
            RETURN e1.event_type, e2.event_type, COUNT(*) as frequency
            ORDER BY frequency DESC
            LIMIT 10
            """
            
            result = self.connection.execute(sequence_query, parameters={
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            patterns['common_sequences'] = []
            for row in result.get_next():
                patterns['common_sequences'].append({
                    'from_type': row[0],
                    'to_type': row[1],
                    'frequency': row[2]
                })
            
            # Padrão 2: Eventos centrais (mais conectados)
            centrality_query = """
            MATCH (e:Event)
            WHERE e.timestamp >= datetime($start_time)
              AND e.timestamp <= datetime($end_time)
            OPTIONAL MATCH (e)-[r]-(connected:Event)
            WITH e, COUNT(DISTINCT r) as connection_count
            WHERE connection_count > 0
            RETURN e.event_id, e.event_type, e.timestamp, connection_count
            ORDER BY connection_count DESC
            LIMIT 5
            """
            
            result = self.connection.execute(centrality_query, parameters={
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            patterns['