# src/multi_scenario_system.py
import logging
import time
import re
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class TemporalAnchor:
    """Representa uma âncora temporal para eventos de construção civil"""
    timestamp: Optional[datetime] = None
    time_expression: Optional[str] = None  
    temporal_type: str = "relative"  # absolute, relative, duration, schedule
    confidence: float = 1.0

@dataclass
class DynamicEventUnit:
    """Dynamic Event Unit (DEU) adaptado para construção civil"""
    id: str
    content: str
    entities: List[str]
    temporal_anchor: TemporalAnchor
    event_type: str  # "regulation", "procedure", "measurement", "schedule", etc.
    source_document: str
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

class TemporalExtractor:
    """Extrator de informações temporais específico para construção civil"""
    
    def __init__(self):
        self.temporal_patterns = {
            'time_schedules': [
                r'das? (\d{1,2})h? às? (\d{1,2})h?',
                r'entre (\d{1,2})h? e (\d{1,2})h?',
                r'durante o dia|período diurno',
                r'durante a noite|período noturno',
                r'horário comercial',
                r'fim de semana|sábado|domingo'
            ],
            'durations': [
                r'por (\d+) (minutos?|horas?|dias?|semanas?|meses?)',
                r'a cada (\d+)(m³|dia|semana|mês)',
                r'até (\d+) (minutos?|horas?|dias?)',
                r'máximo de (\d+) (minutos?|horas?|dias?)'
            ],
            'thresholds': [
                r'superior a (\d+)(m²|m³|dB|lux)',
                r'acima de (\d+) metros',
                r'não pode exceder (\d+)',
                r'limite de (\d+)'
            ],
            'regulatory_dates': [
                r'NBR \d+/(\d{4})',
                r'aprovado em (\d{4})',
                r'vigente desde (\d{4})',
                r'revisão (\d{4})'
            ],
            'sequence_markers': [
                r'antes de|previamente|primeiro|inicialmente',
                r'após|depois de|em seguida|posteriormente',
                r'durante|enquanto|simultaneamente',
                r'ao final|finalmente|por último'
            ]
        }
    
    def extract_temporal_info(self, text: str) -> TemporalAnchor:
        """Extrai informação temporal com foco em construção civil"""
        text_lower = text.lower()
        
        # Detecta horários específicos
        for pattern in self.temporal_patterns['time_schedules']:
            match = re.search(pattern, text_lower)
            if match:
                return TemporalAnchor(
                    time_expression=match.group(0),
                    temporal_type="schedule",
                    confidence=0.9
                )
        
        # Detecta durações e frequências
        for pattern in self.temporal_patterns['durations']:
            match = re.search(pattern, text_lower)
            if match:
                return TemporalAnchor(
                    time_expression=match.group(0),
                    temporal_type="duration",
                    confidence=0.8
                )
        
        # Detecta limites e thresholds temporais
        for pattern in self.temporal_patterns['thresholds']:
            match = re.search(pattern, text_lower)
            if match:
                return TemporalAnchor(
                    time_expression=match.group(0),
                    temporal_type="threshold",
                    confidence=0.7
                )
        
        # Detecta datas regulamentares
        for pattern in self.temporal_patterns['regulatory_dates']:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    year = int(match.group(1))
                    timestamp = datetime(year, 1, 1)
                    return TemporalAnchor(
                        timestamp=timestamp,
                        time_expression=f"regulamentação {year}",
                        temporal_type="absolute",
                        confidence=0.8
                    )
                except:
                    pass
        
        # Detecta marcadores de sequência
        for pattern in self.temporal_patterns['sequence_markers']:
            if re.search(pattern, text_lower):
                return TemporalAnchor(
                    time_expression=pattern.split('|')[0],
                    temporal_type="sequence",
                    confidence=0.6
                )
        
        # Temporal padrão se nada encontrado
        return TemporalAnchor(
            time_expression="contexto_geral",
            temporal_type="relative",
            confidence=0.3
        )

class EntityExtractor:
    """Extrator de entidades específicas do domínio de construção civil"""
    
    def __init__(self):
        self.entity_patterns = {
            'regulations': [
                r'NBR \d+',
                r'Lei \d+/\d+',
                r'Resolução \d+',
                r'Portaria \d+',
                r'NR-?\d+'
            ],
            'measurements': [
                r'\d+\s*dB',
                r'\d+\s*lux',
                r'\d+\s*m[²³]?',
                r'\d+\s*kg/m³',
                r'\d+\s*%',
                r'\d+\s*MPa'
            ],
            'equipment': [
                r'capacete[s]?',
                r'óculos de proteção',
                r'luvas?',
                r'calçados? de segurança',
                r'cint[oõ]s? de segurança',
                r'EPI[s]?',
                r'medidor[es]?',
                r'equipamentos?'
            ],
            'processes': [
                r'ensaios? de resistência',
                r'controle de qualidade',
                r'medição de ruído',
                r'licenciamento',
                r'EIA[/-]?RIMA',
                r'concretagem',
                r'calibração'
            ],
            'locations': [
                r'canteiro[s]?',
                r'obra[s]?',
                r'área[s]? residencial[is]?',
                r'limite da propriedade',
                r'receptor[es]? sensíve[is]?'
            ]
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """Extrai entidades do texto com categorização"""
        entities = []
        text_lower = text.lower()
        
        for category, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                entities.extend([f"{category}:{match}" for match in matches])
        
        return list(dict.fromkeys(entities))  # Remove duplicatas

# ==================== CENÁRIO A: RAG VETORIAL APENAS ====================

class VectorOnlyRAG:
    """Cenário A: RAG tradicional usando apenas busca vetorial"""
    
    def __init__(self, config=None):
        self.config = config
        self.device = self._get_optimal_device()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.faiss_index = None
        self.documents = []
        self.document_chunks = []
        
        logger.info("✅ Vector-Only RAG System inicializado")
    
    def _get_optimal_device(self):
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
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        """Divide documentos em chunks menores para melhor recuperação"""
        chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            words = content.split()
            
            # Cria chunks com overlap
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if len(chunk_text.strip()) > 50:  # Ignora chunks muito pequenos
                    chunk = {
                        'id': f"{doc['id']}_chunk_{i//chunk_size}",
                        'content': chunk_text,
                        'source_doc': doc['id'],
                        'title': doc.get('title', ''),
                        'category': doc.get('category', ''),
                        'keywords': doc.get('keywords', []),
                        'chunk_index': i//chunk_size
                    }
                    chunks.append(chunk)
        
        return chunks
    
    def build_index(self, documents: List[Dict]):
        """Constrói índice FAISS para busca vetorial"""
        logger.info("🔧 Construindo índice vetorial...")
        
        # Divide documentos em chunks
        self.document_chunks = self.chunk_documents(documents)
        
        if not self.document_chunks:
            raise ValueError("Nenhum chunk de documento criado")
        
        # Gera embeddings
        contents = [chunk['content'] for chunk in self.document_chunks]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        # Cria índice FAISS
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        logger.info(f"✅ Índice construído: {len(self.document_chunks)} chunks, dimensão {dimension}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recupera chunks mais relevantes"""
        if not self.faiss_index:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Busca no índice
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.document_chunks))
        )
        
        # Recupera chunks
        retrieved_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_chunks):
                chunk = self.document_chunks[idx].copy()
                chunk['relevance_score'] = 1.0 / (1.0 + float(score))
                retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def query(self, question: str) -> Dict[str, Any]:
        """Interface principal para consultas"""
        start_time = time.time()
        
        try:
            # Recupera chunks relevantes
            retrieved_chunks = self.retrieve(question)
            
            # Constrói resposta simples
            if retrieved_chunks:
                best_chunk = retrieved_chunks[0]
                answer = f"Baseado no documento '{best_chunk['title']}':\n\n{best_chunk['content']}"
                
                if len(retrieved_chunks) > 1:
                    answer += f"\n\n(Encontrados {len(retrieved_chunks)} trechos relacionados)"
            else:
                answer = "Nenhuma informação relevante encontrada."
            
            response_time = time.time() - start_time
            
            return {
                'question': question,
                'answer': answer,
                'retrieved_chunks': retrieved_chunks,
                'response_time': response_time,
                'relevance_score': retrieved_chunks[0]['relevance_score'] if retrieved_chunks else 0,
                'retrieved_docs': len(retrieved_chunks),
                'method': 'vector_only_rag',
                'metadata': {
                    'total_chunks': len(self.document_chunks),
                    'device': self.device
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta Vector RAG: {e}")
            return {
                'question': question,
                'answer': f"❌ Erro: {str(e)}",
                'retrieved_chunks': [],
                'response_time': time.time() - start_time,
                'relevance_score': 0,
                'retrieved_docs': 0,
                'method': 'vector_only_rag',
                'metadata': {'error': str(e)}
            }

# ==================== CENÁRIO B: RAG HÍBRIDO (VETORIAL + GRAFOS) ====================

class HybridRAG:
    """Cenário B: RAG híbrido usando busca vetorial + raciocínio em grafos"""
    
    def __init__(self, config=None):
        self.config = config
        self.device = self._get_optimal_device()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.temporal_extractor = TemporalExtractor()
        self.entity_extractor = EntityExtractor()
        
        # Estruturas de dados
        self.deus: List[DynamicEventUnit] = []
        self.event_graph = nx.MultiDiGraph()
        self.faiss_index = None
        self.deu_index_map = {}
        
        logger.info("✅ Hybrid RAG System inicializado")
    
    def _get_optimal_device(self):
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
    
    def create_dynamic_event_units(self, documents: List[Dict]) -> List[DynamicEventUnit]:
        """Converte documentos em Dynamic Event Units"""
        deus = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_id = doc.get('id', 'unknown')
            
            # Divide conteúdo em eventos/sentenças
            sentences = self._split_into_events(content)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 30:  # Ignora sentenças muito curtas
                    continue
                
                # Extrai informações temporais e entidades
                temporal_anchor = self.temporal_extractor.extract_temporal_info(sentence)
                entities = self.entity_extractor.extract_entities(sentence)
                
                # Classifica tipo de evento
                event_type = self._classify_event_type(sentence, doc.get('category', 'general'))
                
                # Cria DEU
                deu = DynamicEventUnit(
                    id=f"{doc_id}_event_{i}",
                    content=sentence,
                    entities=entities,
                    temporal_anchor=temporal_anchor,
                    event_type=event_type,
                    source_document=doc_id,
                    keywords=doc.get('keywords', [])
                )
                
                deus.append(deu)
        
        logger.info(f"✅ Criados {len(deus)} Dynamic Event Units")
        return deus
    
    def _split_into_events(self, content: str) -> List[str]:
        """Divide conteúdo em eventos/sentenças relevantes"""
        # Padrões específicos para construção civil
        split_patterns = [
            r'\. (?=[A-ZÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙ])',  # Final de sentença
            r';\s*(?=[A-ZÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙ])',  # Ponto e vírgula + maiúscula
            r':\s*(?=\d+\.)',  # Dois pontos + numeração
            r'(?<=\d)\.\s*(?=[A-ZÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙ])'  # Número + ponto + maiúscula
        ]
        
        sentences = [content]
        
        for pattern in split_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # Limpa e filtra
        cleaned_sentences = []
        for s in sentences:
            s = s.strip()
            if s and len(s) > 20:
                cleaned_sentences.append(s)
        
        return cleaned_sentences
    
    def _classify_event_type(self, content: str, category: str) -> str:
        """Classifica tipo de evento baseado no conteúdo"""
        content_lower = content.lower()
        
        event_classifiers = {
            'measurement': ['medição', 'medir', 'aferir', 'calibrado', 'db', 'lux', 'ensaio'],
            'requirement': ['obrigatório', 'deve', 'necessário', 'requer', 'precisa', 'exigido'],
            'procedure': ['realizar', 'executar', 'implementar', 'seguir', 'aplicar'],
            'limit': ['limite', 'máximo', 'mínimo', 'não pode exceder', 'superior a'],
            'schedule': ['horário', 'das', 'às', 'período', 'durante', 'entre'],
            'authorization': ['autorização', 'licença', 'permissão', 'aprovação'],
            'equipment': ['epi', 'equipamento', 'capacete', 'óculos', 'luvas'],
            'regulation': ['nbr', 'lei', 'norma', 'regulamento', 'resolução']
        }
        
        for event_type, keywords in event_classifiers.items():
            if any(keyword in content_lower for keyword in keywords):
                return event_type
        
        return category
    
    def build_event_graph(self, deus: List[DynamicEventUnit]):
        """Constrói grafo de eventos com relacionamentos temporais e semânticos"""
        self.event_graph = nx.MultiDiGraph()
        
        # Adiciona nós
        for deu in deus:
            self.event_graph.add_node(deu.id, deu=deu)
        
        # Adiciona arestas baseadas em relacionamentos
        for i, deu1 in enumerate(deus):
            for j, deu2 in enumerate(deus):
                if i >= j:
                    continue
                
                # Similaridade de entidades
                entities1 = set(deu1.entities)
                entities2 = set(deu2.entities)
                
                entity_similarity = 0
                if entities1 and entities2:
                    entity_similarity = len(entities1 & entities2) / len(entities1 | entities2)
                
                # Proximidade temporal
                temporal_similarity = self._calculate_temporal_similarity(
                    deu1.temporal_anchor, deu2.temporal_anchor
                )
                
                # Similaridade de conteúdo (palavras em comum)
                words1 = set(deu1.content.lower().split())
                words2 = set(deu2.content.lower().split())
                content_similarity = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                
                # Score combinado
                combined_similarity = (0.4 * entity_similarity + 
                                     0.3 * temporal_similarity + 
                                     0.3 * content_similarity)
                
                # Cria aresta se similaridade suficiente
                if combined_similarity > 0.25:
                    self.event_graph.add_edge(
                        deu1.id, deu2.id,
                        weight=combined_similarity,
                        relation_type="semantic_temporal",
                        entity_overlap=entity_similarity,
                        temporal_similarity=temporal_similarity
                    )
        
        logger.info(f"✅ Grafo construído: {self.event_graph.number_of_nodes()} nós, {self.event_graph.number_of_edges()} arestas")
    
    def _calculate_temporal_similarity(self, anchor1: TemporalAnchor, anchor2: TemporalAnchor) -> float:
        """Calcula similaridade temporal entre âncoras"""
        # Timestamps absolutos
        if anchor1.timestamp and anchor2.timestamp:
            time_diff = abs((anchor1.timestamp - anchor2.timestamp).days)
            return max(0, 1 - time_diff / 365)
        
        # Mesmo tipo temporal
        if anchor1.temporal_type == anchor2.temporal_type:
            return 0.6
        
        # Expressões similares
        if (anchor1.time_expression and anchor2.time_expression and 
            anchor1.time_expression in anchor2.time_expression):
            return 0.8
        
        return 0.1
    
    def build_faiss_index(self, deus: List[DynamicEventUnit]):
        """Constrói índice FAISS para busca vetorial inicial"""
        if not deus:
            return
        
        # Gera embeddings
        contents = [deu.content for deu in deus]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        
        # Salva embeddings nos DEUs
        for deu, embedding in zip(deus, embeddings):
            deu.embedding = embedding
        
        # Cria índice FAISS
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Mapeamento índice -> DEU
        self.deu_index_map = {i: deu.id for i, deu in enumerate(deus)}
        
        logger.info(f"✅ Índice FAISS criado: {len(deus)} DEUs, dimensão {dimension}")
    
    def retrieve_seed_events(self, query: str, top_k: int = 3) -> List[DynamicEventUnit]:
        """Recupera eventos semente usando busca vetorial"""
        if not self.faiss_index:
            return []
        
        # Busca vetorial inicial
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.deus))
        )
        
        # Recupera DEUs
        seed_events = []
        for score, idx in zip(scores[0], indices[0]):
            if idx in self.deu_index_map:
                deu_id = self.deu_index_map[idx]
                deu = next((d for d in self.deus if d.id == deu_id), None)
                if deu:
                    relevance_score = 1.0 / (1.0 + float(score))
                    deu_copy = deu
                    seed_events.append((deu_copy, relevance_score))
        
        return [deu for deu, _ in seed_events]
    
    def expand_through_graph(self, seed_events: List[DynamicEventUnit], 
                           query: str, max_hops: int = 2) -> List[DynamicEventUnit]:
        """Expande através do grafo para encontrar eventos relacionados"""
        expanded_events = set(seed_events)
        current_nodes = [deu.id for deu in seed_events]
        
        for hop in range(max_hops):
            next_nodes = []
            
            for node_id in current_nodes:
                if node_id in self.event_graph:
                    # Vizinhos diretos
                    neighbors = list(self.event_graph.successors(node_id))
                    neighbors.extend(list(self.event_graph.predecessors(node_id)))
                    
                    for neighbor_id in neighbors:
                        neighbor_deu = next((d for d in self.deus if d.id == neighbor_id), None)
                        if neighbor_deu and neighbor_deu not in expanded_events:
                            # Verifica relevância do vizinho
                            relevance = self._calculate_query_relevance(query, neighbor_deu)
                            if relevance > 0.2:  # Threshold de relevância
                                expanded_events.add(neighbor_deu)
                                next_nodes.append(neighbor_id)
            
            current_nodes = next_nodes
            if not next_nodes:
                break
        
        return list(expanded_events)
    
    def _calculate_query_relevance(self, query: str, deu: DynamicEventUnit) -> float:
        """Calcula relevância de um DEU para uma query"""
        query_words = set(query.lower().split())
        deu_words = set(deu.content.lower().split())
        
        # Similaridade de palavras
        word_overlap = len(query_words & deu_words) / len(query_words | deu_words) if query_words | deu_words else 0
        
        # Bonus por entidades em comum
        query_entities = self.entity_extractor.extract_entities(query)
        entity_overlap = self._calculate_entity_similarity(query_entities, deu.entities)
        
        # Bonus por temporal matching
        query_temporal = self.temporal_extractor.extract_temporal_info(query)
        temporal_match = self._calculate_temporal_similarity(query_temporal, deu.temporal_anchor)
        
        return 0.5 * word_overlap + 0.3 * entity_overlap + 0.2 * temporal_match
    
    def _calculate_entity_similarity(self, entities1: List[str], entities2: List[str]) -> float:
        """Calcula similaridade entre listas de entidades"""
        if not entities1 or not entities2:
            return 0.0
        
        set1 = set(entities1)
        set2 = set(entities2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def time_chain_of_thought(self, query: str, events: List[DynamicEventUnit]) -> str:
        """Gera resposta usando Time Chain-of-Thought"""
        if not events:
            return "❌ Nenhum evento relevante encontrado."
        
        # Ordena eventos por relevância temporal e semântica
        sorted_events = self._sort_events_temporally(events, query)
        
        # Constrói resposta estruturada
        response_parts = []
        
        # Contexto temporal da query
        query_temporal = self.temporal_extractor.extract_temporal_info(query)
        if query_temporal.time_expression and query_temporal.confidence > 0.5:
            response_parts.append(f"Considerando o contexto '{query_temporal.time_expression}':")
        
        # Apresenta eventos principais
        for i, event in enumerate(sorted_events[:3], 1):
            temporal_info = ""
            if event.temporal_anchor.time_expression:
                temporal_info = f" [{event.temporal_anchor.time_expression}]"
            
            response_parts.append(f"\n{i}. {event.content}{temporal_info}")
        
        # Adiciona conexões encontradas
        connections = self._find_event_connections(sorted_events[:3])
        if connections:
            response_parts.append(f"\nRelacionamentos: {', '.join(connections)}")
        
        return ''.join(response_parts)
    
    def _sort_events_temporally(self, events: List[DynamicEventUnit], query: str) -> List[DynamicEventUnit]:
        """Ordena eventos por relevância temporal e sequencial"""
        # Separa eventos com timestamps absolutos
        absolute_events = [e for e in events if e.temporal_anchor.timestamp]
        relative_events = [e for e in events if not e.temporal_anchor.timestamp]
        
        # Ordena absolutos por timestamp
        absolute_events.sort(key=lambda x: x.temporal_anchor.timestamp)
        
        # Ordena relativos por relevância para a query
        relative_events.sort(key=lambda x: self._calculate_query_relevance(query, x), reverse=True)
        
        return absolute_events + relative_events
    
    def _find_event_connections(self, events: List[DynamicEventUnit]) -> List[str]:
        """Encontra conexões entre eventos no grafo"""
        connections = []
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i >= j:
                    continue
                
                if self.event_graph.has_edge(event1.id, event2.id):
                    edge_data = self.event_graph.get_edge_data(event1.id, event2.id)
                    if edge_data:
                        weight = list(edge_data.values())[0].get('weight', 0)
                        if weight > 0.5:
                            connections.append(f"{event1.event_type}↔{event2.event_type}")
        
        return list(set(connections))
    
    def build_system(self, documents: List[Dict]):
        """Constrói sistema completo (DEUs + Grafo + Índice)"""
        logger.info("🔧 Construindo sistema híbrido...")
        
        # 1. Cria DEUs
        self.deus = self.create_dynamic_event_units(documents)
        
        # 2. Constrói grafo de eventos
        self.build_event_graph(self.deus)
        
        # 3. Constrói índice FAISS
        self.build_faiss_index(self.deus)
        
        logger.info("✅ Sistema híbrido construído!")
    
    def query(self, question: str, use_graph_expansion: bool = True) -> Dict[str, Any]:
        """Interface principal para consultas híbridas"""
        start_time = time.time()
        
        try:
            # 1. Recuperação inicial vetorial (eventos semente)
            seed_events = self.retrieve_seed_events(question, top_k=3)
            
            # 2. Expansão através do grafo (opcional)
            if use_graph_expansion and seed_events:
                expanded_events = self.expand_through_graph(seed_events, question)
            else:
                expanded_events = seed_events
            
            # 3. Geração usando Time Chain-of-Thought
            answer = self.time_chain_of_thought(question, expanded_events)
            
            response_time = time.time() - start_time
            
            # Calcula relevância média
            avg_relevance = 0
            if expanded_events:
                relevances = [self._calculate_query_relevance(question, event) for event in expanded_events]
                avg_relevance = sum(relevances) / len(relevances)
            
            return {
                'question': question,
                'answer': answer,
                'seed_events': len(seed_events),
                'expanded_events': len(expanded_events),
                'response_time': response_time,
                'relevance_score': avg_relevance,
                'retrieved_docs': len(expanded_events),
                'method': 'hybrid_rag',
                'graph_expansion_used': use_graph_expansion,
                'metadata': {
                    'total_deus': len(self.deus),
                    'graph_nodes': self.event_graph.number_of_nodes(),
                    'graph_edges': self.event_graph.number_of_edges(),
                    'device': self.device
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta Hybrid RAG: {e}")
            return {
                'question': question,
                'answer': f"❌ Erro: {str(e)}",
                'seed_events': 0,
                'expanded_events': 0,
                'response_time': time.time() - start_time,
                'relevance_score': 0,
                'retrieved_docs': 0,
                'method': 'hybrid_rag',
                'graph_expansion_used': use_graph_expansion,
                'metadata': {'error': str(e)}
            }

# ==================== CENÁRIO C: LLM APENAS (SEM RAG) ====================

class LLMOnly:
    """Cenário C: LLM puro sem recuperação de documentos"""
    
    def __init__(self, api_provider: str = "openai"):
        self.api_provider = api_provider
        self.client = None
        self.use_mock = False
        self.construction_knowledge = self._build_construction_knowledge()
        
        # Custos aproximados por token
        self.cost_per_1k_tokens = {
            "openai": 0.002,
            "anthropic": 0.003,
            "mock": 0.002
        }
        
        self._setup_client()
        logger.info("✅ LLM-Only System inicializado")
    
    def _build_construction_knowledge(self) -> str:
        """Constrói conhecimento base sobre construção civil para o LLM"""
        return """
        Conhecimento base sobre construção civil:
        
        NORMAS DE RUÍDO:
        - NBR 10151: Limites de ruído em áreas urbanas
        - Dia (7h-22h): máximo 70 dB em áreas residenciais
        - Noite (22h-7h): máximo 60 dB em áreas residenciais
        - Medição: medidores calibrados, no limite da propriedade
        
        EQUIPAMENTOS DE PROTEÇÃO (EPIs):
        - Capacete de segurança classe A ou B
        - Óculos de proteção contra impactos
        - Luvas adequadas à atividade
        - Calçados de segurança com biqueira de aço
        - Cintos de segurança para altura > 2 metros
        
        CONTROLE DE QUALIDADE:
        - Concreto: ensaios a cada 50m³ ou por dia de concretagem
        - Inspeções regulares de materiais
        - Documentação completa de procedimentos
        
        OBRAS NOTURNAS:
        - Autorização especial da prefeitura
        - Iluminação mínima: 200 lux
        - Sinalização reforçada com dispositivos refletivos
        - Horário: geralmente 22h às 6h
        
        IMPACTO AMBIENTAL:
        - EIA/RIMA obrigatório para obras > 3000m²
        - Licenciamento prévio necessário
        - Plano de gerenciamento de resíduos
        - Controle de erosão e proteção hídrica
        """
    
    def _setup_client(self):
        """Configura cliente da API ou mock"""
        try:
            if self.api_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("⚠️ OPENAI_API_KEY não encontrada - usando mock")
                    self.use_mock = True
                    return
                
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                
                # Testa conexão
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    logger.info("✅ Cliente OpenAI configurado")
                except Exception as e:
                    logger.warning(f"⚠️ Erro OpenAI: {e} - usando mock")
                    self.use_mock = True
                    
            else:
                logger.info("🔄 Usando mock para outros provedores")
                self.use_mock = True
                
        except ImportError:
            logger.warning("⚠️ Biblioteca OpenAI não instalada - usando mock")
            self.use_mock = True
    
    def _query_openai(self, question: str) -> Dict[str, Any]:
        """Consulta OpenAI real"""
        if self.use_mock:
            return self._query_mock(question)
        
        try:
            system_prompt = f"""Você é um especialista em construção civil e engenharia. 
            Responda perguntas técnicas com base no seu conhecimento e nas informações fornecidas.
            
            {self.construction_knowledge}
            
            Forneça respostas precisas, técnicas e baseadas em normas quando aplicável."""
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            answer = response.choices[0].message.content.strip()
            
            # Calcula custo
            total_tokens = response.usage.total_tokens
            estimated_cost = (total_tokens / 1000) * self.cost_per_1k_tokens["openai"]
            
            return {
                'answer': answer,
                'response_time': response_time,
                'tokens_used': total_tokens,
                'estimated_cost_usd': estimated_cost,
                'model': 'gpt-3.5-turbo',
                'source': 'openai_api'
            }
            
        except Exception as e:
            logger.error(f"❌ Erro OpenAI: {e} - fallback para mock")
            return self._query_mock(question)
    
    def _query_mock(self, question: str) -> Dict[str, Any]:
        """Mock inteligente baseado no conhecimento de construção civil"""
        start_time = time.time()
        
        # Simula delay de processamento
        import random
        delay = random.uniform(1.0, 2.5)
        time.sleep(delay)
        
        question_lower = question.lower()
        
        # Respostas baseadas em palavras-chave
        if any(word in question_lower for word in ['ruído', 'barulho', 'som', 'decibel', 'db']):
            answer = """Conforme a NBR 10151, os limites de ruído em obras urbanas são:
            
            • Período diurno (7h às 22h): máximo 70 dB em áreas residenciais
            • Período noturno (22h às 7h): máximo 60 dB em áreas residenciais
            
            A medição deve ser realizada com equipamentos calibrados, posicionados no limite da propriedade mais próxima aos receptores sensíveis. É obrigatório manter registros horários das medições."""
            
        elif any(word in question_lower for word in ['epi', 'equipamento', 'proteção', 'capacete', 'segurança']):
            answer = """Os EPIs obrigatórios em canteiros de obra incluem:
            
            • Capacete de segurança classe A ou B
            • Óculos de proteção contra impactos
            • Luvas adequadas ao tipo de atividade
            • Calçados de segurança com biqueira de aço
            • Cintos de segurança para trabalhos em altura superior a 2 metros
            
            A empresa deve fornecer gratuitamente e treinar os trabalhadores sobre o uso correto."""
            
        elif any(word in question_lower for word in ['concreto', 'qualidade', 'ensaio', 'resistência']):
            answer = """O controle de qualidade do concreto estrutural exige:
            
            • Ensaios de resistência à compressão a cada 50m³ ou a cada dia de concretagem (prevalece o menor valor)
            • Inspeções regulares dos materiais utilizados
            • Documentação completa de todos os procedimentos
            • Auditorias internas periódicas
            
            Os ensaios devem seguir as normas técnicas aplicáveis (NBR 5738, NBR 5739)."""
            
        elif any(word in question_lower for word in ['noturna', 'noturno', 'noite', '22h', 'iluminação']):
            answer = """Para obras noturnas são necessários:
            
            • Autorização especial da prefeitura municipal
            • Iluminação adequada com intensidade mínima de 200 lux
            • Sinalização reforçada com dispositivos refletivos e luminosos
            • Equipes especializadas com treinamento em segurança noturna
            • Horário permitido: geralmente das 22h às 6h (varia conforme área urbana)"""
            
        elif any(word in question_lower for word in ['eia', 'rima', 'ambiental', 'licenciamento']):
            answer = """Para impacto ambiental em construções:
            
            • EIA/RIMA obrigatório para obras com área superior a 3000m²
            • Licenciamento prévio do órgão competente
            • Plano de gerenciamento de resíduos sólidos
            • Medidas de controle de erosão
            • Proteção de recursos hídricos
            
            O não cumprimento pode resultar em embargo da obra."""
            
        else:
            answer = """Com base nas normas de construção civil, posso ajudar com informações sobre:
            
            • Normas de ruído (NBR 10151)
            • Equipamentos de proteção individual (EPIs)
            • Controle de qualidade de materiais
            • Regulamentações para obras noturnas
            • Licenciamento e impacto ambiental
            
            Poderia ser mais específico em sua pergunta?"""
        
        response_time = time.time() - start_time
        
        # Estima tokens e custo
        estimated_tokens = len(question + answer) // 4  # ~4 chars por token
        estimated_cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens["mock"]
        
        return {
            'answer': answer,
            'response_time': response_time,
            'tokens_used': estimated_tokens,
            'estimated_cost_usd': estimated_cost,
            'model': 'construction_expert_mock',
            'source': 'intelligent_mock'
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Interface principal para consultas LLM-only"""
        try:
            # Chama API apropriada
            if self.api_provider == "openai":
                result = self._query_openai(question)
            else:
                result = self._query_mock(question)
            
            # Adiciona metadados
            result.update({
                'question': question,
                'method': 'llm_only',
                'api_provider': self.api_provider,
                'using_mock': self.use_mock,
                'retrieved_docs': 0,  # Sem recuperação
                'relevance_score': 0.8 if not result['answer'].startswith('❌') else 0
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro LLM-Only: {e}")
            return {
                'question': question,
                'answer': f"❌ Erro ao processar consulta: {str(e)}",
                'response_time': 0,
                'tokens_used': 0,
                'estimated_cost_usd': 0,
                'method': 'llm_only',
                'api_provider': self.api_provider,
                'using_mock': self.use_mock,
                'retrieved_docs': 0,
                'relevance_score': 0,
                'metadata': {'error': str(e)}
            }

# ==================== SISTEMA MULTI-CENÁRIO ====================

class MultiScenarioSystem:
    """Sistema que permite alternar entre os três cenários"""
    
    def __init__(self, config=None):
        self.config = config
        
        # Inicializa os três sistemas
        self.vector_rag = VectorOnlyRAG(config)
        self.hybrid_rag = HybridRAG(config)
        self.llm_only = LLMOnly()
        
        self.documents = []
        self.is_built = False
        
        logger.info("✅ Multi-Scenario System inicializado")
    
    def load_documents(self, documents: List[Dict]):
        """Carrega documentos no sistema"""
        self.documents = documents
        logger.info(f"📄 Carregados {len(documents)} documentos")
    
    def build_all_scenarios(self):
        """Constrói índices para todos os cenários que precisam"""
        if not self.documents:
            raise ValueError("Nenhum documento carregado")
        
        logger.info("🏗️ Construindo todos os cenários...")
        
        # Cenário A: Vector RAG
        logger.info("📊 Construindo Cenário A: Vector-Only RAG")
        self.vector_rag.build_index(self.documents)
        
        # Cenário B: Hybrid RAG
        logger.info("🔗 Construindo Cenário B: Hybrid RAG")
        self.hybrid_rag.build_system(self.documents)
        
        # Cenário C: LLM-Only (não precisa de construção)
        logger.info("🤖 Cenário C: LLM-Only pronto")
        
        self.is_built = True
        logger.info("✅ Todos os cenários construídos!")
    
    def query_scenario_a(self, question: str) -> Dict[str, Any]:
        """Consulta usando Cenário A: Vector-Only RAG"""
        if not self.is_built:
            raise ValueError("Sistema não construído. Execute build_all_scenarios() primeiro.")
        
        return self.vector_rag.query(question)
    
    def query_scenario_b(self, question: str, use_graph_expansion: bool = True) -> Dict[str, Any]:
        """Consulta usando Cenário B: Hybrid RAG"""
        if not self.is_built:
            raise ValueError("Sistema não construído. Execute build_all_scenarios() primeiro.")
        
        return self.hybrid_rag.query(question, use_graph_expansion)
    
    def query_scenario_c(self, question: str) -> Dict[str, Any]:
        """Consulta usando Cenário C: LLM-Only"""
        return self.llm_only.query(question)
    
    def compare_all_scenarios(self, question: str) -> Dict[str, Any]:
        """Executa a mesma pergunta em todos os cenários e compara"""
        logger.info(f"🔍 Comparando cenários para: {question}")
        
        results = {}
        
        # Cenário A
        try:
            results['scenario_a'] = self.query_scenario_a(question)
        except Exception as e:
            logger.error(f"❌ Erro Cenário A: {e}")
            results['scenario_a'] = {'error': str(e)}
        
        # Cenário B
        try:
            results['scenario_b'] = self.query_scenario_b(question)
        except Exception as e:
            logger.error(f"❌ Erro Cenário B: {e}")
            results['scenario_b'] = {'error': str(e)}
        
        # Cenário C
        try:
            results['scenario_c'] = self.query_scenario_c(question)
        except Exception as e:
            logger.error(f"❌ Erro Cenário C: {e}")
            results['scenario_c'] = {'error': str(e)}
        
        # Análise comparativa
        comparison = self._analyze_scenarios(results)
        
        return {
            'question': question,
            'results': results,
            'comparison': comparison
        }
    
    def _analyze_scenarios(self, results: Dict) -> Dict[str, Any]:
        """Analisa e compara resultados dos cenários"""
        analysis = {
            'response_times': {},
            'costs': {},
            'relevance_scores': {},
            'methods_used': {},
            'recommendations': []
        }
        
        for scenario, result in results.items():
            if 'error' not in result:
                analysis['response_times'][scenario] = result.get('response_time', 0)
                analysis['costs'][scenario] = result.get('estimated_cost_usd', 0)
                analysis['relevance_scores'][scenario] = result.get('relevance_score', 0)
                analysis['methods_used'][scenario] = result.get('method', 'unknown')
        
        # Recomendações baseadas nos resultados
        if analysis['response_times']:
            fastest = min(analysis['response_times'], key=analysis['response_times'].get)
            analysis['fastest_scenario'] = fastest
            
            cheapest = min(analysis['costs'], key=analysis['costs'].get)
            analysis['cheapest_scenario'] = cheapest
            
            most_relevant = max(analysis['relevance_scores'], key=analysis['relevance_scores'].get)
            analysis['most_relevant_scenario'] = most_relevant
        
        # Gera recomendações
        if analysis.get('cheapest_scenario') == 'scenario_a':
            analysis['recommendations'].append("Cenário A é mais econômico (sem custos de API)")
        
        if analysis.get('most_relevant_scenario') == 'scenario_b':
            analysis['recommendations'].append("Cenário B oferece maior precisão com raciocínio temporal")
        
        if analysis.get('fastest_scenario') == 'scenario_c':
            analysis['recommendations'].append("Cenário C é mais rápido (sem recuperação de documentos)")
        
        return analysis
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna status de todos os cenários"""
        return {
            'documents_loaded': len(self.documents),
            'system_built': self.is_built,
            'scenario_a': {
                'name': 'Vector-Only RAG',
                'ready': self.vector_rag.faiss_index is not None,
                'chunks': len(self.vector_rag.document_chunks),
                'device': self.vector_rag.device
            },
            'scenario_b': {
                'name': 'Hybrid RAG (Vector + Graph)',
                'ready': self.hybrid_rag.faiss_index is not None,
                'deus': len(self.hybrid_rag.deus),
                'graph_nodes': self.hybrid_rag.event_graph.number_of_nodes(),
                'graph_edges': self.hybrid_rag.event_graph.number_of_edges(),
                'device': self.hybrid_rag.device
            },
            'scenario_c': {
                'name': 'LLM-Only',
                'ready': True,
                'api_provider': self.llm_only.api_provider,
                'using_mock': self.llm_only.use_mock
            }
        }

# ==================== FUNÇÃO DE TESTE ====================

def test_multi_scenario_system():
    """Testa o sistema multi-cenário com dados de exemplo"""
    logger.info("🧪 Testando Sistema Multi-Cenário...")
    
    # Dados de exemplo (expandidos com foco temporal)
    sample_documents = [
        {
            "id": "doc_001",
            "title": "Normas de Ruído - NBR 10151:2019",
            "content": "O monitoramento de ruído em obras urbanas deve seguir as diretrizes da NBR 10151, revisão 2019. Os níveis de ruído não podem exceder 70 dB durante o período diurno (das 7h às 22h) e 60 dB durante o período noturno (das 22h às 7h) em áreas residenciais. É obrigatório o uso de medidores calibrados e registros horários. A medição deve ser feita no limite da propriedade mais próxima aos receptores sensíveis. Para obras que ultrapassem 60 dias, é necessário relatório mensal.",
            "category": "regulamentacao",
            "keywords": ["ruído", "NBR 10151", "medição", "limites", "obras urbanas", "horário"]
        },
        {
            "id": "doc_002",
            "title": "EPIs Obrigatórios - NR 6",
            "content": "Equipamentos de proteção individual (EPIs) obrigatórios em canteiros incluem: capacete de segurança classe A ou B, obrigatório durante todo o período de trabalho, óculos de proteção contra impactos para atividades com risco de projeção, luvas adequadas ao tipo de atividade (não usar próximo a máquinas rotativas), calçados de segurança com biqueira de aço, cintos de segurança para trabalho em altura acima de 2 metros. A empresa deve fornecer gratuitamente, treinar os trabalhadores sobre o uso correto e substituir quando necessário. Inspeção semanal obrigatória.",
            "category": "seguranca",
            "keywords": ["EPIs", "capacete", "óculos", "luvas", "calçados", "cintos", "treinamento", "inspeção"]
        },
        {
            "id": "doc_003",
            "title": "Controle de Qualidade do Concreto - NBR 12655",
            "content": "O controle de qualidade do concreto estrutural requer ensaios de resistência à compressão a cada 50m³ ou a cada dia de concretage, prevalecendo o menor valor. Os corpos de prova devem ser moldados no local da obra, curados em condições padronizadas por 28 dias. Ensaios intermediários aos 7 dias permitem avaliação prévia. Para obras de grande porte (acima de 500m³), ensaios adicionais de módulo de elasticidade são recomendados. Documentação completa deve ser mantida por no mínimo 5 anos após conclusão da obra.",
            "category": "qualidade",
            "keywords": ["controle qualidade", "concreto", "ensaios", "50m³", "28 dias", "resistência", "documentação"]
        }
    ]
    
    # Perguntas de teste com características temporais
    test_questions = [
        "Quais são os limites de ruído permitidos durante o dia e a noite?",
        "Com que frequência devem ser realizados os ensaios de concreto?",
        "Quais EPIs são obrigatórios e quando devem ser inspecionados?",
        "Que documentação deve ser mantida após a conclusão da obra?"
    ]
    
    try:
        # Inicializa sistema
        system = MultiScenarioSystem()
        
        # Carrega documentos
        system.load_documents(sample_documents)
        
        # Constrói todos os cenários
        system.build_all_scenarios()
        
        # Status do sistema
        status = system.get_system_status()
        logger.info(f"📊 Status do sistema: {status}")
        
        # Testa cada pergunta em todos os cenários
        all_results = []
        
        for question in test_questions:
            logger.info(f"\n❓ Pergunta: {question}")
            
            comparison_result = system.compare_all_scenarios(question)
            all_results.append(comparison_result)
            
            # Log resumo dos resultados
            results = comparison_result['results']
            for scenario, result in results.items():
                if 'error' not in result:
                    time_taken = result.get('response_time', 0)
                    cost = result.get('estimated_cost_usd', 0)
                    relevance = result.get('relevance_score', 0)
                    logger.info(f"  {scenario}: {time_taken:.2f}s, ${cost:.4f}, relevância: {relevance:.3f}")
                else:
                    logger.error(f"  {scenario}: ERRO - {result['error']}")
        
        # Salva resultados
        results_file = Path("data/evaluation/multi_scenario_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Teste concluído! Resultados salvos em {results_file}")
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        return []

if __name__ == "__main__":
    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Executa teste
    test_multi_scenario_system()
    