
# src/multi_scenario_system.py
import logging
import time
import re
import json
import os
import shutil
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import kuzu

logger = logging.getLogger(__name__)

# --- Data Classes ---
@dataclass
class TemporalAnchor:
    timestamp: Optional[datetime] = None
    time_expression: Optional[str] = None
    temporal_type: str = "relative"
    confidence: float = 1.0

@dataclass
class DynamicEventUnit:
    id: str
    content: str
    entities: List[str]
    temporal_anchor: TemporalAnchor
    event_type: str
    source_document: str
    embedding: Optional[np.ndarray] = None
    keywords: List[str] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []

# --- Extractors (Unaltered) ---
class TemporalExtractor:
    """Extrator de informações temporais específico para construção civil"""
    def __init__(self):
        self.temporal_patterns = {
            'time_schedules': [r'das? (\d{1,2})h? às? (\d{1,2})h?', r'período diurno', r'período noturno'],
            'durations': [r'por (\d+) (dias?|semanas?|meses?)', r'a cada (\d+)m³'],
            'thresholds': [r'superior a (\d+)m²', r'não pode exceder (\d+) dB'],
            'regulatory_dates': [r'NBR \d+/(\d{4})', r'aprovado em (\d{4})'],
            'sequence_markers': [r'antes de', r'após', 'simultaneamente']
        }
    def extract_temporal_info(self, text: str) -> TemporalAnchor:
        text_lower = text.lower()
        for p_type, patterns in self.temporal_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if p_type == 'regulatory_dates':
                        try:
                            return TemporalAnchor(timestamp=datetime(int(match.group(1)), 1, 1), time_expression=match.group(0), temporal_type="absolute")
                        except: continue
                    return TemporalAnchor(time_expression=match.group(0), temporal_type=p_type)
        return TemporalAnchor(time_expression="contexto_geral", temporal_type="relative")

class EntityExtractor:
    """Extrator de entidades específicas do domínio de construção civil"""
    def __init__(self):
        self.entity_patterns = {
            'regulations': [r'NBR \d+', r'NR-?\d+'],
            'measurements': [r'\d+\s*dB', r'\d+\s*m[²³]?'],
            'equipment': [r'capacete', r'luvas', r'EPI[s]?'],
            'processes': [r'ensaio', r'medição de ruído', r'concretagem']
        }
    def extract_entities(self, text: str) -> List[str]:
        entities = []
        for category, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                entities.extend([f"{category}:{match}" for match in matches])
        return list(dict.fromkeys(entities))

# ==================== CENÁRIO A: RAG VETORIAL APENAS (Unaltered) ====================
class VectorOnlyRAG:
    def __init__(self, config=None):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faiss_index = None
        self.document_chunks = []
        logger.info("✅ Vector-Only RAG System inicializado")

    def chunk_documents(self, documents: List[Dict], chunk_size: int = 512, overlap: int = 50) -> List[Dict]:
        chunks = []
        for doc in documents:
            content = doc.get('content', '')
            words = content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text.strip()) > 50:
                    chunks.append({'id': f"{doc['id']}_chunk_{i//(chunk_size-overlap)}", 'content': chunk_text, 'source_doc': doc['id'], 'title': doc.get('title', '')})
        return chunks

    def build_index(self, documents: List[Dict]):
        logger.info("🔧 Construindo índice vetorial...")
        self.document_chunks = self.chunk_documents(documents)
        if not self.document_chunks: raise ValueError("Nenhum chunk criado.")
        contents = [chunk['content'] for chunk in self.document_chunks]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        logger.info(f"✅ Índice vetorial construído: {len(self.document_chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.faiss_index: return []
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k, len(self.document_chunks)))
        retrieved_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.document_chunks[idx].copy()
            chunk['relevance_score'] = 1.0 / (1.0 + float(score))
            retrieved_chunks.append(chunk)
        return retrieved_chunks

    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            retrieved_chunks = self.retrieve(question)
            answer = "Nenhuma informação relevante encontrada."
            if retrieved_chunks:
                best_chunk = retrieved_chunks[0]
                answer = f"Baseado no documento '{best_chunk['title']}':\n\n{best_chunk['content']}"
            return {'answer': answer, 'retrieved_chunks': retrieved_chunks, 'response_time': time.time() - start_time, 'relevance_score': retrieved_chunks[0]['relevance_score'] if retrieved_chunks else 0, 'method': 'vector_only_rag'}
        except Exception as e:
            logger.error(f"❌ Erro na consulta Vector RAG: {e}")
            return {'answer': f"Erro: {e}", 'response_time': time.time() - start_time, 'method': 'vector_only_rag'}


# ==================== CENÁRIO B: RAG HÍBRIDO (VETORIAL + KÙZU) ====================
class HybridRAG:
    """Cenário B: RAG híbrido com busca vetorial e grafo de conhecimento Kùzu"""
    def __init__(self, config=None, db_path="data/kuzu_db"):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.temporal_extractor = TemporalExtractor()
        self.entity_extractor = EntityExtractor()
        self.deus: List[DynamicEventUnit] = []
        self.faiss_index = None
        self.deu_index_map = {}
        
        self.db_path = Path(db_path)
        self.db = None
        self.conn = None
        
        logger.info("✅ Hybrid RAG System inicializado")

    def _init_kuzu_database(self):
        """Inicializa o banco de dados Kùzu e define o esquema."""
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        
        logger.info("🔧 Inicializando esquema do Kùzu DB...")
        self.conn.execute("CREATE NODE TABLE DEU(id STRING, content STRING, event_type STRING, time_expression STRING, PRIMARY KEY (id))")
        self.conn.execute("CREATE NODE TABLE Entity(id STRING, type STRING, PRIMARY KEY (id))")
        self.conn.execute("CREATE REL TABLE HAS_ENTITY(FROM DEU TO Entity)")
        self.conn.execute("CREATE REL TABLE RELATED_TO(FROM DEU TO DEU, weight FLOAT, type STRING)")
        logger.info("✅ Esquema do Kùzu DB criado.")

    def create_dynamic_event_units(self, documents: List[Dict]) -> List[DynamicEventUnit]:
        """Converte documentos em Dynamic Event Units"""
        deus = []
        for doc in documents:
            sentences = re.split(r'\. |\n', doc.get('content', ''))
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20: continue
                temporal_anchor = self.temporal_extractor.extract_temporal_info(sentence)
                entities = self.entity_extractor.extract_entities(sentence)
                deu = DynamicEventUnit(
                    id=f"{doc['id']}_evt_{i}",
                    content=sentence,
                    entities=entities,
                    temporal_anchor=temporal_anchor,
                    event_type="procedure", # Simple classification
                    source_document=doc['id']
                )
                deus.append(deu)
        logger.info(f"✅ Criados {len(deus)} Dynamic Event Units.")
        return deus
    
    def _populate_kuzu_graph(self):
        """Popula o grafo Kùzu com DEUs, Entidades e Relacionamentos."""
        logger.info("🔧 Populando o grafo Kùzu...")
        
        # Inserir DEUs e Entidades
        all_entities = {}
        for deu in self.deus:
            self.conn.execute("CREATE (d:DEU {id: $id, content: $content, event_type: $event_type, time_expression: $time_expr})",
                             parameters={'id': deu.id, 'content': deu.content, 'event_type': deu.event_type, 'time_expr': deu.temporal_anchor.time_expression or ''})
            for entity_str in deu.entities:
                entity_type, entity_id = entity_str.split(":", 1)
                if entity_id not in all_entities:
                    all_entities[entity_id] = entity_type
                    self.conn.execute("CREATE (e:Entity {id: $id, type: $type})", parameters={'id': entity_id, 'type': entity_type})
                self.conn.execute("MATCH (d:DEU), (e:Entity) WHERE d.id = $deu_id AND e.id = $entity_id CREATE (d)-[:HAS_ENTITY]->(e)",
                                 parameters={'deu_id': deu.id, 'entity_id': entity_id})

        # Inserir relacionamentos RELATED_TO entre DEUs
        for i, deu1 in enumerate(self.deus):
            for j, deu2 in enumerate(self.deus):
                if i >= j: continue
                
                # Simples similaridade de entidades para criar a aresta
                common_entities = set(deu1.entities) & set(deu2.entities)
                if len(common_entities) > 0:
                    weight = len(common_entities) / (len(set(deu1.entities) | set(deu2.entities)) or 1)
                    if weight > 0.1:
                         self.conn.execute("""
                            MATCH (d1:DEU), (d2:DEU) WHERE d1.id = $id1 AND d2.id = $id2
                            CREATE (d1)-[:RELATED_TO {weight: $w, type: 'entity'}]->(d2)
                         """, parameters={'id1': deu1.id, 'id2': deu2.id, 'w': weight})
        
        graph_summary = self.conn.execute("MATCH (n) RETURN count(n)").get_next()[0]
        logger.info(f"✅ Grafo Kùzu populado com {graph_summary} nós.")

    def build_faiss_index(self, deus: List[DynamicEventUnit]):
        """Constrói índice FAISS para busca vetorial inicial de DEUs"""
        if not deus: return
        contents = [deu.content for deu in deus]
        embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
        for deu, emb in zip(deus, embeddings):
            deu.embedding = emb
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        self.deu_index_map = {i: deu.id for i, deu in enumerate(deus)}
        logger.info(f"✅ Índice FAISS de DEUs construído: {len(deus)} DEUs.")

    def build_system(self, documents: List[Dict]):
        """Constrói o sistema híbrido completo."""
        logger.info("🔧 Construindo sistema híbrido...")
        self.deus = self.create_dynamic_event_units(documents)
        self._init_kuzu_database()
        self._populate_kuzu_graph()
        self.build_faiss_index(self.deus)
        logger.info("✅ Sistema híbrido construído!")

    def retrieve_seed_events(self, query: str, top_k: int = 3) -> List[DynamicEventUnit]:
        """Recupera DEUs semente usando busca vetorial."""
        if not self.faiss_index: return []
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.faiss_index.search(query_embedding.astype('float32'), min(top_k, len(self.deus)))
        deu_map = {deu.id: deu for deu in self.deus}
        return [deu_map[self.deu_index_map[idx]] for idx in indices[0] if idx in self.deu_index_map]

    def expand_through_kuzu_graph(self, seed_deu_ids: List[str], max_hops: int = 1) -> List[DynamicEventUnit]:
        """Expande através do grafo Kùzu para encontrar eventos relacionados."""
        if not self.conn: return []
        
        cypher_query = """
            MATCH (seed:DEU)-[r:RELATED_TO]-(neighbor:DEU)
            WHERE seed.id IN $seed_ids AND r.weight > 0.2
            RETURN DISTINCT neighbor.id, neighbor.content, neighbor.event_type, neighbor.time_expression
            """
        query_result = self.conn.execute(cypher_query, parameters={'seed_ids': seed_deu_ids})
        
        expanded_deus = []
        deu_map = {deu.id: deu for deu in self.deus} # Para recuperar o objeto completo
        
        while query_result.has_next():
            row = query_result.get_next()
            deu_id = row[0]
            if deu_id in deu_map:
                expanded_deus.append(deu_map[deu_id])
        
        return expanded_deus
    
    def time_chain_of_thought(self, query: str, events: List[DynamicEventUnit]) -> str:
        """Gera resposta usando Time Chain-of-Thought."""
        if not events: return "Nenhum evento relevante encontrado."
        
        # Simplesmente junta os conteúdos por agora
        response_parts = [f"Considerando a pergunta '{query}':"]
        for i, event in enumerate(events[:5], 1): # Limita a 5 eventos para ser conciso
            temporal_info = f"[{event.temporal_anchor.time_expression}]" if event.temporal_anchor.time_expression else ""
            response_parts.append(f"\n{i}. {event.content} {temporal_info}")
        
        return "".join(response_parts)

    def query(self, question: str, use_graph_expansion: bool = True) -> Dict[str, Any]:
        """Interface principal para consultas híbridas."""
        start_time = time.time()
        try:
            seed_events = self.retrieve_seed_events(question, top_k=3)
            
            final_events = seed_events
            if use_graph_expansion and seed_events:
                seed_ids = [deu.id for deu in seed_events]
                expanded_events = self.expand_through_kuzu_graph(seed_ids)
                # Combina e remove duplicatas
                final_events_map = {deu.id: deu for deu in seed_events}
                final_events_map.update({deu.id: deu for deu in expanded_events})
                final_events = list(final_events_map.values())

            answer = self.time_chain_of_thought(question, final_events)
            
            # Cálculo de relevância simples
            avg_relevance = np.mean([1] * len(final_events)) if final_events else 0

            return {
                'answer': answer,
                'retrieved_events': [e.id for e in final_events],
                'response_time': time.time() - start_time,
                'relevance_score': float(avg_relevance),
                'method': 'hybrid_rag',
                'graph_expansion_used': use_graph_expansion
            }
        except Exception as e:
            logger.error(f"❌ Erro na consulta Hybrid RAG: {e}", exc_info=True)
            return {'answer': f"Erro: {e}", 'response_time': time.time() - start_time, 'method': 'hybrid_rag'}

# ==================== CENÁRIO C: LLM APENAS (Unaltered, but simplified) ====================
class LLMOnly:
    def __init__(self, api_provider: str = "openai"):
        self.use_mock = True # Forçando mock para simplicidade e custo
        logger.info("✅ LLM-Only System inicializado (usando mock)")

    def _query_mock(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        time.sleep(0.5) # Simula latência
        answer = "Resposta genérica do mock LLM."
        if 'ruído' in question.lower():
            answer = "Segundo a NBR 10151, o limite diurno é 70 dB e noturno 60 dB."
        elif 'epi' in question.lower():
            answer = "EPIs obrigatórios incluem capacete, luvas e botas. A inspeção deve ser diária."
        return {'answer': answer, 'response_time': time.time() - start_time, 'estimated_cost_usd': 0.0005}

    def query(self, question: str) -> Dict[str, Any]:
        result = self._query_mock(question)
        result.update({'question': question, 'method': 'llm_only', 'relevance_score': 0.8})
        return result


# ==================== SISTEMA MULTI-CENÁRIO (Unaltered) ====================
class MultiScenarioSystem:
    def __init__(self, config=None):
        self.vector_rag = VectorOnlyRAG(config)
        self.hybrid_rag = HybridRAG(config)
        self.llm_only = LLMOnly()
        self.documents = []
        self.is_built = False
        logger.info("✅ Multi-Scenario System inicializado")

    def load_documents(self, documents: List[Dict]):
        self.documents = documents
        logger.info(f"📄 Carregados {len(documents)} documentos")
    
    def build_all_scenarios(self):
        if not self.documents: raise ValueError("Nenhum documento carregado")
        logger.info("🏗️ Construindo todos os cenários...")
        self.vector_rag.build_index(self.documents)
        self.hybrid_rag.build_system(self.documents)
        self.is_built = True
        logger.info("✅ Todos os cenários construídos!")

    def query_scenario_a(self, question: str) -> Dict[str, Any]:
        return self.vector_rag.query(question)

    def query_scenario_b(self, question: str) -> Dict[str, Any]:
        return self.hybrid_rag.query(question)

    def query_scenario_c(self, question: str) -> Dict[str, Any]:
        return self.llm_only.query(question)
    
    def compare_all_scenarios(self, question: str) -> Dict[str, Any]:
        logger.info(f"🔍 Comparando cenários para: {question}")
        results = {
            'scenario_a': self.query_scenario_a(question),
            'scenario_b': self.query_scenario_b(question),
            'scenario_c': self.query_scenario_c(question)
        }
        return {'question': question, 'results': results}
