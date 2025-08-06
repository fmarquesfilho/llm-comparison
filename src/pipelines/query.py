"""
Pipeline de processamento de consultas
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..core.router import QueryRouter

logger = logging.getLogger(__name__)

class QueryProcessingPipeline:
    """
    Pipeline para processamento e otimização de consultas
    """
    
    def __init__(self):
        self.router = QueryRouter()
        
        # Templates de normalização de consultas
        self.query_templates = {
            'violation_check': [
                'houve violações', 'violations detected', 'normas de ruído',
                'regulamentações', 'compliance issues'
            ],
            'pattern_analysis': [
                'padrão de ruído', 'noise pattern', 'tendência',
                'comportamento temporal', 'análise temporal'
            ],
            'count_query': [
                'quantos eventos', 'how many', 'número de',
                'count', 'total de eventos'
            ],
            'peak_analysis': [
                'pico de ruído', 'peak noise', 'máximo',
                'intensidade máxima', 'loudest event'
            ]
        }
        
        logger.info("QueryProcessingPipeline initialized")
    
    def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa consulta completa com normalização e roteamento
        
        Args:
            query: Consulta em linguagem natural
            context: Contexto adicional (timestamp, filtros, etc.)
            
        Returns:
            Dict com consulta processada e metadados
        """
        try:
            # Normaliza consulta
            normalized_query = self._normalize_query(query)
            
            # Extrai intenção
            query_intent = self._extract_intent(normalized_query)
            
            # Roteamento inteligente
            routing_result = self.router.route_query(normalized_query)
            
            # Extrai parâmetros temporais
            temporal_params = self._extract_temporal_parameters(query, context)
            
            # Extrai filtros
            filters = self._extract_filters(query)
            
            processed_query = {
                'original_query': query,
                'normalized_query': normalized_query,
                'intent': query_intent,
                'routing': routing_result,
                'temporal_parameters': temporal_params,
                'filters': filters,
                'processing_timestamp': datetime.now().isoformat(),
                'confidence_score': routing_result.get('confidence', 0.7)
            }
            
            logger.info(f"Processed query with intent: {query_intent}")
            return processed_query
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'original_query': query,
                'processing_timestamp': datetime.now().isoformat()
            }
    
    def _normalize_query(self, query: str) -> str:
        """Normaliza consulta removendo ruídos e padronizando"""
        normalized = query.lower().strip()
        
        # Remove pontuação desnecessária
        import re
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove espaços múltiplos
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Expansões de abreviações comuns
        expansions = {
            'db': 'decibéis',
            'h': 'horas',
            'min': 'minutos',
            'seg': 'segundos'
        }
        
        for abbrev, full in expansions.items():
            normalized = normalized.replace(f' {abbrev} ', f' {full} ')
        
        return normalized
    
    def _extract_intent(self, query: str) -> str:
        """Extrai intenção principal da consulta"""
        for intent, keywords in self.query_templates.items():
            if any(keyword in query for keyword in keywords):
                return intent
        
        # Análise baseada em palavras-chave específicas
        if any(word in query for word in ['quantos', 'how many', 'count', 'número']):
            return 'count_query'
        elif any(word in query for word in ['violação', 'violation', 'problema']):
            return 'violation_check'
        elif any(word in query for word in ['padrão', 'pattern', 'tendência']):
            return 'pattern_analysis'
        elif any(word in query for word in ['pico', 'peak', 'máximo', 'maximum']):
            return 'peak_analysis'
        else:
            return 'general_query'