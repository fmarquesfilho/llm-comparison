"""
Pipeline de processamento de consultas
"""

import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

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
    
    def _extract_temporal_parameters(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extrai parâmetros temporais da consulta"""
        temporal_params = {
            'reference_time': datetime.now(),
            'time_range': None,
            'relative_period': None
        }
        
        # Usa contexto se fornecido
        if context and 'reference_time' in context:
            temporal_params['reference_time'] = context['reference_time']
        
        query_lower = query.lower()
        
        # Detecta períodos relativos
        if 'última semana' in query_lower or 'last week' in query_lower:
            temporal_params['relative_period'] = 'last_week'
            temporal_params['time_range'] = {
                'start': temporal_params['reference_time'] - timedelta(weeks=1),
                'end': temporal_params['reference_time']
            }
        elif 'últimos 3 dias' in query_lower or 'last 3 days' in query_lower:
            temporal_params['relative_period'] = 'last_3_days'
            temporal_params['time_range'] = {
                'start': temporal_params['reference_time'] - timedelta(days=3),
                'end': temporal_params['reference_time']
            }
        elif 'ontem' in query_lower or 'yesterday' in query_lower:
            temporal_params['relative_period'] = 'yesterday'
            yesterday_start = temporal_params['reference_time'].replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            yesterday_end = yesterday_start + timedelta(days=1)
            temporal_params['time_range'] = {
                'start': yesterday_start,
                'end': yesterday_end
            }
        elif 'terça-feira passada' in query_lower or 'last tuesday' in query_lower:
            # Calcula a terça-feira da semana passada
            days_back = (temporal_params['reference_time'].weekday() - 1) % 7 + 7  # Terça = 1
            tuesday_start = temporal_params['reference_time'] - timedelta(days=days_back)
            tuesday_start = tuesday_start.replace(hour=0, minute=0, second=0, microsecond=0)
            tuesday_end = tuesday_start + timedelta(days=1)
            temporal_params['relative_period'] = 'last_tuesday'
            temporal_params['time_range'] = {
                'start': tuesday_start,
                'end': tuesday_end
            }
        elif 'horário comercial' in query_lower or 'business hours' in query_lower:
            temporal_params['business_hours_filter'] = True
        
        return temporal_params
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extrai filtros específicos da consulta"""
        filters = {}
        query_lower = query.lower()
        
        # Filtros por tipo de evento
        equipment_types = ['martelo', 'serra', 'betoneira', 'britadeira', 'guindaste', 'caminhão']
        for equipment in equipment_types:
            if equipment in query_lower:
                if 'event_types' not in filters:
                    filters['event_types'] = []
                filters['event_types'].append(equipment)
        
        # Filtros por intensidade
        if any(word in query_lower for word in ['alto', 'high', 'forte', 'loud']):
            filters['min_loudness'] = 70
        elif any(word in query_lower for word in ['muito alto', 'very high', 'extremo']):
            filters['min_loudness'] = 90
        
        # Filtros por violação
        if any(word in query_lower for word in ['violação', 'violation', 'irregular']):
            filters['violations_only'] = True
        
        # Filtros por sensor
        sensor_pattern = re.search(r'sensor[_\s]?([a-zA-Z0-9]+)', query_lower)
        if sensor_pattern:
            filters['sensor_id'] = f"sensor_{sensor_pattern.group(1)}"
        
        # Filtros por localização
        locations = ['norte', 'sul', 'central', 'acesso']
        for location in locations:
            if location in query_lower:
                if 'locations' not in filters:
                    filters['locations'] = []
                filters['locations'].append(f"area_{location}")
        
        return filters
    
    def optimize_query_for_scenario(self, processed_query: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """
        Otimiza consulta processada para cenário específico
        
        Args:
            processed_query: Consulta já processada
            scenario: 'vector_rag', 'graph_rag', ou 'llm_only'
            
        Returns:
            Dict com parâmetros otimizados para o cenário
        """
        base_params = {
            'query': processed_query['normalized_query'],
            'original_query': processed_query['original_query'],
            'intent': processed_query['intent'],
            'filters': processed_query.get('filters', {})
        }
        
        if scenario == 'vector_rag':
            # Otimizações para Vector RAG
            base_params.update({
                'top_k': self._get_optimal_k_for_intent(processed_query['intent']),
                'similarity_threshold': 0.6,
                'temporal_weight': 0.3 if processed_query.get('temporal_parameters', {}).get('time_range') else 0.1
            })
            
        elif scenario == 'graph_rag':
            # Otimizações para Graph RAG
            base_params.update({
                'top_k': self._get_optimal_k_for_intent(processed_query['intent']) + 3,  # Mais eventos para análise temporal
                'enable_multi_hop': processed_query['intent'] in ['pattern_analysis', 'correlation'],
                'temporal_reasoning': True,
                'graph_traversal_depth': 2 if processed_query['intent'] == 'pattern_analysis' else 1
            })
            
        elif scenario == 'llm_only':
            # Para LLM-Only, fornece mais contexto na consulta
            enhanced_query = self._enhance_query_for_llm(processed_query)
            base_params.update({
                'enhanced_query': enhanced_query,
                'context_provided': True
            })
        
        # Adiciona parâmetros temporais se disponíveis
        if 'temporal_parameters' in processed_query:
            base_params['temporal_parameters'] = processed_query['temporal_parameters']
        
        return base_params
    
    def _get_optimal_k_for_intent(self, intent: str) -> int:
        """Retorna número ótimo de resultados baseado na intenção"""
        intent_k_mapping = {
            'count_query': 20,  # Mais resultados para contagem precisa
            'pattern_analysis': 15,  # Muitos eventos para identificar padrões
            'violation_check': 10,  # Moderado para verificar violações
            'peak_analysis': 8,  # Foco nos eventos mais intensos
            'general_query': 5  # Padrão para consultas gerais
        }
        return intent_k_mapping.get(intent, 5)
    
    def _enhance_query_for_llm(self, processed_query: Dict[str, Any]) -> str:
        """Enriquece consulta para LLM-Only com contexto adicional"""
        base_query = processed_query['original_query']
        enhancements = []
        
        # Adiciona contexto temporal
        if 'temporal_parameters' in processed_query:
            temp_params = processed_query['temporal_parameters']
            if temp_params.get('relative_period'):
                enhancements.append(f"Período de referência: {temp_params['relative_period']}")
        
        # Adiciona contexto de filtros
        filters = processed_query.get('filters', {})
        if filters.get('event_types'):
            enhancements.append(f"Tipos de equipamentos: {', '.join(filters['event_types'])}")
        
        if filters.get('violations_only'):
            enhancements.append("Foco especificamente em violações de normas de ruído")
        
        # Adiciona contexto do domínio
        domain_context = "Contexto: Análise de eventos sonoros em canteiro de obras de construção civil, considerando regulamentações de ruído urbano."
        
        if enhancements:
            enhanced_query = f"{domain_context} {base_query} ({'; '.join(enhancements)})"
        else:
            enhanced_query = f"{domain_context} {base_query}"
        
        return enhanced_query
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Valida consulta e retorna feedback de qualidade
        
        Returns:
            Dict com validação e sugestões de melhoria
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'suggestions': [],
            'complexity_score': 0.5
        }
        
        if not query or len(query.strip()) < 3:
            validation['is_valid'] = False
            validation['issues'].append("Consulta muito curta ou vazia")
            return validation
        
        query_lower = query.lower()
        
        # Verifica clareza temporal
        temporal_indicators = ['quando', 'ontem', 'última', 'durante', 'período']
        has_temporal = any(word in query_lower for word in temporal_indicators)
        
        if not has_temporal and any(word in query_lower for word in ['padrão', 'tendência', 'análise']):
            validation['suggestions'].append("Considere especificar um período temporal para análises mais precisas")
        
        # Verifica especificidade
        if len(query.split()) < 4:
            validation['suggestions'].append("Consultas mais específicas tendem a produzir melhores resultados")
            validation['complexity_score'] = 0.3
        elif len(query.split()) > 15:
            validation['complexity_score'] = 0.8
        
        # Verifica presença de critérios específicos
        specific_terms = ['tipo', 'intensidade', 'violação', 'equipamento', 'sensor']
        specificity_count = sum(1 for term in specific_terms if term in query_lower)
        validation['complexity_score'] += specificity_count * 0.1
        
        return validation