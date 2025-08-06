"""
Query Router para direcionamento de consultas entre os diferentes cenários RAG
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Router inteligente para direcionamento de consultas entre Vector RAG,
    Hybrid RAG (DyG-RAG), e LLM-Only baseado na complexidade da consulta.
    """
    
    def __init__(self):
        # Palavras-chave para classificação de complexidade
        self.simple_keywords = [
            'quantos', 'quanto', 'count', 'número', 'how many', 'lista', 'list'
        ]
        
        self.complex_keywords = [
            'padrão', 'pattern', 'correlação', 'correlation', 'porque', 'why',
            'causa', 'cause', 'tendência', 'trend', 'análise', 'analysis',
            'relacionamento', 'relationship', 'sequência', 'sequence'
        ]
        
        self.temporal_keywords = [
            'temporal', 'tempo', 'time', 'cronológico', 'chronological',
            'horário', 'schedule', 'período', 'period', 'duração', 'duration'
        ]
        
        logger.info("QueryRouter initialized")
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Roteia consulta para o método mais apropriado.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Dict com recomendação de roteamento e justificativa
        """
        query_lower = query.lower()
        
        # Análise de palavras-chave
        has_simple = any(kw in query_lower for kw in self.simple_keywords)
        has_complex = any(kw in query_lower for kw in self.complex_keywords)
        has_temporal = any(kw in query_lower for kw in self.temporal_keywords)
        
        # Análise de comprimento (consultas mais longas tendem a ser complexas)
        query_length = len(query.split())
        
        # Lógica de roteamento
        if has_simple and not has_complex:
            recommended_scenario = 'scenario_a'  # Vector RAG
            complexity = 'simple'
            justification = "Consulta simples de contagem/listagem - Vector RAG é suficiente e eficiente"
            
        elif has_complex or has_temporal or query_length > 15:
            recommended_scenario = 'scenario_b'  # Hybrid RAG (DyG-RAG)
            complexity = 'complex'
            justification = "Consulta complexa requer análise temporal e raciocínio - DyG-RAG oferece melhor precisão"
            
        elif query_length < 5 and not has_complex:
            recommended_scenario = 'scenario_c'  # LLM-Only
            complexity = 'general'
            justification = "Consulta geral sem necessidade de dados específicos - LLM-Only é adequado"
            
        else:
            recommended_scenario = 'scenario_a'  # Vector RAG (default)
            complexity = 'medium'
            justification = "Consulta média - Vector RAG oferece bom equilíbrio"
        
        # Detecta tipo de pergunta
        question_type = self._classify_question_type(query_lower)
        
        routing_result = {
            'recommended_scenario': recommended_scenario,
            'complexity': complexity,
            'question_type': question_type,
            'justification': justification,
            'confidence': self._calculate_confidence(has_simple, has_complex, has_temporal),
            'alternative_scenarios': self._suggest_alternatives(recommended_scenario),
            'keywords_detected': {
                'simple': has_simple,
                'complex': has_complex,
                'temporal': has_temporal
            }
        }
        
        logger.debug(f"Routed query to {recommended_scenario}: {justification}")
        return routing_result
    
    def _classify_question_type(self, query_lower: str) -> str:
        """Classifica o tipo de pergunta"""
        if any(word in query_lower for word in ['quantos', 'quanto', 'count']):
            return 'count'
        elif any(word in query_lower for word in ['violação', 'violation', 'problema']):
            return 'violation_detection'
        elif any(word in query_lower for word in ['padrão', 'pattern']):
            return 'pattern_analysis'
        elif any(word in query_lower for word in ['porque', 'why', 'causa']):
            return 'causal_analysis'
        elif any(word in query_lower for word in ['correlação', 'correlation']):
            return 'correlation_analysis'
        else:
            return 'general_query'
    
    def _calculate_confidence(self, has_simple: bool, has_complex: bool, has_temporal: bool) -> float:
        """Calcula confiança na decisão de roteamento"""
        base_confidence = 0.7
        
        # Aumenta confiança se há indicadores claros
        if has_simple and not has_complex:
            base_confidence += 0.2
        elif has_complex or has_temporal:
            base_confidence += 0.2
        
        # Diminui se há sinais mistos
        if has_simple and has_complex:
            base_confidence -= 0.1
            
        return min(1.0, max(0.5, base_confidence))
    
    def _suggest_alternatives(self, recommended_scenario: str) -> list:
        """Sugere cenários alternativos"""
        alternatives = []
        
        if recommended_scenario == 'scenario_a':
            alternatives = ['scenario_b', 'scenario_c']
        elif recommended_scenario == 'scenario_b':
            alternatives = ['scenario_a', 'scenario_c']
        else:  # scenario_c
            alternatives = ['scenario_a', 'scenario_b']
            
        return alternatives
    
    def explain_routing_decision(self, routing_result: Dict[str, Any]) -> str:
        """Gera explicação textual da decisão de roteamento"""
        scenario_names = {
            'scenario_a': 'Vector RAG',
            'scenario_b': 'Hybrid RAG (DyG-RAG)',
            'scenario_c': 'LLM-Only'
        }
        
        recommended_name = scenario_names.get(routing_result['recommended_scenario'], 'Unknown')
        confidence = routing_result['confidence']
        justification = routing_result['justification']
        
        explanation = f"""
        🎯 **Recomendação de Roteamento**
        
        **Cenário Recomendado:** {recommended_name}
        **Confiança:** {confidence:.1%}
        **Complexidade:** {routing_result['complexity'].title()}
        **Tipo:** {routing_result['question_type'].replace('_', ' ').title()}
        
        **Justificativa:** {justification}
        
        **Palavras-chave detectadas:**
        - Simples: {'✅' if routing_result['keywords_detected']['simple'] else '❌'}
        - Complexa: {'✅' if routing_result['keywords_detected']['complex'] else '❌'}  
        - Temporal: {'✅' if routing_result['keywords_detected']['temporal'] else '❌'}
        """
        
        return explanation.strip()