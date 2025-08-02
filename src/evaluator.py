# src/evaluator.py
import json
import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class SimpleEvaluator:
    """Avaliador simplificado para MVP com métricas essenciais"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_response_time_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calcula métricas de tempo de resposta"""
        if not results:
            return {}
        
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        
        if not response_times:
            return {}
        
        return {
            'avg_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'std_response_time': np.std(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'under_3s_rate': sum(1 for t in response_times if t < 3.0) / len(response_times)
        }
    
    def calculate_cost_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calcula métricas de custo"""
        if not results:
            return {}
        
        costs = [r.get('estimated_cost_usd', 0) for r in results if 'estimated_cost_usd' in r]
        
        if not costs:
            return {}
        
        total_cost = sum(costs)
        num_queries = len(costs)
        
        return {
            'total_cost_usd': total_cost,
            'avg_cost_per_query': total_cost / num_queries if num_queries > 0 else 0,
            'cost_per_1k_queries': (total_cost / num_queries) * 1000 if num_queries > 0 else 0,
            'min_cost_per_query': min(costs) if costs else 0,
            'max_cost_per_query': max(costs) if costs else 0
        }
    
    def calculate_relevance_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calcula métricas de relevância (para RAG)"""
        if not results:
            return {}
        
        relevance_scores = []
        num_retrieved_docs = []
        
        for r in results:
            if 'relevance_score' in r:
                relevance_scores.append(r['relevance_score'])
            if 'retrieved_docs' in r:
                num_retrieved_docs.append(r['retrieved_docs'])
        
        metrics = {}
        
        if relevance_scores:
            metrics.update({
                'avg_relevance_score': np.mean(relevance_scores),
                'median_relevance_score': np.median(relevance_scores),
                'min_relevance_score': np.min(relevance_scores),
                'max_relevance_score': np.max(relevance_scores),
                'high_relevance_rate': sum(1 for s in relevance_scores if s > 0.7) / len(relevance_scores),
                'low_relevance_rate': sum(1 for s in relevance_scores if s < 0.3) / len(relevance_scores)
            })
        
        if num_retrieved_docs:
            metrics.update({
                'avg_retrieved_docs': np.mean(num_retrieved_docs),
                'total_retrievals': sum(num_retrieved_docs)
            })
        
        return metrics
    
    def calculate_basic_rouge_l(self, prediction: str, reference: str) -> float:
        """Implementação básica de ROUGE-L sem bibliotecas extras"""
        if not prediction or not reference:
            return 0.0
        
        # Tokenização simples
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Calcula LCS (Longest Common Subsequence) simplificado
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if not common_tokens:
            return 0.0
        
        # ROUGE-L simplificado baseado em tokens comuns
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def evaluate_answer_quality(self, results: List[Dict], 
                               expected_concepts: Dict[str, List[str]] = None) -> Dict[str, float]:
        """Avalia qualidade das respostas de forma básica"""
        if not results:
            return {}
        
        quality_scores = []
        concept_coverage_scores = []
        
        for result in results:
            answer = result.get('answer', '')
            question_id = result.get('question_id', '')
            
            # Score básico de qualidade
            quality_score = 0.0
            
            # Verifica se resposta não é erro
            if not answer.startswith('❌') and len(answer) > 20:
                quality_score += 0.3
            
            # Verifica se resposta tem tamanho adequado
            if 50 <= len(answer) <= 500:
                quality_score += 0.2
            
            # Verifica se não é resposta genérica
            generic_phrases = ['não encontrei', 'não sei', 'sem informação']
            if not any(phrase in answer.lower() for phrase in generic_phrases):
                quality_score += 0.3
            
            # Verifica cobertura de conceitos esperados
            concept_score = 0.0
            if expected_concepts and question_id in expected_concepts:
                expected = expected_concepts[question_id]
                found_concepts = sum(1 for concept in expected 
                                   if concept.lower() in answer.lower())
                concept_score = found_concepts / len(expected) if expected else 0
                quality_score += concept_score * 0.2
            
            quality_scores.append(quality_score)
            concept_coverage_scores.append(concept_score)
        
        return {
            'avg_quality_score': np.mean(quality_scores),
            'avg_concept_coverage': np.mean(concept_coverage_scores) if concept_coverage_scores else 0,
            'high_quality_rate': sum(1 for s in quality_scores if s > 0.7) / len(quality_scores),
            'low_quality_rate': sum(1 for s in quality_scores if s < 0.3) / len(quality_scores)
        }
    
    def compare_architectures(self, rag_results: List[Dict], 
                            api_results: List[Dict]) -> Dict[str, Any]:
        """Compara resultados entre arquiteturas RAG e API"""
        
        comparison = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'rag_metrics': {},
            'api_metrics': {},
            'comparison_summary': {}
        }
        
        # Métricas RAG
        if rag_results:
            comparison['rag_metrics'] = {
                'num_queries': len(rag_results),
                'time_metrics': self.calculate_response_time_metrics(rag_results),
                'relevance_metrics': self.calculate_relevance_metrics(rag_results),
                'cost_metrics': {'total_cost_usd': 0, 'avg_cost_per_query': 0}  # RAG local = gratuito
            }
        
        # Métricas API
        if api_results:
            comparison['api_metrics'] = {
                'num_queries': len(api_results),
                'time_metrics': self.calculate_response_time_metrics(api_results),
                'cost_metrics': self.calculate_cost_metrics(api_results)
            }
        
        # Comparação direta
        if rag_results and api_results:
            rag_avg_time = comparison['rag_metrics']['time_metrics'].get('avg_response_time', 0)
            api_avg_time = comparison['api_metrics']['time_metrics'].get('avg_response_time', 0)
            
            rag_cost = comparison['rag_metrics']['cost_metrics'].get('avg_cost_per_query', 0)
            api_cost = comparison['api_metrics']['cost_metrics'].get('avg_cost_per_query', 0)
            
            comparison['comparison_summary'] = {
                'time_winner': 'RAG' if rag_avg_time < api_avg_time else 'API',
                'cost_winner': 'RAG' if rag_cost < api_cost else 'API',
                'time_difference_sec': abs(rag_avg_time - api_avg_time),
                'cost_difference_usd': abs(rag_cost - api_cost),
                'rag_faster_by': max(0, api_avg_time - rag_avg_time),
                'rag_cheaper_by': max(0, api_cost - rag_cost)
            }
        
        return comparison
    
    def generate_evaluation_report(self, output_dir: Path = None) -> Dict[str, Any]:
        """Gera relatório completo de avaliação"""
        
        if output_dir is None:
            output_dir = Path("data/evaluation")
        
        # Carrega resultados salvos
        rag_results = []
        api_results = []
        
        rag_file = output_dir / "rag_results.json"
        if rag_file.exists():
            with open(rag_file, 'r', encoding='utf-8') as f:
                rag_results = json.load(f)
        
        api_file = output_dir / "api_baseline_results.json"
        if api_file.exists():
            with open(api_file, 'r', encoding='utf-8') as f:
                api_results = json.load(f)
        
        # Carrega conceitos esperados
        expected_concepts = {}
        questions_file = output_dir / "test_questions.json"
        if questions_file.exists():
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                for q in questions:
                    expected_concepts[q['id']] = q.get('expected_concepts', [])
        
        # Avalia qualidade das respostas
        rag_quality = self.evaluate_answer_quality(rag_results, expected_concepts) if rag_results else {}
        api_quality = self.evaluate_answer_quality(api_results, expected_concepts) if api_results else {}
        
        # Comparação entre arquiteturas
        architecture_comparison = self.compare_architectures(rag_results, api_results)
        
        # Relatório completo
        report = {
            'evaluation_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_questions_tested': len(set(
                    [r.get('question_id') for r in rag_results] + 
                    [r.get('question_id') for r in api_results]
                )),
                'architectures_tested': (['RAG'] if rag_results else []) + (['API'] if api_results else [])
            },
            'rag_evaluation': {
                'available': len(rag_results) > 0,
                'results': rag_results if rag_results else [],
                'quality_metrics': rag_quality,
                'performance_metrics': self.calculate_response_time_metrics(rag_results) if rag_results else {},
                'relevance_metrics': self.calculate_relevance_metrics(rag_results) if rag_results else {}
            },
            'api_evaluation': {
                'available': len(api_results) > 0,
                'results': api_results if api_results else [],
                'quality_metrics': api_quality,
                'performance_metrics': self.calculate_response_time_metrics(api_results) if api_results else {},
                'cost_metrics': self.calculate_cost_metrics(api_results) if api_results else {}
            },
            'architecture_comparison': architecture_comparison,
            'recommendations': self._generate_recommendations(architecture_comparison, rag_quality, api_quality)
        }
        
        # Salva relatório
        report_file = output_dir / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Relatório de avaliação salvo em {report_file}")
        return report
    
    def _generate_recommendations(self, comparison: Dict, rag_quality: Dict, api_quality: Dict) -> Dict[str, Any]:
        """Gera recomendações baseadas na avaliação"""
        
        recommendations = {
            'primary_recommendation': None,
            'reasoning': [],
            'trade_offs': {},
            'next_steps': []
        }
        
        # Extrai métricas para comparação
        rag_available = comparison.get('rag_metrics', {}).get('num_queries', 0) > 0
        api_available = comparison.get('api_metrics', {}).get('num_queries', 0) > 0
        
        if not rag_available and not api_available:
            recommendations['primary_recommendation'] = 'insufficient_data'
            recommendations['reasoning'].append('Dados insuficientes para recomendação')
            return recommendations
        
        # Apenas RAG testado
        if rag_available and not api_available:
            rag_quality_score = rag_quality.get('avg_quality_score', 0)
            rag_speed = comparison['rag_metrics']['time_metrics'].get('avg_response_time', 0)
            
            if rag_quality_score > 0.6 and rag_speed < 3.0:
                recommendations['primary_recommendation'] = 'rag_local'
                recommendations['reasoning'].append('RAG local apresenta qualidade e velocidade adequadas')
                recommendations['reasoning'].append('Custo operacional próximo de zero')
            else:
                recommendations['primary_recommendation'] = 'improve_rag_or_try_api'
                recommendations['reasoning'].append('RAG precisa melhorias ou testar APIs externas')
        
        # Apenas API testada
        elif api_available and not rag_available:
            api_cost = comparison['api_metrics']['cost_metrics'].get('avg_cost_per_query', 0)
            api_quality_score = api_quality.get('avg_quality_score', 0)
            
            if api_cost < 0.01 and api_quality_score > 0.6:  # < 1 centavo por query
                recommendations['primary_recommendation'] = 'api_external'
                recommendations['reasoning'].append('API externa com custo e qualidade aceitáveis')
            else:
                recommendations['primary_recommendation'] = 'evaluate_alternatives'
                recommendations['reasoning'].append('Custo ou qualidade da API podem ser problemáticos')
        
        # Ambos testados - comparação completa
        else:
            rag_quality_score = rag_quality.get('avg_quality_score', 0)
            api_quality_score = api_quality.get('avg_quality_score', 0)
            
            rag_speed = comparison['rag_metrics']['time_metrics'].get('avg_response_time', 0)
            api_speed = comparison['api_metrics']['time_metrics'].get('avg_response_time', 0)
            
            api_cost = comparison['api_metrics']['cost_metrics'].get('avg_cost_per_query', 0)
            
            # Critérios de decisão
            quality_diff = abs(rag_quality_score - api_quality_score)
            speed_diff = abs(rag_speed - api_speed)
            
            # RAG vence se qualidade similar e muito mais barato
            if quality_diff < 0.2 and api_cost > 0.005:  # > 0.5 centavos por query
                recommendations['primary_recommendation'] = 'rag_local'
                recommendations['reasoning'].append('RAG local: qualidade similar, custo muito menor')
                recommendations['reasoning'].append(f'Economia estimada: ${api_cost:.4f} por query')
            
            # API vence se qualidade significativamente melhor
            elif api_quality_score > rag_quality_score + 0.3:
                recommendations['primary_recommendation'] = 'api_external'
                recommendations['reasoning'].append('API externa: qualidade significativamente superior')
                recommendations['reasoning'].append(f'Custo adicional: ${api_cost:.4f} por query')
            
            # Empate técnico - decide por volume esperado
            else:
                recommendations['primary_recommendation'] = 'volume_dependent'
                recommendations['reasoning'].append('Recomendação depende do volume esperado:')
                recommendations['reasoning'].append('• < 1000 queries/dia: RAG local')
                recommendations['reasoning'].append('• > 1000 queries/dia: avaliar API externa')
            
            # Trade-offs
            recommendations['trade_offs'] = {
                'rag_advantages': ['Custo zero operacional', 'Controle total', 'Privacidade'],
                'rag_disadvantages': ['Setup inicial', 'Manutenção técnica', 'Limitações de modelo'],
                'api_advantages': ['Qualidade superior', 'Sem manutenção', 'Escalabilidade'],
                'api_disadvantages': ['Custo por uso', 'Dependência externa', 'Latência de rede']
            }
        
        # Próximos passos
        recommendations['next_steps'] = [
            'Testar com mais documentos reais do domínio',
            'Avaliação manual de qualidade das respostas',
            'Definir volume esperado de consultas',
            'Calcular ROI para cenários específicos',
            'Implementar interface de usuário para testes'
        ]
        
        return recommendations
    
    def create_metrics_summary(self, results_file: Path = None) -> pd.DataFrame:
        """Cria resumo das métricas em formato tabular"""
        
        if results_file is None:
            results_file = Path("data/evaluation/evaluation_report.json")
        
        if not results_file.exists():
            logger.warning(f"Arquivo de resultados não encontrado: {results_file}")
            return pd.DataFrame()
        
        with open(results_file, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # Extrai métricas principais
        summary_data = []
        
        # RAG metrics
        if report['rag_evaluation']['available']:
            rag_metrics = report['rag_evaluation']
            summary_data.append({
                'Architecture': 'RAG Local',
                'Avg_Response_Time_sec': rag_metrics['performance_metrics'].get('avg_response_time', 0),
                'Avg_Quality_Score': rag_metrics['quality_metrics'].get('avg_quality_score', 0),
                'Avg_Relevance_Score': rag_metrics['relevance_metrics'].get('avg_relevance_score', 0),
                'Cost_Per_Query_USD': 0.0,
                'High_Quality_Rate': rag_metrics['quality_metrics'].get('high_quality_rate', 0),
                'Fast_Response_Rate': rag_metrics['performance_metrics'].get('under_3s_rate', 0)
            })
        
        # API metrics
        if report['api_evaluation']['available']:
            api_metrics = report['api_evaluation']
            summary_data.append({
                'Architecture': 'API External',
                'Avg_Response_Time_sec': api_metrics['performance_metrics'].get('avg_response_time', 0),
                'Avg_Quality_Score': api_metrics['quality_metrics'].get('avg_quality_score', 0),
                'Avg_Relevance_Score': 0.0,  # N/A para API
                'Cost_Per_Query_USD': api_metrics['cost_metrics'].get('avg_cost_per_query', 0),
                'High_Quality_Rate': api_metrics['quality_metrics'].get('high_quality_rate', 0),
                'Fast_Response_Rate': api_metrics['performance_metrics'].get('under_3s_rate', 0)
            })
        
        df = pd.DataFrame(summary_data)
        
        # Salva resumo
        summary_file = Path("data/evaluation/metrics_summary.csv")
        df.to_csv(summary_file, index=False)
        
        logger.info(f"✅ Resumo de métricas salvo em {summary_file}")
        return df


def run_evaluation():
    """Executa avaliação completa dos resultados"""
    logger.info("📊 Executando avaliação completa...")
    
    evaluator = SimpleEvaluator()
    
    # Gera relatório completo
    report = evaluator.generate_evaluation_report()
    
    # Cria resumo tabular
    summary_df = evaluator.create_metrics_summary()
    
    # Log principais insights
    if not summary_df.empty:
        logger.info("📈 RESUMO DA AVALIAÇÃO:")
        for _, row in summary_df.iterrows():
            arch = row['Architecture']
            time_avg = row['Avg_Response_Time_sec']
            quality = row['Avg_Quality_Score']
            cost = row['Cost_Per_Query_USD']
            
            logger.info(f"  {arch}:")
            logger.info(f"    • Tempo médio: {time_avg:.2f}s")
            logger.info(f"    • Qualidade: {quality:.3f}")
            logger.info(f"    • Custo: ${cost:.4f}/query")
    
    # Log recomendação principal
    recommendation = report.get('recommendations', {}).get('primary_recommendation')
    if recommendation:
        logger.info(f"\n🎯 RECOMENDAÇÃO: {recommendation}")
        
        reasoning = report.get('recommendations', {}).get('reasoning', [])
        for reason in reasoning:
            logger.info(f"   • {reason}")
    
    return report


if __name__ == "__main__":
    # Teste standalone
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
