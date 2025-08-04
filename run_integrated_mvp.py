#!/usr/bin/env python3
"""
MVP Integrado - Comparação Multi-Cenário RAG
Adaptado para incluir os três cenários: Vector RAG, Hybrid RAG e LLM-Only

Execute com: python run_integrated_mvp.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Verifica ambiente com dependências para os três cenários"""
    logger.info("🔍 Verificando ambiente para multi-cenário...")
    
    missing_deps = []
    
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import networkx
        import streamlit
        logger.info("✅ Dependências principais encontradas")
    except ImportError as e:
        missing_deps.append(str(e))
    
    if missing_deps:
        logger.error("❌ Dependências faltando:")
        for dep in missing_deps:
            logger.error(f"  {dep}")
        logger.error("Execute: conda env update -f environment.yml")
        return False
    
    return True

def create_enhanced_sample_data():
    """Cria dados de exemplo enriquecidos com informações temporais"""
    logger.info("📝 Criando dados de exemplo com contexto temporal...")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dados enriquecidos com contexto temporal e entidades
    enhanced_docs = [
        {
            "id": "doc_001",
            "title": "Normas de Ruído - NBR 10151:2019",
            "content": "O monitoramento de ruído em obras urbanas deve seguir as diretrizes da NBR 10151, revisão 2019. Os níveis de ruído não podem exceder 70 dB durante o período diurno (das 7h às 22h) e 60 dB durante o período noturno (das 22h às 7h) em áreas residenciais. É obrigatório o uso de medidores calibrados classe 1 ou 2, com certificado de calibração válido. A medição deve ser feita no limite da propriedade mais próxima aos receptores sensíveis. Para obras que ultrapassem 60 dias consecutivos, é necessário relatório mensal de monitoramento. Em caso de reclamações, medições extraordinárias devem ser realizadas em até 48 horas.",
            "category": "regulamentacao",
            "keywords": ["ruído", "NBR 10151", "medição", "limites", "obras urbanas", "horário", "calibração"],
            "temporal_markers": ["7h às 22h", "22h às 7h", "60 dias", "mensal", "48 horas"],
            "entities": ["NBR 10151", "70 dB", "60 dB", "medidores classe 1", "certificado calibração"]
        },
        {
            "id": "doc_002", 
            "title": "EPIs Obrigatórios - NR 6:2020",
            "content": "Equipamentos de proteção individual (EPIs) obrigatórios em canteiros incluem: capacete de segurança classe A ou B, obrigatório durante todo o período de trabalho das 6h às 18h. Óculos de proteção contra impactos para atividades com risco de projeção de partículas. Luvas adequadas ao tipo de atividade (não usar próximo a máquinas rotativas). Calçados de segurança com biqueira de aço, obrigatórios em toda a área da obra. Cintos de segurança para trabalho em altura acima de 2 metros, com inspeção diária obrigatória. A empresa deve fornecer gratuitamente, treinar os trabalhadores sobre o uso correto no primeiro dia de trabalho e substituir quando necessário. Inspeção semanal dos EPIs é obrigatória, com registro em planilha. EPIs danificados devem ser substituídos imediatamente.",
            "category": "seguranca",
            "keywords": ["EPIs", "capacete", "óculos", "luvas", "calçados", "cintos", "treinamento", "inspeção"],
            "temporal_markers": ["6h às 18h", "primeiro dia", "diária", "semanal", "imediatamente"],
            "entities": ["NR 6", "classe A", "classe B", "2 metros", "biqueira de aço"]
        },
        {
            "id": "doc_003",
            "title": "Controle de Qualidade do Concreto - NBR 12655:2015",
            "content": "O controle de qualidade do concreto estrutural requer ensaios de resistência à compressão a cada 50m³ ou a cada dia de concretagem, prevalecendo o menor valor. Os corpos de prova devem ser moldados no local da obra, curados em condições padronizadas por 28 dias para ensaio definitivo. Ensaios intermediários aos 7 dias e 14 dias permitem avaliação prévia da evolução da resistência. Para obras de grande porte (acima de 500m³), ensaios adicionais de módulo de elasticidade são recomendados mensalmente. Documentação completa deve ser mantida por no mínimo 5 anos após conclusão da obra. Em caso de resultado insatisfatório, nova amostragem deve ser feita imediatamente e ensaios refeitos em 48 horas.",
            "category": "qualidade",
            "keywords": ["controle qualidade", "concreto", "ensaios", "50m³", "28 dias", "resistência", "documentação"],
            "temporal_markers": ["50m³", "dia de concretagem", "28 dias", "7 dias", "14 dias", "mensalmente", "5 anos", "48 horas"],
            "entities": ["NBR 12655", "28 dias", "500m³", "módulo elasticidade"]
        },
        {
            "id": "doc_004",
            "title": "Obras Noturnas - Lei Municipal 16.402/2016",
            "content": "Obras noturnas requerem autorização especial da prefeitura municipal, com solicitação feita com antecedência mínima de 15 dias úteis. Iluminação adequada com intensidade mínima de 200 lux deve ser mantida em toda a área de trabalho durante o período noturno (das 22h às 6h). Sinalização reforçada com dispositivos refletivos e luminosos deve ser instalada até 50 metros antes da obra. Equipes especializadas com treinamento específico em segurança noturna são obrigatórias, com certificação renovada anualmente. O horário permitido varia entre 22h e 6h em áreas comerciais, e entre 23h e 5h em áreas residenciais. Relatório diário de atividades deve ser enviado à fiscalização até às 8h do dia seguinte.",
            "category": "regulamentacao", 
            "keywords": ["obras noturnas", "autorização", "iluminação", "200 lux", "sinalização", "22h-6h"],
            "temporal_markers": ["15 dias úteis", "22h às 6h", "23h e 5h", "anualmente", "8h do dia seguinte"],
            "entities": ["Lei 16.402/2016", "200 lux", "50 metros", "22h-6h", "23h-5h"]
        },
        {
            "id": "doc_005",
            "title": "Controle Ambiental - Resolução CONAMA 307/2002",
            "content": "O controle de impacto ambiental em construções requer licenciamento prévio do órgão competente, solicitado com antecedência mínima de 90 dias do início da obra. Elaboração de plano de gerenciamento de resíduos sólidos é obrigatória, com revisão trimestral. Implementação de medidas de controle de erosão deve ocorrer antes do período chuvoso. Proteção de recursos hídricos com barreiras físicas num raio de 30 metros de corpos d'água. Obras com área superior a 3000m² necessitam de estudo de impacto ambiental (EIA/RIMA), com prazo de análise de 180 dias. Monitoramento da qualidade do ar deve ser realizado semanalmente durante toda a obra. Relatórios mensais devem ser enviados aos órgãos ambientais até o 5º dia útil do mês seguinte.",
            "category": "meio_ambiente",
            "keywords": ["licenciamento", "resíduos", "erosão", "recursos hídricos", "EIA", "RIMA"],
            "temporal_markers": ["90 dias", "trimestral", "período chuvoso", "180 dias", "semanalmente", "mensalmente", "5º dia útil"],
            "entities": ["CONAMA 307/2002", "3000m²", "30 metros", "EIA/RIMA"]
        }
    ]
    
    # Salva documentos
    for doc in enhanced_docs:
        doc_file = data_dir / f"{doc['id']}.json"
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Criados {len(enhanced_docs)} documentos enriquecidos em {data_dir}")
    return enhanced_docs

def create_temporal_test_questions():
    """Cria perguntas de teste com características temporais específicas"""
    temporal_questions = [
        {
            "id": "q001",
            "question": "Quais são os limites de ruído permitidos durante o dia e durante a noite?",
            "expected_concepts": ["70 dB", "60 dB", "7h às 22h", "22h às 7h", "NBR 10151"],
            "category": "regulamentacao",
            "temporal_focus": "schedule",
            "complexity": "basic"
        },
        {
            "id": "q002", 
            "question": "Com que frequência devem ser realizados os ensaios de resistência do concreto?",
            "expected_concepts": ["50m³", "dia de concretagem", "28 dias", "ensaios"],
            "category": "qualidade",
            "temporal_focus": "frequency",
            "complexity": "intermediate"
        },
        {
            "id": "q003",
            "question": "Que procedimentos temporais são necessários antes de iniciar obras noturnas?",
            "expected_concepts": ["15 dias úteis", "autorização", "antecedência", "prefeitura"],
            "category": "regulamentacao",
            "temporal_focus": "sequence",
            "complexity": "intermediate"
        },
        {
            "id": "q004",
            "question": "Qual o prazo de análise para EIA/RIMA e quando deve ser solicitado?",
            "expected_concepts": ["180 dias", "90 dias", "3000m²", "EIA/RIMA"],
            "category": "meio_ambiente",
            "temporal_focus": "duration",
            "complexity": "advanced"
        },
        {
            "id": "q005",
            "question": "Quando os EPIs devem ser inspecionados e por quanto tempo manter documentação?",
            "expected_concepts": ["semanal", "diária", "5 anos", "inspeção"],
            "category": "seguranca",
            "temporal_focus": "multiple_timeframes",
            "complexity": "advanced"
        }
    ]
    
    # Salva perguntas
    questions_file = Path("data/evaluation/temporal_test_questions.json")
    questions_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(temporal_questions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Criadas {len(temporal_questions)} perguntas temporais")
    return temporal_questions

def run_multi_scenario_comparison():
    """Executa comparação completa entre os três cenários"""
    logger.info("🚀 Iniciando comparação multi-cenário...")
    
    try:
        # Importa sistema multi-cenário
        sys.path.append('src')
        from multi_scenario_system import MultiScenarioSystem
        
        # Carrega dados
        data_dir = Path("data/raw")
        documents = []
        
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
        
        if not documents:
            logger.error("❌ Nenhum documento encontrado")
            return []
        
        # Carrega perguntas
        questions_file = Path("data/evaluation/temporal_test_questions.json")
        if not questions_file.exists():
            logger.error("❌ Perguntas de teste não encontradas")
            return []
        
        with open(questions_file, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        # Inicializa sistema multi-cenário
        system = MultiScenarioSystem()
        system.load_documents(documents)
        system.build_all_scenarios()
        
        # Status do sistema
        status = system.get_system_status()
        logger.info(f"📊 Status do sistema: {status}")
        
        # Executa comparação para cada pergunta
        all_comparisons = []
        scenario_performance = {
            'scenario_a': {'total_time': 0, 'total_cost': 0, 'total_relevance': 0, 'success_count': 0},
            'scenario_b': {'total_time': 0, 'total_cost': 0, 'total_relevance': 0, 'success_count': 0},
            'scenario_c': {'total_time': 0, 'total_cost': 0, 'total_relevance': 0, 'success_count': 0}
        }
        
        for question in test_questions:
            logger.info(f"\n❓ Pergunta {question['id']}: {question['question']}")
            
            # Compara todos os cenários
            comparison = system.compare_all_scenarios(question['question'])
            
            # Adiciona metadados da pergunta
            comparison['question_metadata'] = {
                'id': question['id'],
                'category': question['category'],
                'temporal_focus': question['temporal_focus'],
                'complexity': question['complexity'],
                'expected_concepts': question['expected_concepts']
            }
            
            all_comparisons.append(comparison)
            
            # Coleta métricas de performance
            for scenario, result in comparison['results'].items():
                if 'error' not in result:
                    perf = scenario_performance[scenario]
                    perf['total_time'] += result.get('response_time', 0)
                    perf['total_cost'] += result.get('estimated_cost_usd', 0)
                    perf['total_relevance'] += result.get('relevance_score', 0)
                    perf['success_count'] += 1
                    
                    # Log resultado individual
                    logger.info(f"  {scenario}: {result.get('response_time', 0):.2f}s, "
                              f"${result.get('estimated_cost_usd', 0):.4f}, "
                              f"relevância: {result.get('relevance_score', 0):.3f}")
                else:
                    logger.error(f"  {scenario}: ERRO - {result['error']}")
        
        # Calcula métricas agregadas
        aggregated_metrics = {}
        for scenario, perf in scenario_performance.items():
            if perf['success_count'] > 0:
                aggregated_metrics[scenario] = {
                    'avg_response_time': perf['total_time'] / perf['success_count'],
                    'total_cost': perf['total_cost'],
                    'avg_relevance': perf['total_relevance'] / perf['success_count'],
                    'success_rate': perf['success_count'] / len(test_questions),
                    'successful_queries': perf['success_count']
                }
            else:
                aggregated_metrics[scenario] = {
                    'avg_response_time': 0,
                    'total_cost': 0,
                    'avg_relevance': 0,
                    'success_rate': 0,
                    'successful_queries': 0
                }
        
        # Salva resultados detalhados
        results_file = Path("data/evaluation/multi_scenario_comparison.json")
        detailed_results = {
            'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_status': status,
            'individual_comparisons': all_comparisons,
            'aggregated_metrics': aggregated_metrics,
            'test_summary': {
                'total_questions': len(test_questions),
                'documents_used': len(documents)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Resultados salvos em {results_file}")
        return detailed_results
        
    except Exception as e:
        logger.error(f"❌ Erro na comparação: {e}")
        return {}

def generate_comparative_report(results: Dict):
    """Gera relatório comparativo entre os cenários"""
    logger.info("📋 Gerando relatório comparativo...")
    
    if not results or 'aggregated_metrics' not in results:
        logger.error("❌ Resultados insuficientes para relatório")
        return {}
    
    metrics = results['aggregated_metrics']
    
    # Análise comparativa
    report = {
        'execution_date': results.get('execution_timestamp', 'N/A'),
        'summary': {
            'total_questions_tested': results.get('test_summary', {}).get('total_questions', 0),
            'documents_analyzed': results.get('test_summary', {}).get('documents_used', 0)
        },
        'scenario_comparison': {},
        'recommendations': {
            'best_for_speed': None,
            'best_for_cost': None,
            'best_for_accuracy': None,
            'overall_recommendation': None,
            'detailed_analysis': []
        }
    }
    
    # Análise por cenário
    for scenario, metric in metrics.items():
        scenario_name = {
            'scenario_a': 'Vector-Only RAG',
            'scenario_b': 'Hybrid RAG (Vector + Graph)', 
            'scenario_c': 'LLM-Only'
        }.get(scenario, scenario)
        
        report['scenario_comparison'][scenario_name] = {
            'average_response_time_sec': round(metric['avg_response_time'], 3),
            'total_cost_usd': round(metric['total_cost'], 4),
            'average_relevance_score': round(metric['avg_relevance'], 3),
            'success_rate': round(metric['success_rate'] * 100, 1),
            'successful_queries': metric['successful_queries']
        }
    
    # Determina melhores cenários
    valid_scenarios = {k: v for k, v in metrics.items() if v['success_rate'] > 0}
    
    if valid_scenarios:
        # Melhor velocidade
        fastest = min(valid_scenarios, key=lambda x: valid_scenarios[x]['avg_response_time'])
        report['recommendations']['best_for_speed'] = fastest
        
        # Melhor custo
        cheapest = min(valid_scenarios, key=lambda x: valid_scenarios[x]['total_cost'])
        report['recommendations']['best_for_cost'] = cheapest
        
        # Melhor precisão
        most_accurate = max(valid_scenarios, key=lambda x: valid_scenarios[x]['avg_relevance'])
        report['recommendations']['best_for_accuracy'] = most_accurate
        
        # Análise detalhada
        analysis = []
        
        if fastest == 'scenario_c':
            analysis.append("LLM-Only oferece maior velocidade por não ter overhead de recuperação")
        elif fastest == 'scenario_a':
            analysis.append("Vector RAG oferece boa velocidade com informações contextualizadas")
        else:
            analysis.append("Hybrid RAG prioriza precisão sobre velocidade")
        
        if cheapest in ['scenario_a', 'scenario_b']:
            analysis.append("Cenários RAG são mais econômicos por não usar APIs externas")
        else:
            analysis.append("LLM-Only tem custos de API mas oferece respostas mais elaboradas")
        
        if most_accurate == 'scenario_b':
            analysis.append("Hybrid RAG com grafos oferece melhor raciocínio temporal")
        elif most_accurate == 'scenario_a':
            analysis.append("Vector RAG oferece boa precisão com base documental")
        else:
            analysis.append("LLM-Only tem conhecimento geral robusto")
        
        report['recommendations']['detailed_analysis'] = analysis
        
        # Recomendação geral baseada em balance
        speed_rank = {v: k for k, v in enumerate(sorted(valid_scenarios, key=lambda x: valid_scenarios[x]['avg_response_time']))}
        cost_rank = {v: k for k, v in enumerate(sorted(valid_scenarios, key=lambda x: valid_scenarios[x]['total_cost']))}
        accuracy_rank = {v: k for k, v in enumerate(sorted(valid_scenarios, key=lambda x: valid_scenarios[x]['avg_relevance'], reverse=True))}
        
        # Score balanceado (menor é melhor)
        balanced_scores = {}
        for scenario in valid_scenarios:
            score = speed_rank[scenario] + cost_rank[scenario] + accuracy_rank[scenario]
            balanced_scores[scenario] = score
        
        best_balanced = min(balanced_scores, key=balanced_scores.get)
        report['recommendations']['overall_recommendation'] = best_balanced
    
    # Salva relatório
    report_file = Path("data/evaluation/comparative_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Relatório comparativo salvo em {report_file}")
    return report

def print_executive_summary(report: Dict):
    """Imprime resumo executivo dos resultados"""
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMO EXECUTIVO - COMPARAÇÃO MULTI-CENÁRIO")
    logger.info("="*60)
    
    if not report or 'scenario_comparison' not in report:
        logger.error("❌ Relatório inválido")
        return
    
    # Informações gerais
    summary = report.get('summary', {})
    logger.info(f"📋 Perguntas testadas: {summary.get('total_questions_tested', 0)}")
    logger.info(f"📄 Documentos analisados: {summary.get('documents_analyzed', 0)}")
    logger.info(f"📅 Execução: {report.get('execution_date', 'N/A')}")
    
    # Performance por cenário
    logger.info("\n🎯 PERFORMANCE POR CENÁRIO:")
    for scenario, metrics in report['scenario_comparison'].items():
        logger.info(f"\n• {scenario}:")
        logger.info(f"  ⚡ Tempo médio: {metrics['average_response_time_sec']}s")
        logger.info(f"  💰 Custo total: ${metrics['total_cost_usd']}")
        logger.info(f"  🎯 Relevância média: {metrics['average_relevance_score']}")
        logger.info(f"  ✅ Taxa de sucesso: {metrics['success_rate']}%")
    
    # Recomendações
    recommendations = report.get('recommendations', {})
    logger.info("\n🏆 RECOMENDAÇÕES:")
    
    if recommendations.get('best_for_speed'):
        logger.info(f"⚡ Mais rápido: {recommendations['best_for_speed']}")
    
    if recommendations.get('best_for_cost'):
        logger.info(f"💰 Mais econômico: {recommendations['best_for_cost']}")
    
    if recommendations.get('best_for_accuracy'):
        logger.info(f"🎯 Mais preciso: {recommendations['best_for_accuracy']}")
    
    if recommendations.get('overall_recommendation'):
        logger.info(f"🌟 Recomendação geral: {recommendations['overall_recommendation']}")
    
    analysis = recommendations.get('detailed_analysis', [])
    if analysis:
        logger.info("\n📋 ANÁLISE DETALHADA:")
        for point in analysis:
            logger.info(f"  • {point}")
    
    logger.info("\n" + "="*60)

def main():
    """Função principal do MVP integrado"""
    logger.info("🚀 EXECUTANDO MVP INTEGRADO - COMPARAÇÃO MULTI-CENÁRIO")
    logger.info("="*70)
    
    # Passo 1: Verificar ambiente
    if not check_environment():
        logger.error("❌ Ambiente não configurado adequadamente")
        return False
    
    # Passo 2: Validar configuração base
    sys.path.append('src')
    try:
        from config import Config
        if not Config.validate_setup():
            logger.error("❌ Configuração base inválida")
            return False
    except ImportError:
        logger.warning("⚠️ Config não encontrado, continuando...")
    
    # Passo 3: Criar dados de exemplo enriquecidos
    enhanced_docs = create_enhanced_sample_data()
    temporal_questions = create_temporal_test_questions()
    
    # Passo 4: Executar comparação multi-cenário
    comparison_results = run_multi_scenario_comparison()
    
    if not comparison_results:
        logger.error("❌ Falha na comparação multi-cenário")
        return False
    
    # Passo 5: Gerar relatório comparativo
    comparative_report = generate_comparative_report(comparison_results)
    
    # Passo 6: Exibir resumo executivo
    print_executive_summary(comparative_report)
    
    # Passo 7: Próximos passos
    logger.info("\n📋 PRÓXIMOS PASSOS:")
    logger.info("   1. Execute: streamlit run app/multi_scenario_dashboard.py")
    logger.info("   2. Analise os resultados detalhados nos arquivos JSON")
    logger.info("   3. Teste com seus próprios documentos e perguntas")
    logger.info("   4. Escolha o cenário mais adequado para produção")
    
    logger.info(f"\n✅ MVP INTEGRADO CONCLUÍDO COM SUCESSO!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    