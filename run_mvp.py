#!/usr/bin/env python3
"""
MVP Runner - LLM Architecture Comparison
Script simplificado para validação rápida da abordagem

Execute com: python run_mvp.py
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
    """Verifica ambiente básico"""
    logger.info("🔍 Verificando ambiente...")
    
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import streamlit
        logger.info("✅ Dependências principais encontradas")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

def create_sample_data():
    """Cria dados de exemplo para o MVP"""
    logger.info("📝 Criando dados de exemplo...")
    
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dados sintéticos sobre construção civil
    sample_docs = [
        {
            "id": "doc_001",
            "title": "Normas de Ruído - NBR 10151",
            "content": "O monitoramento de ruído em obras urbanas deve seguir as diretrizes da NBR 10151. Os níveis de ruído não podem exceder 70 dB durante o dia e 60 dB durante a noite em áreas residenciais. É obrigatório o uso de medidores calibrados e registros horários. A medição deve ser feita no limite da propriedade mais próxima aos receptores sensíveis.",
            "category": "regulamentacao",
            "keywords": ["ruído", "NBR 10151", "medição", "limites", "obras urbanas"]
        },
        {
            "id": "doc_002", 
            "title": "EPIs Obrigatórios em Canteiros",
            "content": "Equipamentos de proteção individual (EPIs) obrigatórios em canteiros incluem: capacete de segurança classe A ou B, óculos de proteção contra impactos, luvas adequadas ao tipo de atividade, calçados de segurança com biqueira de aço, cintos de segurança para trabalho em altura acima de 2 metros. A empresa deve fornecer gratuitamente e treinar os trabalhadores sobre o uso correto.",
            "category": "seguranca",
            "keywords": ["EPIs", "capacete", "óculos", "luvas", "calçados", "cintos", "treinamento"]
        },
        {
            "id": "doc_003",
            "title": "Controle de Impacto Ambiental",
            "content": "O controle de impacto ambiental em construções requer licenciamento prévio do órgão competente, elaboração de plano de gerenciamento de resíduos sólidos, implementação de medidas de controle de erosão e proteção de recursos hídricos. Obras com área superior a 3000m² necessitam de estudo de impacto ambiental (EIA/RIMA).",
            "category": "meio_ambiente", 
            "keywords": ["licenciamento", "resíduos", "erosão", "recursos hídricos", "EIA", "RIMA"]
        },
        {
            "id": "doc_004",
            "title": "Obras Noturnas - Regulamentação",
            "content": "Obras noturnas requerem autorização especial da prefeitura municipal, iluminação adequada com intensidade mínima de 200 lux, sinalização reforçada com dispositivos refletivos e luminosos, equipes especializadas com treinamento em segurança noturna. O horário permitido varia entre 22h e 6h, dependendo da classificação da área urbana.",
            "category": "regulamentacao",
            "keywords": ["obras noturnas", "autorização", "iluminação", "200 lux", "sinalização", "22h-6h"]
        },
        {
            "id": "doc_005",
            "title": "Controle de Qualidade do Concreto",
            "content": "O controle de qualidade em construção envolve inspeções regulares dos materiais, realização de ensaios de resistência à compressão, documentação completa de todos os procedimentos e auditorias internas periódicas. Para concreto estrutural, devem ser realizados ensaios a cada 50m³ ou a cada dia de concretagem, prevalecendo o menor valor.",
            "category": "qualidade",
            "keywords": ["controle qualidade", "inspeções", "ensaios", "concreto", "50m³", "resistência"]
        }
    ]
    
    # Salva documentos individuais
    for doc in sample_docs:
        doc_file = data_dir / f"{doc['id']}.json"
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Criados {len(sample_docs)} documentos em {data_dir}")
    return sample_docs

def create_test_questions():
    """Cria perguntas de teste para avaliação"""
    test_questions = [
        {
            "id": "q001",
            "question": "Quais são os limites de ruído permitidos em obras urbanas?",
            "expected_concepts": ["70 dB", "60 dB", "NBR 10151", "dia", "noite"],
            "category": "regulamentacao"
        },
        {
            "id": "q002", 
            "question": "Que equipamentos de proteção são obrigatórios em canteiros de obra?",
            "expected_concepts": ["capacete", "óculos", "luvas", "calçados", "cintos"],
            "category": "seguranca"
        },
        {
            "id": "q003",
            "question": "Como deve ser feito o controle de qualidade do concreto?",
            "expected_concepts": ["50m³", "ensaios", "resistência", "dia de concretagem"],
            "category": "qualidade"
        },
        {
            "id": "q004",
            "question": "Quais são os requisitos para obras noturnas?",
            "expected_concepts": ["autorização", "200 lux", "22h", "6h", "sinalização"],
            "category": "regulamentacao"
        },
        {
            "id": "q005",
            "question": "Quando é necessário fazer EIA/RIMA para uma obra?", 
            "expected_concepts": ["3000m²", "estudo de impacto", "licenciamento"],
            "category": "meio_ambiente"
        }
    ]
    
    # Salva perguntas de teste
    questions_file = Path("data/evaluation/test_questions.json")
    questions_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(test_questions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Criadas {len(test_questions)} perguntas de teste")
    return test_questions

def setup_rag_simple():
    """Configura RAG simples com modelo leve"""
    logger.info("🤖 Configurando RAG simples...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        
        # Carrega documentos
        data_dir = Path("data/raw")
        documents = []
        
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
        
        if not documents:
            logger.error("❌ Nenhum documento encontrado")
            return False
        
        # Modelo de embedding leve
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cria embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        
        # Cria índice FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Salva índice e metadados
        embeddings_dir = Path("data/embeddings")
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(embeddings_dir / "index.faiss"))
        
        # Salva documentos para recuperação
        docs_df = pd.DataFrame(documents)
        docs_df.to_json(embeddings_dir / "documents.json", orient='records', indent=2)
        
        logger.info(f"✅ RAG configurado: {len(documents)} docs, dimensão {dimension}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no setup RAG: {e}")
        return False

def test_rag_basic():
    """Testa RAG básico com perguntas simples"""
    logger.info("🧪 Testando RAG básico...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import pandas as pd
        
        # Carrega índice
        embeddings_dir = Path("data/embeddings") 
        index = faiss.read_index(str(embeddings_dir / "index.faiss"))
        docs_df = pd.read_json(embeddings_dir / "documents.json", orient='records')
        documents = docs_df.to_dict('records')
        
        # Carrega modelo
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Carrega perguntas
        with open("data/evaluation/test_questions.json", 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        results = []
        
        for q in test_questions:
            start_time = time.time()
            
            # Busca documentos relevantes
            query_embedding = embedding_model.encode([q['question']])
            scores, indices = index.search(query_embedding.astype('float32'), k=3)
            
            # Recupera documentos
            retrieved_docs = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(documents):
                    doc = documents[idx].copy()
                    doc['relevance_score'] = float(score)
                    retrieved_docs.append(doc)
            
            response_time = time.time() - start_time
            
            # Resposta simples (concatena conteúdo dos docs mais relevantes)
            if retrieved_docs:
                answer = retrieved_docs[0]['content'][:200] + "..."
                relevance = 1.0 / (1.0 + retrieved_docs[0]['relevance_score'])  # Converte distância em relevância
            else:
                answer = "Nenhum documento relevante encontrado."
                relevance = 0.0
            
            result = {
                'question_id': q['id'],
                'question': q['question'],
                'answer': answer,
                'response_time': response_time,
                'relevance_score': relevance,
                'retrieved_docs': len(retrieved_docs)
            }
            
            results.append(result)
            
            logger.info(f"Q{q['id']}: {response_time:.2f}s, relevância: {relevance:.3f}")
        
        # Salva resultados
        results_file = Path("data/evaluation/rag_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # Métricas agregadas
        avg_time = sum(r['response_time'] for r in results) / len(results)
        avg_relevance = sum(r['relevance_score'] for r in results) / len(results)
        
        logger.info(f"✅ RAG testado: {avg_time:.2f}s médio, relevância {avg_relevance:.3f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Erro no teste RAG: {e}")
        return []

def estimate_costs():
    """Estima custos básicos para diferentes cenários"""
    logger.info("💰 Estimando custos...")
    
    scenarios = [
        {"name": "Piloto", "queries_per_day": 100},
        {"name": "Produção Pequena", "queries_per_day": 1000}, 
        {"name": "Produção Média", "queries_per_day": 5000}
    ]
    
    # Custos aproximados (USD)
    costs = {
        "rag_local": {
            "setup_cost": 0,  # Usando hardware existente
            "cost_per_1k_queries": 0.01,  # Apenas eletricidade/desgaste
            "monthly_fixed": 0
        },
        "openai_api": {
            "setup_cost": 0,
            "cost_per_1k_queries": 2.00,  # GPT-3.5-turbo aproximado
            "monthly_fixed": 0
        }
    }
    
    results = []
    
    for scenario in scenarios:
        monthly_queries = scenario["queries_per_day"] * 30
        
        for arch, cost_config in costs.items():
            monthly_cost = (
                cost_config["monthly_fixed"] + 
                (monthly_queries / 1000) * cost_config["cost_per_1k_queries"]
            )
            
            annual_cost = cost_config["setup_cost"] + (monthly_cost * 12)
            
            results.append({
                "scenario": scenario["name"],
                "architecture": arch,
                "queries_per_day": scenario["queries_per_day"],
                "monthly_cost_usd": monthly_cost,
                "annual_cost_usd": annual_cost,
                "cost_per_query_usd": monthly_cost / monthly_queries if monthly_queries > 0 else 0
            })
    
    # Salva estimativas
    costs_file = Path("data/evaluation/cost_estimates.json")
    with open(costs_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("✅ Custos estimados salvos")
    return results

def generate_summary_report():
    """Gera relatório sumário do MVP""" 
    logger.info("📋 Gerando relatório sumário...")
    
    try:
        # Carrega resultados
        rag_results = []
        costs = []
        
        rag_file = Path("data/evaluation/rag_results.json")
        if rag_file.exists():
            with open(rag_file, 'r', encoding='utf-8') as f:
                rag_results = json.load(f)
        
        costs_file = Path("data/evaluation/cost_estimates.json")
        if costs_file.exists():
            with open(costs_file, 'r', encoding='utf-8') as f:
                costs = json.load(f)
        
        # Métricas agregadas RAG
        if rag_results:
            avg_time = sum(r['response_time'] for r in rag_results) / len(rag_results)
            avg_relevance = sum(r['relevance_score'] for r in rag_results) / len(rag_results)
        else:
            avg_time, avg_relevance = 0, 0
        
        report = {
            "mvp_summary": {
                "execution_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_test_questions": len(rag_results),
                "rag_performance": {
                    "avg_response_time_sec": round(avg_time, 2),
                    "avg_relevance_score": round(avg_relevance, 3),
                    "success_rate": len([r for r in rag_results if r['relevance_score'] > 0.1]) / len(rag_results) if rag_results else 0
                },
                "cost_analysis": costs
            },
            "recommendations": {
                "for_pilot": "RAG local se < 500 queries/dia",
                "for_production": "Avaliar APIs externas se > 1000 queries/dia",
                "next_steps": [
                    "Testar com mais documentos reais",
                    "Implementar interface usuário", 
                    "Avaliar qualidade respostas manualmente",
                    "Decidir entre arquiteturas"
                ]
            },
            "technical_notes": {
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_search": "FAISS",
                "limitations": [
                    "Apenas 5 documentos teste",
                    "Sem modelo geração próprio",
                    "Métricas básicas apenas"
                ]
            }
        }
        
        # Salva relatório
        report_file = Path("data/evaluation/mvp_summary_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Relatório salvo em {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"❌ Erro no relatório: {e}")
        return {}

def main():
    """Função principal do MVP"""
    logger.info("🚀 EXECUTANDO MVP - LLM COMPARISON")
    logger.info("="*50)
    
    # Passo 1: Verificar ambiente
    if not check_environment():
        logger.error("❌ Ambiente não configurado. Execute: pip install -r requirements.txt")
        return False
    
    # Passo 2: Validar configuração
    sys.path.append('src')
    from config import Config
    
    if not Config.validate_setup():
        logger.error("❌ Configuração inválida")
        return False
    
    # Passo 3: Criar dados de exemplo
    sample_docs = create_sample_data()
    test_questions = create_test_questions()
    
    # Passo 4: Setup RAG
    if not setup_rag_simple():
        logger.error("❌ Falha no setup RAG")
        return False
    
    # Passo 5: Testar RAG
    rag_results = test_rag_basic()
    if not rag_results:
        logger.error("❌ Falha nos testes RAG")
        return False
    
    # Passo 6: Estimar custos
    cost_estimates = estimate_costs()
    
    # Passo 7: Gerar relatório
    report = generate_summary_report()
    
    # Resumo final
    logger.info("\n" + "="*50)
    logger.info("🎉 MVP CONCLUÍDO COM SUCESSO!")
    logger.info("="*50)
    
    if rag_results:
        avg_time = sum(r['response_time'] for r in rag_results) / len(rag_results)
        avg_relevance = sum(r['relevance_score'] for r in rag_results) / len(rag_results)
        
        logger.info(f"📊 Resultados RAG:")
        logger.info(f"   • Tempo médio: {avg_time:.2f}s")
        logger.info(f"   • Relevância média: {avg_relevance:.3f}")
        logger.info(f"   • Perguntas testadas: {len(rag_results)}")
    
    if cost_estimates:
        logger.info(f"💰 Custos estimados:")
        for cost in cost_estimates[:3]:  # Primeiros 3 cenários
            logger.info(f"   • {cost['scenario']}: ${cost['monthly_cost_usd']:.2f}/mês")
    
    logger.info(f"\n📋 Próximos passos:")
    logger.info(f"   1. Execute: streamlit run app/dashboard.py")
    logger.info(f"   2. Avalie resultados manualmente")
    logger.info(f"   3. Decida próxima arquitetura")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
        