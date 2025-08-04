#!/usr/bin/env python3
"""
MVP Integrado - Comparação Multi-Cenário RAG com Kùzu DB
"""
import sys
import json
import time
import logging
from pathlib import Path

# Adiciona o diretório 'src' ao path para encontrar o multi_scenario_system
sys.path.append(str(Path(__file__).parent / 'src'))

from multi_scenario_system import MultiScenarioSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Verifica se as dependências cruciais estão instaladas."""
    logger.info("🔍 Verificando ambiente...")
    try:
        import faiss
        import kuzu
        import sentence_transformers
        logger.info("✅ Dependências principais (faiss, kuzu, sentence-transformers) encontradas.")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}. Por favor, instale com 'pip install -r requirements.txt' ou via 'conda env create -f environment.yml'")
        return False

def create_enhanced_sample_data():
    """Cria dados de exemplo com informações temporais ricas para construção civil."""
    logger.info("📝 Criando dados de exemplo com contexto temporal...")
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    docs = [
        {
            "id": "doc_001", "title": "Normas de Ruído - NBR 10151:2019",
            "content": "A NBR 10151, com revisão de 2019, dita as regras de ruído. No período diurno, das 7h às 22h, o limite é de 70 dB. Já no período noturno, das 22h às 7h, o valor não pode exceder 60 dB. A medição de ruído deve ocorrer a cada 2 horas. Relatórios mensais são exigidos para obras com duração superior a 90 dias.",
            "category": "regulamentacao"
        },
        {
            "id": "doc_002", "title": "EPIs Obrigatórios - NR 6",
            "content": "A NR-6 exige o uso de EPIs. O capacete é obrigatório durante todo o expediente. As luvas devem ser inspecionadas diariamente, antes de iniciar o trabalho. Botas de segurança devem ser trocadas a cada 6 meses. O treinamento de segurança ocorre na primeira semana de trabalho de um novo funcionário.",
            "category": "seguranca"
        },
        {
            "id": "doc_003", "title": "Controle do Concreto - NBR 12655",
            "content": "O controle de qualidade do concreto é vital. Ensaios de resistência são feitos a cada 50m³ de concreto utilizado. A cura dos corpos de prova deve durar 28 dias. Após o ensaio, a documentação deve ser arquivada por 5 anos. A concretagem da laje 3 só pode começar 3 dias após a concretagem da laje 2.",
            "category": "qualidade"
        }
    ]
    
    for doc in docs:
        with open(data_dir / f"{doc['id']}.json", 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            
    logger.info(f"✅ Criados {len(docs)} documentos em {data_dir}")
    return docs

def create_temporal_test_questions():
    """Cria perguntas de teste com foco em inferências temporais."""
    questions = [
        {"id": "q001", "question": "Qual o limite de ruído às 23h?"},
        {"id": "q002", "question": "Quando devo inspecionar as luvas de proteção?"},
        {"id": "q003", "question": "Por quanto tempo a documentação do ensaio de concreto deve ser mantida?"},
        {"id": "q004", "question": "Se a laje 2 foi concretada hoje, quando posso concretar a laje 3?"}
    ]
    
    q_path = Path("data/evaluation/temporal_test_questions.json")
    q_path.parent.mkdir(parents=True, exist_ok=True)
    with open(q_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
        
    logger.info(f"✅ Criadas {len(questions)} perguntas de teste.")
    return questions

def run_comparison(system, questions):
    """Roda a comparação e salva os resultados."""
    all_comparisons = []
    for q in questions:
        logger.info("-" * 50)
        comparison = system.compare_all_scenarios(q['question'])
        comparison['question_metadata'] = q
        all_comparisons.append(comparison)
        
        # Log simplificado
        for name, res in comparison['results'].items():
            logger.info(f"  -> {name}: {res.get('answer', 'Erro')[:80]}...")
            
    results_file = Path("data/evaluation/multi_scenario_comparison.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_comparisons, f, ensure_ascii=False, indent=4)
        
    logger.info(f"✅ Resultados da comparação salvos em {results_file}")

def main():
    """Função principal do MVP integrado."""
    logger.info("🚀 EXECUTANDO MVP INTEGRADO - COMPARAÇÃO MULTI-CENÁRIO COM KÙZU")
    if not check_environment(): return
    
    docs = create_enhanced_sample_data()
    questions = create_temporal_test_questions()
    
    system = MultiScenarioSystem()
    system.load_documents(docs)
    system.build_all_scenarios()
    
    run_comparison(system, questions)
    
    logger.info("\n✅ MVP INTEGRADO CONCLUÍDO COM SUCESSO!")
    logger.info("👉 Próximo passo: execute 'streamlit run app/dashboard.py' para ver os resultados.")

if __name__ == "__main__":
    main()
    