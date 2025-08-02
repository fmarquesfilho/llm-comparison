#!/usr/bin/env python3
"""
Script principal para executar todos os experimentos de comparação LLM
Otimizado para Mac M1 com Metal Performance Shaders
"""

import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import torch
from datetime import datetime

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.data_processing.ingestion import JsonDocumentIngestionPipeline
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.benchmark import BenchmarkGenerator
from src.evaluation.cost_analysis import CostAnalyzer

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self):
        self.config = Config()
        self.config.setup_directories()
        self.results = {}
        
        # Verifica disponibilidade de aceleração
        self._check_hardware()
    
    def _check_hardware(self):
        """Verifica e loga informações de hardware"""
        if torch.backends.mps.is_available():
            logger.info("✅ Metal Performance Shaders (MPS) disponível - M1/M2 Mac")
            self.device = "mps"
        elif torch.cuda.is_available():
            logger.info(f"✅ CUDA disponível - {torch.cuda.device_count()} GPU(s)")
            self.device = "cuda"
        else:
            logger.info("⚠️  Usando CPU - considere usar Colab para aceleração")
            self.device = "cpu"
    
    def step_1_data_ingestion(self):
        """Passo 1: Ingestão e processamento de dados"""
        logger.info("=== PASSO 1: INGESTÃO DE DADOS ===")
        
        data_dir = self.config.DATA_DIR / "raw"
        processed_dir = self.config.DATA_DIR / "processed"
        embeddings_dir = self.config.DATA_DIR / "embeddings"
        
        if not data_dir.exists() or not list(data_dir.iterdir()):
            logger.warning(f"Diretório {data_dir} vazio. Criando dados de exemplo...")
            self._create_sample_data(data_dir)
        
        # Pipeline de ingestão
        pipeline = JsonDocumentIngestionPipeline(self.config)
        
        try:
            pipeline.run(data_dir, embeddings_dir)
            self.results['data_ingestion'] = {
                'status': 'success',
                'num_chunks': len(pipeline.chunks),
                'embedding_dim': pipeline.embeddings.shape[1] if pipeline.embeddings is not None else 0,
                'device_used': pipeline.embedding_model.device
            }
            logger.info("✅ Ingestão de dados concluída")
        except Exception as e:
            logger.error(f"❌ Erro na ingestão: {e}")
            self.results['data_ingestion'] = {'status': 'failed', 'error': str(e)}
            return False
        
        return True
    
    def step_2_create_golden_dataset(self):
        """Passo 2: Criação do dataset golden"""
        logger.info("=== PASSO 2: DATASET GOLDEN ===")
        
        golden_dir = self.config.DATA_DIR / "golden_set"
        golden_file = golden_dir / "golden_dataset.json"
        
        if golden_file.exists():
            logger.info("Dataset golden já existe, carregando...")
            with open(golden_file, 'r', encoding='utf-8') as f:
                golden_data = json.load(f)
        else:
            logger.info("Criando novo dataset golden...")
            
            # Carrega documentos processados
            embeddings_dir = self.config.DATA_DIR / "embeddings"
            docs_file = embeddings_dir / "documents.json"
            
            if not docs_file.exists():
                logger.error("Documentos não encontrados. Execute primeiro a ingestão.")
                return False
            
            docs_df = pd.read_json(docs_file, orient='records')
            documents = docs_df['content'].head(10).tolist()  # Primeiros 10 docs
            
            # Gera dataset golden
            generator = BenchmarkGenerator()
            golden_data = generator.create_golden_dataset(
                documents=documents,
                seed_questions=[
                    "Como monitorar ruído em obras urbanas?",
                    "Quais são os padrões de segurança para canteiros?",
                    "Como calcular impacto ambiental de construções?",
                    "Que equipamentos são necessários para obras noturnas?",
                    "Como implementar controle de qualidade em construção?"
                ],
                output_path=golden_file
            )
        
        self.results['golden_dataset'] = {
            'status': 'success',
            'num_synthetic_qa': len(golden_data.get('synthetic_qa', [])),
            'num_seed_questions': len(golden_data.get('seed_questions', [])),
            'num_adversarial': len(golden_data.get('adversarial_questions', []))
        }
        
        logger.info("✅ Dataset golden preparado")
        return True
    
    def step_3_test_rag_simple(self):
        """Passo 3: Teste do RAG simples"""
        logger.info("=== PASSO 3: RAG SIMPLES ===")
        
        try:
            from src.architectures.rag_simple import SimpleRAGSystem
            
            # Inicializa sistema RAG
            rag_system = SimpleRAGSystem(self.config, model_name="phi-2")
            
            # Carrega índice
            embeddings_dir = self.config.DATA_DIR / "embeddings"
            rag_system.load_index(embeddings_dir)
            
            # Testa com algumas perguntas
            test_questions = [
                "Como monitorar ruído em obras?",
                "Quais equipamentos de segurança são obrigatórios?",
                "Como controlar impacto ambiental na construção?"
            ]
            
            responses = []
            for question in test_questions:
                result = rag_system.query(question)
                responses.append({
                    'question': question,
                    'answer': result['answer'],
                    'num_docs_retrieved': result['metadata']['num_retrieved']
                })
            
            # Salva resultados de teste
            test_results_file = self.config.DATA_DIR / "evaluation" / "rag_simple_test.json"
            with open(test_results_file, 'w', encoding='utf-8') as f:
                json.dump(responses, f, ensure_ascii=False, indent=2)
            
            self.results['rag_simple'] = {
                'status': 'success',
                'test_questions': len(test_questions),
                'avg_retrieved_docs': sum(r['num_docs_retrieved'] for r in responses) / len(responses),
                'model_used': rag_system.model_config.name
            }
            
            logger.info("✅ RAG simples testado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no teste RAG: {e}")
            self.results['rag_simple'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def step_4_evaluation_metrics(self):
        """Passo 4: Avaliação com métricas"""
        logger.info("=== PASSO 4: AVALIAÇÃO MÉTRICAS ===")
        
        try:
            # Carrega dataset golden
            golden_file = self.config.DATA_DIR / "golden_set" / "golden_dataset.json"
            with open(golden_file, 'r', encoding='utf-8') as f:
                golden_data = json.load(f)
            
            # Carrega resultados do RAG
            test_results_file = self.config.DATA_DIR / "evaluation" / "rag_simple_test.json"
            with open(test_results_file, 'r', encoding='utf-8') as f:
                rag_results = json.load(f)
            
            # Prepara dados para avaliação
            predictions = [r['answer'] for r in rag_results]
            references = [r['question'] for r in rag_results]  # Placeholder
            
            # Calcula métricas
            evaluator = EvaluationMetrics()
            metrics = evaluator.evaluate_response_quality(predictions, references)
            
            # Salva métricas
            metrics_file = self.config.DATA_DIR / "evaluation" / "metrics_results.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            
            self.results['evaluation'] = {
                'status': 'success',
                'metrics': metrics
            }
            
            logger.info(f"✅ Avaliação concluída: ROUGE-L: {metrics.get('rougeL', 0):.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na avaliação: {e}")
            self.results['evaluation'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def step_5_cost_analysis(self):
        """Passo 5: Análise de custos"""
        logger.info("=== PASSO 5: ANÁLISE DE CUSTOS ===")
        
        try:
            analyzer = CostAnalyzer()
            
            # Cenários de teste
            scenarios = [
                {'name': 'Piloto', 'queries_per_day': 100},
                {'name': 'Produção Pequena', 'queries_per_day': 1000},
                {'name': 'Produção Média', 'queries_per_day': 5000}
            ]
            
            cost_report = analyzer.compare_architectures(scenarios)
            
            # Salva relatório
            cost_file = self.config.DATA_DIR / "evaluation" / "cost_analysis.csv"
            cost_report.to_csv(cost_file, index=False)
            
            self.results['cost_analysis'] = {
                'status': 'success',
                'scenarios_analyzed': len(scenarios),
                'architectures_compared': len(cost_report['architecture'].unique())
            }
            
            logger.info("✅ Análise de custos concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de custos: {e}")
            self.results['cost_analysis'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def _create_sample_data(self, data_dir: Path):
        """Cria dados de exemplo se não existirem"""
        data_dir.mkdir(parents=True, exist_ok=True)
        
        sample_docs = [
            {
                "content": "O monitoramento de ruído em obras urbanas deve seguir as diretrizes da NBR 10151. Os níveis de ruído não podem exceder 70 dB durante o dia e 60 dB durante a noite em áreas residenciais. É obrigatório o uso de medidores calibrados e registros horários.",
                "title": "Normas de Ruído em Obras",
                "category": "regulamentacao"
            },
            {
                "content": "Equipamentos de proteção individual (EPIs) obrigatórios em canteiros incluem: capacete, óculos de proteção, luvas, calçados de segurança, cintos de segurança para trabalho em altura. A empresa deve fornecer gratuitamente e treinar os trabalhadores.",
                "title": "EPIs Obrigatórios",
                "category": "seguranca"
            },
            {
                "content": "O controle de impacto ambiental em construções requer licenciamento prévio, plano de gerenciamento de resíduos, controle de erosão e proteção de recursos hídricos. Obras acima de 3000m² necessitam de estudo de impacto ambiental.",
                "title": "Impacto Ambiental",
                "category": "meio_ambiente"
            },
            {
                "content": "Obras noturnas requerem autorização especial da prefeitura, iluminação adequada (mínimo 200 lux), sinalização reforçada e equipes especializadas. O horário permitido varia entre 22h e 6h, dependendo da localização.",
                "title": "Obras Noturnas",
                "category": "regulamentacao"
            },
            {
                "content": "Controle de qualidade em construção envolve inspeções regulares, testes de materiais, documentação de procedimentos e auditorias internas. Concreto deve ser testado a cada 50m³ ou a cada dia de concretagem.",
                "title": "Controle de Qualidade",
                "category": "qualidade"
            }
        ]
        
        for i, doc in enumerate(sample_docs):
            doc_file = data_dir / f"documento_{i+1:02d}.json"
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Criados {len(sample_docs)} documentos de exemplo em {data_dir}")
    
    def generate_final_report(self):
        """Gera relatório final dos experimentos"""
        logger.info("=== RELATÓRIO FINAL ===")
        
        report = {
            'experiment_date': datetime.now().isoformat(),
            'hardware_used': self.device,
            'config': {
                'chunk_size': self.config.RAG.chunk_size,
                'embedding_model': self.config.RAG.embedding_model,
                'top_k': self.config.RAG.top_k
            },
            'results': self.results
        }
        
        # Salva relatório completo
        report_file = self.config.DATA_DIR / "evaluation" / "final_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Log sumário
        logger.info("📊 SUMÁRIO DOS RESULTADOS:")
        for step, result in self.results.items():
            status = result.get('status', 'unknown')
            emoji = "✅" if status == "success" else "❌"
            logger.info(f"   {emoji} {step}: {status}")
        
        logger.info(f"📄 Relatório completo salvo em: {report_file}")
        
        return report
    
    def run_all_experiments(self):
        """Executa todos os experimentos em sequência"""
        logger.info("🚀 INICIANDO EXPERIMENTOS LLM COMPARAÇÃO")
        logger.info(f"📱 Hardware: {self.device}")
        
        steps = [
            ("Ingestão de Dados", self.step_1_data_ingestion),
            ("Dataset Golden", self.step_2_create_golden_dataset),
            ("RAG Simples", self.step_3_test_rag_simple),
            ("Avaliação Métricas", self.step_4_evaluation_metrics),
            ("Análise de Custos", self.step_5_cost_analysis)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*50}")
            logger.info(f"Executando: {step_name}")
            logger.info(f"{'='*50}")
            
            try:
                success = step_func()
                if not success:
                    logger.error(f"❌ Falha em: {step_name}")
                    break
            except Exception as e:
                logger.error(f"❌ Erro em {step_name}: {e}")
                break
        
        # Gera relatório final
        self.generate_final_report()
        
        logger.info("\n🎉 EXPERIMENTOS CONCLUÍDOS!")
        logger.info("📊 Execute 'streamlit run app/streamlit_dashboard.py' para ver os resultados")


def main():
    """Função principal"""
    runner = ExperimentRunner()
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
    