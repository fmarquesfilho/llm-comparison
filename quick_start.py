#!/usr/bin/env python3
"""
Quick Start - LLM Architecture Comparison
Script para executar rapidamente todo o pipeline no Mac M1

Execute com: python quick_start.py
"""

import subprocess
import sys
import os
from pathlib import Path
import logging
import json
import time

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Verifica se o ambiente está configurado corretamente"""
    logger.info("🔍 Verificando ambiente...")
    
    # Verifica se está em ambiente conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        logger.info(f"✅ Ambiente conda ativo: {conda_env}")
    else:
        logger.warning("⚠️ Ambiente conda não detectado")
    
    # Verifica PyTorch com Metal
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} instalado")
        
        if torch.backends.mps.is_available():
            logger.info("✅ Metal Performance Shaders disponível")
        else:
            logger.warning("⚠️ MPS não disponível - usando CPU")
            
    except ImportError:
        logger.error("❌ PyTorch não instalado")
        return False
    
    # Verifica dependências principais
    required_packages = [
        'transformers', 'sentence_transformers', 'faiss',
        'streamlit', 'gradio', 'pandas', 'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"✅ {package} disponível")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} não encontrado")
    
    if missing_packages:
        logger.error(f"Instale as dependências faltantes: {', '.join(missing_packages)}")
        return False
    
    return True

def setup_directories():
    """Cria diretórios necessários"""
    logger.info("📁 Criando estrutura de diretórios...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "data/evaluation",
        "data/golden_set",
        "models/base",
        "models/fine_tuned",
        "models/embeddings",
        "logs",
        "notebooks"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ {dir_path}")

def create_sample_config():
    """Cria arquivo de configuração de exemplo"""
    logger.info("⚙️ Criando configuração de exemplo...")
    
    config_content = {
        "project_name": "LLM Architecture Comparison",
        "domain": "construcao_civil",
        "models": {
            "embedding": "sentence-transformers/all-mpnet-base-v2",
            "generation": "microsoft/phi-2",
            "fallback": "microsoft/DialoGPT-small"
        },
        "rag_config": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5
        },
        "evaluation": {
            "metrics": ["rouge_l", "bleu", "semantic_similarity", "factual_accuracy"],
            "scenarios": ["piloto", "producao_pequena", "producao_media"]
        }
    }
    
    config_file = Path("config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_content, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Configuração salva em {config_file}")

def run_setup_scripts():
    """Executa scripts de setup"""
    logger.info("🚀 Executando scripts de setup...")
    
    scripts = [
        ("Setup do ambiente", "python scripts/setup_environment.py"),
        ("Download de modelos", "python scripts/download_models.py")
    ]
    
    for name, command in scripts:
        logger.info(f"▶️ {name}...")
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=600  # 10 minutos timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {name} concluído")
            else:
                logger.error(f"❌ {name} falhou: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {name} excedeu tempo limite")
            return False
        except Exception as e:
            logger.error(f"❌ Erro em {name}: {e}")
            return False
    
    return True

def run_main_pipeline():
    """Executa pipeline principal"""
    logger.info("🔄 Executando pipeline principal...")
    
    try:
        result = subprocess.run(
            ["python", "scripts/run_experiments.py"],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutos
        )
        
        if result.returncode == 0:
            logger.info("✅ Pipeline principal concluído")
            return True
        else:
            logger.error(f"❌ Pipeline falhou: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ Pipeline excedeu tempo limite")
        return False
    except Exception as e:
        logger.error(f"❌ Erro no pipeline: {e}")
        return False

def launch_dashboard():
    """Lança dashboard Streamlit"""
    logger.info("🌐 Preparando para lançar dashboard...")
    
    dashboard_file = Path("app/streamlit_dashboard.py")
    
    if not dashboard_file.exists():
        logger.error(f"❌ Dashboard não encontrado: {dashboard_file}")
        return False
    
    logger.info("🚀 Lançando dashboard Streamlit...")
    logger.info("📱 Acesse: http://localhost:8501")
    
    try:
        # Lança Streamlit
        subprocess.run([
            "streamlit", "run", str(dashboard_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        logger.info("👋 Dashboard encerrado pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro ao lançar dashboard: {e}")
        return False
    
    return True

def show_quick_demo():
    """Mostra demo rápido via Gradio"""
    logger.info("🎭 Lançando demo rápido...")
    
    try:
        # Importa e executa demo
        sys.path.append('app')
        from gradio_demo import create_demo, initialize_rag
        
        # Inicializa RAG
        initialize_rag()
        
        # Cria e lança demo
        demo = create_demo()
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except ImportError as e:
        logger.error(f"❌ Erro ao importar demo: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro no demo: {e}")
        return False
    
    return True

def main():
    """Função principal do quick start"""
    
    print("""
🤖 LLM Architecture Comparison - Quick Start
=============================================

Este script irá configurar e executar automaticamente:
1. Verificação do ambiente
2. Setup de diretórios e configurações
3. Download de modelos (se necessário)
4. Execução do pipeline de comparação
5. Lançamento do dashboard interativo

Tempo estimado: 15-30 minutos (dependendo da velocidade de download)
""")
    
    # Confirmação do usuário
    response = input("\n🚀 Prosseguir com o setup automático? [y/N]: ").lower().strip()
    
    if response not in ['y', 'yes', 's', 'sim']:
        print("👋 Setup cancelado pelo usuário")
        return
    
    start_time = time.time()
    
    # Passo 1: Verificar ambiente
    if not check_environment():
        logger.error("❌ Ambiente não está configurado corretamente")
        print("\n📋 Para configurar o ambiente:")
        print("1. conda install pytorch torchvision torchaudio -c pytorch")
        print("2. conda install -c conda-forge sentence-transformers transformers")
        print("3. pip install streamlit gradio faiss-cpu")
        return
    
    # Passo 2: Setup diretórios
    setup_directories()
    
    # Passo 3: Configuração
    create_sample_config()
    
    # Passo 4: Scripts de setup
    print("\n🔧 Executando scripts de configuração...")
    if not run_setup_scripts():
        logger.error("❌ Falha nos scripts de setup")
        return
    
    # Passo 5: Pipeline principal
    print("\n⚙️ Executando pipeline de comparação...")
    if not run_main_pipeline():
        logger.error("❌ Falha no pipeline principal")
        print("ℹ️ Você ainda pode tentar executar manualmente:")
        print("   python scripts/run_experiments.py")
    
    # Tempo total
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ Setup concluído em {elapsed_time/60:.1f} minutos")
    
    # Passo 6: Opções de visualização
    print(f"""
🎉 Setup concluído com sucesso!

Próximos passos:
1. 📊 Dashboard executivo: streamlit run app/streamlit_dashboard.py
2. 🎭 Demo interativo: python app/gradio_demo.py
3. 📝 Notebooks: jupyter lab notebooks/

O que foi criado:
✅ Estrutura de diretórios
✅ Dados de exemplo
✅ Índices de embeddings
✅ Modelo RAG funcional
✅ Análise de custos
✅ Dashboard interativo
""")
    
    # Pergunta se quer lançar dashboard
    launch_choice = input("\n🌐 Lançar dashboard agora? [y/N]: ").lower().strip()
    
    if launch_choice in ['y', 'yes', 's', 'sim']:
        launch_dashboard()
    else:
        print("👋 Para lançar depois: streamlit run app/streamlit_dashboard.py")

if __name__ == "__main__":
    main()
