#!/usr/bin/env python3
"""
Setup Script - LLM Architecture Comparison MVP
Script para configuração inicial rápida do ambiente

Execute com: python setup.py
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Verifica se a versão do Python é adequada"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        logger.error(f"❌ Python 3.8+ necessário. Versão atual: {version.major}.{version.minor}")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def create_virtual_env():
    """Cria ambiente virtual se não existir"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        logger.info("✅ Ambiente virtual já existe")
        return True
    
    try:
        logger.info("📦 Criando ambiente virtual...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        logger.info("✅ Ambiente virtual criado")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao criar ambiente virtual: {e}")
        return False

def get_pip_command():
    """Retorna comando pip correto para o ambiente"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\pip"
    else:  # macOS/Linux
        return "venv/bin/pip"

def install_requirements():
    """Instala dependências do requirements.txt"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        logger.error("❌ Arquivo requirements.txt não encontrado")
        return False
    
    try:
        pip_cmd = get_pip_command()
        logger.info("📥 Instalando dependências...")
        
        # Upgrade pip primeiro
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Instala requirements
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        logger.info("✅ Dependências instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erro ao instalar dependências: {e}")
        logger.info("💡 Tente manualmente: pip install -r requirements.txt")
        return False

def create_directories():
    """Cria estrutura de diretórios necessária"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "data/evaluation",
        "logs",
        "cache"
    ]
    
    logger.info("📁 Criando diretórios...")
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Estrutura de diretórios criada")

def create_env_file():
    """Cria arquivo .env a partir do exemplo"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        logger.info("✅ Arquivo .env já existe")
        return
    
    if env_example.exists():
        try:
            # Copia exemplo para .env
            with open(env_example, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            logger.info("✅ Arquivo .env criado a partir do exemplo")
            logger.info("💡 Configure suas API keys em .env se necessário")
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar .env: {e}")
    else:
        # Cria .env básico
        basic_env = """# LLM Architecture Comparison MVP
# Configure conforme necessário

# API Keys (opcional para MVP)
OPENAI_API_KEY=

# Configuracoes de ambiente
PYTORCH_ENABLE_MPS_FALLBACK=1
TOKENIZERS_PARALLELISM=false
LOG_LEVEL=INFO
"""
        
        with open(env_file, 'w') as f:
            f.write(basic_env)
        
        logger.info("✅ Arquivo .env básico criado")

def test_imports():
    """Testa se as principais dependências foram instaladas"""
    logger.info("🧪 Testando imports principais...")
    
    test_packages = [
        "torch",
        "transformers", 
        "sentence_transformers",
        "faiss",
        "pandas",
        "numpy",
        "streamlit"
    ]
    
    failed_imports = []
    
    for package in test_packages:
        try:
            __import__(package.replace('-', '_'))
            logger.info(f"  ✅ {package}")
        except ImportError:
            logger.error(f"  ❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"❌ Pacotes não importados: {', '.join(failed_imports)}")
        return False
    
    logger.info("✅ Todos os imports funcionando")
    return True

def test_torch_device():
    """Testa dispositivos disponíveis para PyTorch"""
    try:
        import torch
        
        logger.info("🔍 Testando dispositivos PyTorch...")
        logger.info(f"  • PyTorch version: {torch.__version__}")
        
        # CPU sempre disponível
        logger.info("  ✅ CPU disponível")
        
        # Testa CUDA
        if torch.cuda.is_available():
            logger.info(f"  ✅ CUDA disponível: {torch.cuda.device_count()} GPU(s)")
        else:
            logger.info("  ⚠️ CUDA não disponível")
        
        # Testa MPS (Mac M1/M2)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("  ✅ MPS (Metal) disponível - Mac M1/M2")
        else:
            logger.info("  ⚠️ MPS não disponível")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar PyTorch: {e}")
        return False

def run_quick_validation():
    """Executa validação rápida do MVP"""
    logger.info("🚀 Executando validação rápida...")
    
    try:
        # Tenta importar configuração
        sys.path.append('src')
        from config import Config
        
        # Valida setup
        if Config.validate_setup():
            logger.info("✅ Configuração validada")
            return True
        else:
            logger.warning("⚠️ Configuração com problemas")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        return False

def show_next_steps():
    """Mostra próximos passos após setup"""
    logger.info("\n" + "="*50)
    logger.info("🎉 SETUP CONCLUÍDO!")
    logger.info("="*50)
    
    logger.info("\n📋 Próximos passos:")
    logger.info("1. Ative o ambiente virtual:")
    if os.name == 'nt':
        logger.info("   venv\\Scripts\\activate")
    else:
        logger.info("   source venv/bin/activate")
    
    logger.info("\n2. Execute o MVP completo:")
    logger.info("   python run_mvp.py")
    
    logger.info("\n3. Visualize resultados:")
    logger.info("   streamlit run app/dashboard.py")
    
    logger.info("\n4. (Opcional) Configure API keys:")
    logger.info("   Edite o arquivo .env")
    
    logger.info("\n💡 Documentação completa: README.md")

def main():
    """Função principal do setup"""
    logger.info("🚀 SETUP - LLM Architecture Comparison MVP")
    logger.info("="*50)
    
    steps = [
        ("Verificar Python", check_python_version),
        ("Criar ambiente virtual", create_virtual_env),
        ("Instalar dependências", install_requirements), 
        ("Criar diretórios", create_directories),
        ("Configurar .env", create_env_file),
        ("Testar imports", test_imports),
        ("Testar PyTorch", test_torch_device),
        ("Validar configuração", run_quick_validation)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        logger.info(f"\n▶️ {step_name}...")
        try:
            if not step_func():
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"❌ Erro em {step_name}: {e}")
            failed_steps.append(step_name)
    
    # Resultado final
    if failed_steps:
        logger.error(f"\n❌ Setup com problemas nas etapas: {', '.join(failed_steps)}")
        logger.info("\n💡 Você ainda pode tentar executar manualmente:")
        logger.info("   python run_mvp.py")
        return False
    else:
        show_next_steps()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    