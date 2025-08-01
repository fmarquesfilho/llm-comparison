import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_requirements():
    """Instala dependências Python"""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"
    if requirements_path.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
        logger.info("Dependências instaladas com sucesso")
    else:
        logger.error("Arquivo requirements.txt não encontrado")

def setup_directories():
    """Cria estrutura de diretórios"""
    try:
        from src.config import Config
        Config.setup_directories()
    except ImportError as e:
        logger.warning(f"Não foi possível importar Config para setup de diretórios: {e}")

    additional_dirs = [
        "models/base",
        "models/fine_tuned", 
        "models/embeddings",
        "data/golden_set",
        "notebooks",
        "logs"
    ]
    for directory in additional_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("Estrutura de diretórios criada")

def download_nltk_data():
    """Baixa dados necessários do NLTK"""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("Dados NLTK baixados")
    except ImportError:
        logger.warning("NLTK não está instalado, pulando download dos dados")

def create_env_file():
    """Cria arquivo .env se não existir"""
    env_path = Path(".env")
    if not env_path.exists():
        env_content = (
            "# Configurações do projeto\n"
            "CUDA_VISIBLE_DEVICES=0\n"
            "TOKENIZERS_PARALLELISM=false\n"
            "TRANSFORMERS_CACHE=./cache\n"
            "HF_HOME=./cache\n"
        )
        env_path.write_text(env_content, encoding="utf-8")
        logger.info("Arquivo .env criado")

def main():
    logger.info("Iniciando setup do ambiente...")
    
    install_requirements()
    setup_directories()
    download_nltk_data()
    create_env_file()
    
    logger.info("Setup concluído!")

if __name__ == "__main__":
    main()
