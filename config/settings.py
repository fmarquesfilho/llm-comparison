import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configurations
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')

# Temporal settings
TIME_WINDOW = int(os.getenv('TIME_WINDOW', '300'))  # 5 minutes in seconds
TIME_EMBEDDING_DIM = int(os.getenv('TIME_EMBEDDING_DIM', '64'))

# Path configurations
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
