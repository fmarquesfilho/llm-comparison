# Setup para Mac M1 - Execute no seu ambiente conda ativo

# Ativar ambiente (substitua pelo nome do seu ambiente)
conda activate otoh-llm

# Instalar dependências específicas para M1
conda install -c conda-forge sentence-transformers
conda install -c conda-forge transformers
conda install -c conda-forge datasets
conda install -c conda-forge accelerate
conda install -c conda-forge peft
conda install pandas numpy scikit-learn matplotlib seaborn plotly
conda install -c conda-forge streamlit gradio
conda install -c conda-forge rouge-score nltk

# Instalar dependências via pip (algumas funcionam melhor via pip no M1)
pip install torch-audio  # caso tenha problemas com torchaudio
pip install bitsandbytes-darwin  # versão específica para macOS
pip install faiss-gpu  # versão GPU para Metal
pip install haystack-ai
pip install python-docx PyPDF2  # para processamento de documentos
pip install python-dotenv  # para .env

# Verificar instalação do PyTorch com Metal
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'MPS built: {torch.backends.mps.is_built()}')"

echo "✅ Setup M1 concluído! Execute o próximo script para verificar GPU acceleration."
