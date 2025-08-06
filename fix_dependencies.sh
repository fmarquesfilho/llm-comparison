#!/bin/bash

echo "🔧 Quick Fix Script for Multi-Scenario RAG Dependencies"
echo "========================================================"

# Check if conda environment exists
if conda env list | grep -q "llm-comparison-multi"; then
    echo "✅ Conda environment 'llm-comparison-multi' found"
    
    echo "📦 Updating environment with fixed dependencies..."
    
    # Activate environment and update
    eval "$(conda shell.bash hook)"
    conda activate llm-comparison-multi
    
    # Update specific packages that are causing issues
    echo "🔄 Updating pyarrow..."
    pip install --upgrade "pyarrow>=12.0.0"
    
    echo "🔄 Updating datasets..."
    pip install --upgrade "datasets>=2.14.0"
    
    echo "🔄 Updating sentence-transformers..."
    pip install --upgrade "sentence-transformers>=2.2.0"
    
    echo "🔄 Installing additional dependencies..."
    pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    echo "✅ Dependencies updated successfully!"
    
else
    echo "❌ Conda environment 'llm-comparison-multi' not found"
    echo "Creating environment from scratch..."
    
    # Create environment from updated yml
    conda env create -f environment.yml
    
    echo "✅ Environment created successfully!"
    echo "🔄 Activating environment..."
    
    eval "$(conda shell.bash hook)"
    conda activate llm-comparison-multi
    
    # Install additional packages
    pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "🧪 Testing imports..."

# Test critical imports
python -c "
try:
    import sentence_transformers
    print('✅ sentence-transformers: OK')
except ImportError as e:
    print(f'❌ sentence-transformers: {e}')

try:
    import datasets
    print('✅ datasets: OK')
except ImportError as e:
    print(f'❌ datasets: {e}')

try:
    import pyarrow
    print(f'✅ pyarrow: {pyarrow.__version__}')
except ImportError as e:
    print(f'❌ pyarrow: {e}')

try:
    from src.core.events import DynamicEventUnit
    print('✅ src.core.events: OK')
except ImportError as e:
    print(f'❌ src.core.events: {e}')

try:
    from src.utils.embeddings import get_embedding_model
    print('✅ src.utils.embeddings: OK')
except ImportError as e:
    print(f'❌ src.utils.embeddings: {e}')
"

echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: conda activate llm-comparison-multi"
echo "2. Run the MVP script: python run_integrated_mvp.py"
echo ""

# Test the actual script
echo "🚀 Testing MVP script..."
python run_integrated_mvp.py --test-imports-only 2>/dev/null || echo "⚠️  Run 'python run_integrated_mvp.py' to start the full comparison"

echo ""
echo "✅ Quick fix completed!"