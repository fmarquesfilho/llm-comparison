#!/usr/bin/env python3
"""
Script para criar a estrutura de diretórios necessária do projeto
"""

import os
from pathlib import Path

def create_project_structure():
    """Cria a estrutura de diretórios necessária"""
    
    # Diretório base do projeto
    base_dir = Path(__file__).parent
    
    # Estrutura de diretórios necessária
    directories = [
        "data",
        "data/raw",
        "data/processed", 
        "data/evaluation",
        "src",
        "src/core",
        "src/utils",
        "src/config",
        "app"
    ]
    
    print("🚀 Criando estrutura de diretórios do projeto...")
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Criado: {dir_path}")
        
        # Cria __init__.py nos diretórios Python se não existir
        if directory.startswith("src"):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"✅ Criado: {init_file}")
    
    print("\n🎯 Estrutura de diretórios criada com sucesso!")
    print("\nPróximos passos:")
    print("1. Execute: python setup_project_structure.py")
    print("2. Execute: python run_integrated_mvp.py")
    print("3. Execute: streamlit run app/dashboard.py")

if __name__ == "__main__":
    create_project_structure()