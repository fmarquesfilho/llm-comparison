#!/usr/bin/env python3
"""
Script para criar a estrutura de diretórios necessária do projeto
Versão corrigida com criação de arquivos essenciais
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
        "src/pipelines",
        "src/evaluation",
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
    
    # Cria arquivo router.py que está faltando
    router_file = base_dir / "src" / "core" / "router.py"
    if not router_file.exists():
        router_content = '''"""
Query Router para direcionamento de consultas
"""

class QueryRouter:
    """Router simples para direcionamento de consultas"""
    
    def __init__(self):
        pass
    
    def route_query(self, query: str) -> str:
        """Roteia consulta para o método mais apropriado"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['quantos', 'count', 'número']):
            return 'simple'
        elif any(word in query_lower for word in ['padrão', 'pattern', 'correlação', 'porque']):
            return 'complex'
        else:
            return 'medium'
'''
        router_file.write_text(router_content, encoding='utf-8')
        print(f"✅ Criado: {router_file}")
    
    # Cria arquivos de pipeline que estão faltando
    ingestion_file = base_dir / "src" / "pipelines" / "ingestion.py"
    if not ingestion_file.exists():
        ingestion_content = '''"""
Pipeline de ingestão de dados
"""

class DataIngestionPipeline:
    """Pipeline para ingestão de dados de eventos"""
    
    def __init__(self):
        pass
    
    def process_data(self, data):
        """Processa dados de entrada"""
        return data
'''
        ingestion_file.write_text(ingestion_content, encoding='utf-8')
        print(f"✅ Criado: {ingestion_file}")
    
    query_pipeline_file = base_dir / "src" / "pipelines" / "query.py"
    if not query_pipeline_file.exists():
        query_pipeline_content = '''"""
Pipeline de processamento de consultas
"""

class QueryProcessingPipeline:
    """Pipeline para processamento de consultas"""
    
    def __init__(self):
        pass
    
    def process_query(self, query: str):
        """Processa consulta"""
        return query
'''
        query_pipeline_file.write_text(query_pipeline_content, encoding='utf-8')
        print(f"✅ Criado: {query_pipeline_file}")
    
    print("\n🎯 Estrutura de diretórios criada com sucesso!")
    print("\nPróximos passos:")
    print("1. Execute: python setup_project_structure.py")
    print("2. Execute: python run_integrated_mvp.py")
    print("3. Execute: streamlit run app/dashboard.py")

if __name__ == "__main__":
    create_project_structure()