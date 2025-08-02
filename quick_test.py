#!/usr/bin/env python3
"""
Quick Test - LLM Architecture Comparison MVP
Teste rápido para verificar se o sistema está funcionando

Execute com: python quick_test.py
"""

import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Testa imports essenciais"""
    logger.info("🧪 Testando imports...")
    
    try:
        import torch
        import sentence_transformers
        import faiss
        import pandas as pd
        import numpy as np
        logger.info("✅ Imports básicos OK")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_torch_functionality():
    """Testa funcionalidade básica do PyTorch"""
    logger.info("🔥 Testando PyTorch...")
    
    try:
        import torch
        
        # Testa tensor básico
        x = torch.randn(2, 3)
        logger.info(f"  • Tensor criado: {x.shape}")
        
        # Testa device
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("  • Device: MPS (Mac M1/M2)")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("  • Device: CUDA")
        else:
            device = "cpu"
            logger.info("  • Device: CPU")
        
        # Testa operação no device
        x = x.to(device)
        y = torch.matmul(x, x.T)
        logger.info(f"  • Operação no {device}: OK")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch error: {e}")
        return False

def test_sentence_transformers():
    """Testa sentence-transformers com modelo leve"""
    logger.info("🔤 Testando Sentence Transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Carrega modelo leve
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("  • Modelo carregado: all-MiniLM-L6-v2")
        
        # Testa encoding
        sentences = ["Teste de embedding", "Outro texto de exemplo"]
        embeddings = model.encode(sentences)
        
        logger.info(f"  • Embeddings shape: {embeddings.shape}")
        logger.info(f"  • Dimensão: {embeddings.shape[1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sentence Transformers error: {e}")
        return False

def test_faiss():
    """Testa FAISS para busca vetorial"""
    logger.info("🔍 Testando FAISS...")
    
    try:
        import faiss
        import numpy as np
        
        # Cria dados de teste
        dimension = 384  # Dimensão do all-MiniLM-L6-v2
        n_vectors = 5
        
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # Cria índice
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        logger.info(f"  • Índice criado: {index.ntotal} vetores")
        
        # Testa busca
        query = np.random.random((1, dimension)).astype('float32')
        distances, indices = index.search(query, k=3)
        
        logger.info(f"  • Busca OK: {len(indices[0])} resultados")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FAISS error: {e}")
        return False

def test_config():
    """Testa módulo de configuração"""
    logger.info("⚙️ Testando configuração...")
    
    try:
        sys.path.append('src')
        from config import Config
        
        # Testa criação de config
        config = Config()
        logger.info("  • Config criada")
        
        # Testa detecção de device
        device = config.get_device()
        logger.info(f"  • Device detectado: {device}")
        
        # Testa setup de diretórios
        config.setup_directories()
        logger.info("  • Diretórios criados")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return False

def test_simple_pipeline():
    """Testa pipeline simplificado RAG"""
    logger.info("🤖 Testando pipeline RAG...")
    
    try:
        sys.path.append('src')
        from config import Config
        from simple_rag import SimpleRAGSystem
        
        # Inicializa sistema
        config = Config()
        rag_system = SimpleRAGSystem(config)
        
        logger.info("  • RAG System inicializado")
        
        # Testa info do sistema
        info = rag_system.get_system_info()
        logger.info(f"  • Device: {info['device']}")
        logger.info(f"  • Modelo: {info['embedding_model']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG pipeline error: {e}")
        return False

def test_data_creation():
    """Testa criação de dados sintéticos"""
    logger.info("📝 Testando criação de dados...")
    
    try:
        # Simula criação de dados simples
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        sample_doc = {
            "id": "test_001",
            "title": "Documento de Teste",
            "content": "Este é um documento de teste para validar a funcionalidade básica do sistema.",
            "category": "teste"
        }
        
        import json
        test_file = data_dir / "test_doc.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(sample_doc, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  • Documento teste criado: {test_file}")
        
        # Remove arquivo de teste
        test_file.unlink()
        logger.info("  • Limpeza OK")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data creation error: {e}")
        return False

def test_streamlit_import():
    """Testa se Streamlit pode ser importado"""
    logger.info("🖥️ Testando Streamlit...")
    
    try:
        import streamlit as st
        logger.info("  • Streamlit importado OK")
        
        # Verifica se arquivo do dashboard existe
        dashboard_file = Path("app/dashboard.py")
        if dashboard_file.exists():
            logger.info("  • Dashboard encontrado")
        else:
            logger.warning("  ⚠️ Dashboard não encontrado")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Streamlit error: {e}")
        return False

def run_performance_test():
    """Executa teste básico de performance"""
    logger.info("⚡ Teste de performance...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Carrega modelo
        start_time = time.time()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        
        logger.info(f"  • Carregamento modelo: {load_time:.2f}s")
        
        # Testa encoding
        sentences = ["Teste de performance"] * 10
        
        start_time = time.time()
        embeddings = model.encode(sentences)
        encode_time = time.time() - start_time
        
        logger.info(f"  • Encoding 10 frases: {encode_time:.2f}s")
        logger.info(f"  • Velocidade: {len(sentences)/encode_time:.1f} frases/s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test error: {e}")
        return False

def main():
    """Executa todos os testes"""
    logger.info("🚀 QUICK TEST - LLM Architecture Comparison MVP")
    logger.info("="*50)
    
    tests = [
        ("Imports básicos", test_imports),
        ("PyTorch", test_torch_functionality),
        ("Sentence Transformers", test_sentence_transformers),
        ("FAISS", test_faiss),
        ("Configuração", test_config),
        ("Pipeline RAG", test_simple_pipeline),
        ("Criação de dados", test_data_creation),
        ("Streamlit", test_streamlit_import),
        ("Performance", run_performance_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n▶️ {test_name}...")
        
        start_time = time.time()
        try:
            success = test_func()
            test_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ {test_name} OK ({test_time:.2f}s)")
                results.append((test_name, True, test_time))
            else:
                logger.error(f"❌ {test_name} FAILED ({test_time:.2f}s)")
                results.append((test_name, False, test_time))
                
        except Exception as e:
            test_time = time.time() - start_time
            logger.error(f"❌ {test_name} ERROR: {e} ({test_time:.2f}s)")
            results.append((test_name, False, test_time))
    
    # Resumo final
    logger.info("\n" + "="*50)
    logger.info("📊 RESUMO DOS TESTES")
    logger.info("="*50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_time = sum(t for _, _, t in results)
    
    logger.info(f"✅ Aprovados: {passed}/{total}")
    logger.info(f"⏱️ Tempo total: {total_time:.2f}s")
    
    if passed == total:
        logger.info("\n🎉 TODOS OS TESTES PASSARAM!")
        logger.info("💡 Sistema pronto para executar: python run_mvp.py")
        return True
    else:
        failed = total - passed
        logger.error(f"\n❌ {failed} TESTES FALHARAM")
        logger.info("💡 Verifique os erros acima e execute: python setup.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
    