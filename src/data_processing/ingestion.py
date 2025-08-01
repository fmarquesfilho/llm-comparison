# src/data_processing/ingestion.py
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)

class JsonDocumentIngestionPipeline:
    def __init__(self, config: Config):
        """
        Inicializa pipeline de ingestão JSON.

        Args:
            config (Config): Configurações do projeto contendo parâmetros para chunking e modelos.
        """
        self.config = config
        self.embedding_model = SentenceTransformer(config.RAG.embedding_model)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = None
        self.index = None

    def load_json_documents(self, data_dir: Path, content_field: str = "content") -> List[Dict[str, Any]]:
        """
        Carrega documentos JSON do diretório, extraindo conteúdo textual.

        Args:
            data_dir (Path): Diretório contendo arquivos JSON.
            content_field (str): Nome do campo para extrair texto, se o JSON for dict.

        Returns:
            List[Dict[str, Any]]: Lista de documentos carregados com campos 'content', 'source' e 'metadata'.
        """
        json_docs = []
        skipped_files = 0
        for file_path in data_dir.rglob("*.json"):
            try:
                with file_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)

                # Suporta lista de documentos no arquivo JSON
                if isinstance(data, list):
                    for item in data:
                        text = item.get(content_field) if isinstance(item, dict) else str(item)
                        if not text or len(str(text).strip()) < 20:
                            continue
                        json_docs.append({
                            'content': str(text),
                            'source': str(file_path),
                            'metadata': {'file_type': '.json'}
                        })
                else:
                    text = data.get(content_field) if isinstance(data, dict) else str(data)
                    if not text or len(str(text).strip()) < 20:
                        skipped_files += 1
                        logger.warning(f"Pulado documento vazio ou muito curto em {file_path}")
                        continue
                    json_docs.append({
                        'content': str(text),
                        'source': str(file_path),
                        'metadata': {'file_type': '.json'}
                    })

            except Exception as e:
                skipped_files += 1
                logger.error(f"Erro ao ler {file_path}: {e}")
        logger.info(f"Carregados {len(json_docs)} documentos JSON. Pulados/ignorados: {skipped_files}")
        return json_docs

    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = None,
                        overlap: int = None) -> List[Dict[str, Any]]:
        """
        Fragmenta textos em chunks com sobreposição, evitando cortes no meio de palavras.

        Args:
            documents (List[Dict[str, Any]]): Lista de documentos com campo 'content'.
            chunk_size (int, optional): Tamanho máximo do chunk. Se None, usa da config.
            overlap (int, optional): Sobreposição entre chunks. Se None, usa da config.

        Returns:
            List[Dict[str, Any]]: Lista de chunks com texto fragmentado e metadados.
        """
        chunk_size = chunk_size or getattr(self.config.RAG, 'chunk_size', 512)
        overlap = overlap or getattr(self.config.RAG, 'chunk_overlap', 50)

        # Segurança para evitar valores inválidos
        chunk_size = max(chunk_size, 256)
        overlap = min(overlap, chunk_size - 1)

        chunks = []
        for doc in tqdm(documents, desc="Fragmentando documentos"):
            text = doc['content']
            text_len = len(text)
            start_idx = 0
            step = chunk_size - overlap

            while start_idx < text_len:
                end_idx = min(start_idx + chunk_size, text_len)

                # Correção para evitar quebra no meio de palavra
                if end_idx < text_len:
                    last_space = text.rfind(' ', start_idx, end_idx)
                    if last_space > start_idx:
                        end_idx = last_space

                chunk_text = text[start_idx:end_idx].strip()
                if len(chunk_text) >= 50:
                    chunks.append({
                        'content': chunk_text,
                        'source': doc['source'],
                        'chunk_id': len(chunks),
                        'metadata': doc['metadata']
                    })

                start_idx = end_idx - overlap  # move para o próximo segmento com overlap

        logger.info(f"Criados {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Gera embeddings para os chunks usando o modelo definido na configuração.

        Args:
            chunks (List[Dict[str, Any]]): Lista de chunks a serem embutidos.

        Returns:
            np.ndarray: Matriz de embeddings numpy
        """
        texts = [chunk['content'] for chunk in chunks]
        batch_size = min(32, max(1, len(chunks) // 10))
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        logger.info(f"Criados embeddings com dimensão {embeddings.shape}")
        return embeddings

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Constrói índice FAISS com embeddings.

        Args:
            embeddings (np.ndarray): Embeddings dos chunks.

        Returns:
            faiss.IndexFlatL2: Índice FAISS criado
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype('float32'))
        logger.info(f"Índice FAISS criado com {index.ntotal} vetores")
        return index

    def save_index_and_metadata(self, save_dir: Path):
        """
        Salva o índice FAISS e os metadados dos chunks em disco.

        Args:
            save_dir (Path): Diretório onde salvar.
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        pd.DataFrame(self.chunks).to_json(save_dir / "documents.json", orient='records', indent=2)
        logger.info(f"Índice e metadados salvos em {save_dir}")

    def run(self, data_dir: Path, save_dir: Path):
        """
        Executa pipeline completa: carregar, fragmentar, embutir, indexar e salvar.

        Args:
            data_dir (Path): Diretório com arquivos JSON.
            save_dir (Path): Diretório para salvar índice e metadados.
        """
        documents = self.load_json_documents(data_dir)
        self.chunks = self.chunk_documents(documents,
                                           chunk_size=getattr(self.config.RAG, 'chunk_size', 512),
                                           overlap=getattr(self.config.RAG, 'chunk_overlap', 50))
        self.embeddings = self.create_embeddings(self.chunks)
        self.index = self.build_faiss_index(self.embeddings)
        self.save_index_and_metadata(save_dir)
