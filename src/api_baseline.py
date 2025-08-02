# src/api_baseline.py
import os
import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class APIBaseline:
    """Cliente simples para APIs externas (OpenAI, Anthropic)"""
    
    def __init__(self, api_provider: str = "openai"):
        self.api_provider = api_provider
        self.client = None
        self.cost_per_1k_tokens = {
            "openai": 0.002,  # GPT-3.5-turbo aproximado
            "anthropic": 0.003  # Claude Haiku aproximado
        }
        
        self._setup_client()
    
    def _setup_client(self):
        """Configura cliente da API"""
        try:
            if self.api_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("⚠️ OPENAI_API_KEY não encontrada")
                    return False
                
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ Cliente OpenAI configurado")
                
            elif self.api_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("⚠️ ANTHROPIC_API_KEY não encontrada")
                    return False
                
                # Para o MVP, vamos simular o cliente Anthropic
                logger.info("✅ Cliente Anthropic simulado (implementar se necessário)")
                
            return True
            
        except ImportError as e:
            logger.error(f"❌ Biblioteca da API não instalada: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erro na configuração API: {e}")
            return False
    
    def create_context_from_docs(self, documents: List[Dict], top_k: int = 3) -> str:
        """Cria contexto a partir dos documentos mais relevantes"""
        context_docs = documents[:top_k]
        
        context = "Documentos relevantes:\n\n"
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', str(doc))
            # Limita tamanho para não estourar limite de tokens
            if len(content) > 400:
                content = content[:400] + "..."
            context += f"Documento {i}:\n{content}\n\n"
        
        return context
    
    def query_openai(self, question: str, context: str = "") -> Dict[str, Any]:
        """Consulta OpenAI GPT"""
        if not self.client:
            return {
                "answer": "❌ Cliente OpenAI não configurado",
                "error": "API key não encontrada"
            }
        
        try:
            prompt = f"""Com base nos documentos fornecidos, responda a pergunta de forma clara e precisa.

{context}

Pergunta: {question}

Resposta:"""
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Você é um assistente especializado em construção civil. Responda com base apenas nas informações fornecidas."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            
            answer = response.choices[0].message.content.strip()
            
            # Estima custo baseado em tokens
            total_tokens = response.usage.total_tokens
            estimated_cost = (total_tokens / 1000) * self.cost_per_1k_tokens["openai"]
            
            return {
                "answer": answer,
                "response_time": response_time,
                "tokens_used": total_tokens,
                "estimated_cost_usd": estimated_cost,
                "model": "gpt-3.5-turbo"
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta OpenAI: {e}")
            return {
                "answer": f"❌ Erro na API: {str(e)}",
                "error": str(e)
            }
    
    def query_anthropic_mock(self, question: str, context: str = "") -> Dict[str, Any]:
        """Mock do Anthropic Claude para demonstração"""
        # Para o MVP, simula resposta baseada no contexto
        start_time = time.time()
        
        # Resposta simulada simples
        if "ruído" in question.lower():
            answer = "Baseado nos documentos, os limites de ruído são 70 dB durante o dia e 60 dB à noite em áreas residenciais, conforme NBR 10151."
        elif "epi" in question.lower() or "proteção" in question.lower():
            answer = "Os EPIs obrigatórios incluem capacete, óculos de proteção, luvas, calçados de segurança e cintos para trabalho em altura."
        elif "concreto" in question.lower() or "qualidade" in question.lower():
            answer = "O controle de qualidade do concreto requer ensaios a cada 50m³ ou a cada dia de concretagem."
        else:
            answer = "Baseado nos documentos fornecidos, não encontrei informação específica para responder sua pergunta."
        
        response_time = time.time() - start_time + 0.5  # Simula latência
        
        # Estima tokens e custo
        estimated_tokens = len(question.split()) + len(answer.split()) + len(context.split())
        estimated_cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens["anthropic"]
        
        return {
            "answer": answer,
            "response_time": response_time,
            "tokens_used": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": "claude-3-haiku (mock)"
        }
    
    def query(self, question: str, documents: List[Dict] = None) -> Dict[str, Any]:
        """Interface principal para consultas à API"""
        
        # Cria contexto se documentos fornecidos
        context = ""
        if documents:
            context = self.create_context_from_docs(documents)
        
        # Chama API apropriada
        if self.api_provider == "openai":
            result = self.query_openai(question, context)
        elif self.api_provider == "anthropic":
            result = self.query_anthropic_mock(question, context)
        else:
            result = {
                "answer": f"❌ Provedor '{self.api_provider}' não suportado",
                "error": "Provedor inválido"
            }
        
        # Adiciona metadados
        result.update({
            "question": question,
            "api_provider": self.api_provider,
            "context_provided": len(context) > 0,
            "num_context_docs": len(documents) if documents else 0
        })
        
        return result
    
    def batch_query(self, questions: List[str], documents: List[Dict] = None) -> List[Dict[str, Any]]:
        """Processa múltiplas perguntas"""
        results = []
        
        for question in questions:
            result = self.query(question, documents)
            results.append(result)
            
            # Pequena pausa para evitar rate limit
            time.sleep(0.1)
        
        return results
    
    def get_cost_estimate(self, num_queries: int, avg_tokens_per_query: int = 200) -> Dict[str, float]:
        """Estima custos para um volume de consultas"""
        
        cost_per_1k = self.cost_per_1k_tokens.get(self.api_provider, 0.002)
        
        total_tokens = num_queries * avg_tokens_per_query
        total_cost = (total_tokens / 1000) * cost_per_1k
        
        return {
            "num_queries": num_queries,
            "avg_tokens_per_query": avg_tokens_per_query,
            "total_tokens": total_tokens,
            "cost_per_1k_tokens": cost_per_1k,
            "total_cost_usd": total_cost,
            "cost_per_query_usd": total_cost / num_queries if num_queries > 0 else 0
        }


def test_api_baseline():
    """Função de teste para baseline API"""
    logger.info("🧪 Testando baseline API...")
    
    # Carrega documentos de teste
    data_dir = Path("data/raw")
    documents = []
    
    if data_dir.exists():
        for json_file in data_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
    
    # Carrega perguntas de teste
    questions_file = Path("data/evaluation/test_questions.json")
    if not questions_file.exists():
        logger.error("❌ Perguntas de teste não encontradas")
        return []
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)
    
    # Testa OpenAI (se disponível)
    openai_baseline = APIBaseline("openai")
    results = []
    
    for q in test_questions:
        result = openai_baseline.query(q['question'], documents)
        
        # Adiciona informações da pergunta
        result.update({
            'question_id': q['id'],
            'expected_concepts': q['expected_concepts'],
            'category': q['category']
        })
        
        results.append(result)
        
        logger.info(f"API {q['id']}: {result.get('response_time', 0):.2f}s, "
                   f"custo: ${result.get('estimated_cost_usd', 0):.4f}")
    
    # Salva resultados
    results_file = Path("data/evaluation/api_baseline_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Baseline API testado, resultados em {results_file}")
    return results


if __name__ == "__main__":
    # Teste standalone
    logging.basicConfig(level=logging.INFO)
    test_api_baseline()
    