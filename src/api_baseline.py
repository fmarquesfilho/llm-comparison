# src/api_baseline.py
import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SmartMockAPI:
    """Mock inteligente que simula APIs reais de forma mais realista"""
    
    def __init__(self, provider_name: str = "openai_mock"):
        self.provider_name = provider_name
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self) -> Dict[str, str]:
        """Constrói base de conhecimento para respostas mais inteligentes"""
        return {
            "ruído": "Conforme a NBR 10151, os limites de ruído são 70 dB durante o dia (7h às 22h) e 60 dB durante a noite (22h às 7h) em áreas residenciais. A medição deve ser feita com medidores calibrados no limite da propriedade mais próxima aos receptores sensíveis.",
            
            "epi": "Os EPIs obrigatórios em canteiros incluem: capacete de segurança classe A ou B, óculos de proteção contra impactos, luvas adequadas à atividade, calçados de segurança com biqueira de aço, e cintos de segurança para trabalhos em altura superior a 2 metros. A empresa deve fornecer gratuitamente e treinar os trabalhadores.",
            
            "concreto": "O controle de qualidade do concreto estrutural requer ensaios de resistência à compressão a cada 50m³ ou a cada dia de concretagem, prevalecendo o menor valor. Devem ser realizadas inspeções regulares dos materiais e documentação completa de todos os procedimentos.",
            
            "obras noturnas": "Obras noturnas necessitam autorização especial da prefeitura, iluminação mínima de 200 lux, sinalização reforçada com dispositivos refletivos e luminosos, e equipes especializadas. O horário permitido geralmente é das 22h às 6h, dependendo da área urbana.",
            
            "eia": "O Estudo de Impacto Ambiental (EIA/RIMA) é obrigatório para obras com área superior a 3000m². Também é necessário licenciamento prévio, plano de gerenciamento de resíduos sólidos e medidas de controle de erosão e proteção de recursos hídricos.",
            
            "qualidade": "O controle de qualidade em construção envolve inspeções regulares dos materiais, ensaios de resistência, documentação completa e auditorias internas periódicas. Para concreto estrutural, os ensaios devem ser realizados conforme cronograma específico baseado no volume ou tempo de concretagem."
        }
    
    def _find_best_match(self, question: str) -> str:
        """Encontra a melhor resposta baseada na pergunta"""
        question_lower = question.lower()
        
        # Procura palavras-chave na pergunta
        for keyword, answer in self.knowledge_base.items():
            if keyword in question_lower:
                return answer
        
        # Se não encontrar match específico, resposta genérica
        return "Com base nos documentos fornecidos, posso ajudar com informações sobre normas de construção civil, EPIs, controle de qualidade, impacto ambiental e regulamentações. Poderia reformular sua pergunta de forma mais específica?"
    
    def _simulate_response_delay(self) -> float:
        """Simula delay realista de API"""
        # Simula latência de rede + processamento
        base_delay = random.uniform(0.5, 1.5)  # Latência base
        processing_delay = random.uniform(0.8, 2.2)  # Processamento LLM
        return base_delay + processing_delay
    
    def _estimate_tokens(self, question: str, answer: str, context: str = "") -> int:
        """Estima tokens de forma mais realista"""
        # Estimativa aproximada: ~4 chars por token
        prompt_tokens = len(f"Sistema: Você é um assistente...\nContexto: {context}\nPergunta: {question}") // 4
        completion_tokens = len(answer) // 4
        return prompt_tokens + completion_tokens

class APIBaseline:
    """Cliente para APIs externas com mock inteligente quando APIs não disponíveis"""
    
    def __init__(self, api_provider: str = "openai"):
        self.api_provider = api_provider
        self.client = None
        self.mock_api = SmartMockAPI(f"{api_provider}_mock")
        self.use_mock = False
        
        self.cost_per_1k_tokens = {
            "openai": 0.002,  # GPT-3.5-turbo
            "anthropic": 0.003,  # Claude Haiku
            "openai_mock": 0.002,  # Mesmo custo para comparação
            "anthropic_mock": 0.003
        }
        
        self._setup_client()
    
    def _setup_client(self):
        """Configura cliente da API real ou mock"""
        try:
            if self.api_provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("⚠️ OPENAI_API_KEY não encontrada - usando mock inteligente")
                    self.use_mock = True
                    return True
                
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                
                # Testa conexão com request simples
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1
                    )
                    logger.info("✅ Cliente OpenAI configurado e testado")
                    return True
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao testar OpenAI (quota/limite?): {e}")
                    logger.info("🔄 Usando mock inteligente")
                    self.use_mock = True
                    return True
                
            elif self.api_provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("⚠️ ANTHROPIC_API_KEY não encontrada - usando mock")
                    self.use_mock = True
                    return True
                
                # Para o MVP, sempre usar mock para Anthropic
                logger.info("🔄 Usando mock Anthropic (implementar cliente real se necessário)")
                self.use_mock = True
                return True
                
            return True
            
        except ImportError as e:
            logger.warning(f"⚠️ Biblioteca da API não instalada: {e} - usando mock")
            self.use_mock = True
            return True
        except Exception as e:
            logger.warning(f"⚠️ Erro na configuração API: {e} - usando mock")
            self.use_mock = True
            return True
    
    def create_context_from_docs(self, documents: List[Dict], top_k: int = 3) -> str:
        """Cria contexto a partir dos documentos mais relevantes"""
        if not documents:
            return ""
        
        context_docs = documents[:top_k]
        
        context = "Documentos relevantes:\n\n"
        for i, doc in enumerate(context_docs, 1):
            content = doc.get('content', str(doc))
            # Limita tamanho para não estourar limite de tokens
            if len(content) > 400:
                content = content[:400] + "..."
            context += f"Documento {i}: {doc.get('title', 'Sem título')}\n{content}\n\n"
        
        return context
    
    def query_openai(self, question: str, context: str = "") -> Dict[str, Any]:
        """Consulta OpenAI GPT (real ou mock)"""
        
        if self.use_mock:
            return self._query_mock(question, context, "openai")
        
        if not self.client:
            return {
                "answer": "❌ Cliente OpenAI não configurado",
                "error": "API key não encontrada"
            }
        
        try:
            system_prompt = "Você é um assistente especializado em construção civil e engenharia. Responda com base apenas nas informações fornecidas nos documentos."
            
            prompt = f"""Com base nos documentos fornecidos, responda a pergunta de forma clara e precisa.

{context}

Pergunta: {question}

Resposta:"""
            
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
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
                "model": "gpt-3.5-turbo",
                "source": "openai_api"
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta OpenAI: {e}")
            logger.info("🔄 Fallback para mock")
            return self._query_mock(question, context, "openai")
    
    def _query_mock(self, question: str, context: str, provider: str) -> Dict[str, Any]:
        """Query usando mock inteligente"""
        start_time = time.time()
        
        # Simula delay realista
        delay = self.mock_api._simulate_response_delay()
        time.sleep(delay)
        
        # Gera resposta inteligente
        answer = self.mock_api._find_best_match(question)
        
        # Se há contexto, tenta usar informações dos documentos
        if context and len(context) > 50:
            # Extrai algumas informações do contexto
            context_keywords = ["NBR", "dB", "EPIs", "capacete", "50m³", "200 lux", "3000m²"]
            found_keywords = [kw for kw in context_keywords if kw in context]
            
            if found_keywords:
                answer = f"Baseado nos documentos fornecidos: {answer}"
                if len(found_keywords) >= 2:
                    answer += f" (Referências encontradas: {', '.join(found_keywords[:3])})"
        
        response_time = time.time() - start_time
        
        # Estima tokens e custo
        estimated_tokens = self.mock_api._estimate_tokens(question, answer, context)
        estimated_cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens.get(f"{provider}_mock", 0.002)
        
        return {
            "answer": answer,
            "response_time": response_time,
            "tokens_used": estimated_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": f"{provider}-mock",
            "source": f"{provider}_mock",
            "mock_used": True
        }
    
    def query_anthropic_mock(self, question: str, context: str = "") -> Dict[str, Any]:
        """Mock do Anthropic Claude mais realista"""
        return self._query_mock(question, context, "anthropic")
    
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
            
            # Pequena pausa para simular rate limit realista
            if not self.use_mock:
                time.sleep(0.2)
            else:
                time.sleep(0.05)  # Mock mais rápido
        
        return results
    
    def get_cost_estimate(self, num_queries: int, avg_tokens_per_query: int = 200) -> Dict[str, float]:
        """Estima custos para um volume de consultas"""
        
        provider_key = f"{self.api_provider}_mock" if self.use_mock else self.api_provider
        cost_per_1k = self.cost_per_1k_tokens.get(provider_key, 0.002)
        
        total_tokens = num_queries * avg_tokens_per_query
        total_cost = (total_tokens / 1000) * cost_per_1k
        
        return {
            "num_queries": num_queries,
            "avg_tokens_per_query": avg_tokens_per_query,
            "total_tokens": total_tokens,
            "cost_per_1k_tokens": cost_per_1k,
            "total_cost_usd": total_cost,
            "cost_per_query_usd": total_cost / num_queries if num_queries > 0 else 0,
            "using_mock": self.use_mock,
            "provider": f"{self.api_provider}_mock" if self.use_mock else self.api_provider
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retorna informações do sistema"""
        return {
            "api_provider": self.api_provider,
            "using_mock": self.use_mock,
            "client_configured": self.client is not None,
            "cost_per_1k_tokens": self.cost_per_1k_tokens.get(
                f"{self.api_provider}_mock" if self.use_mock else self.api_provider, 0.002
            ),
            "mock_knowledge_topics": list(self.mock_api.knowledge_base.keys()) if self.use_mock else []
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
    
    # Testa OpenAI (real ou mock)
    openai_baseline = APIBaseline("openai")
    
    # Log info do sistema
    system_info = openai_baseline.get_system_info()
    logger.info(f"Sistema API: {system_info}")
    
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
        
        source = "MOCK" if result.get('mock_used', False) else "API"
        logger.info(f"API {q['id']} [{source}]: {result.get('response_time', 0):.2f}s, "
                   f"custo: ${result.get('estimated_cost_usd', 0):.4f}")
    
    # Salva resultados
    results_file = Path("data/evaluation/api_baseline_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Log resumo
    total_cost = sum(r.get('estimated_cost_usd', 0) for r in results)
    avg_time = sum(r.get('response_time', 0) for r in results) / len(results)
    mock_count = sum(1 for r in results if r.get('mock_used', False))
    
    logger.info(f"✅ Baseline API testado:")
    logger.info(f"   • Resultados: {len(results)}")
    logger.info(f"   • Tempo médio: {avg_time:.2f}s")
    logger.info(f"   • Custo total: ${total_cost:.4f}")
    logger.info(f"   • Mocks usados: {mock_count}/{len(results)}")
    logger.info(f"   • Arquivo: {results_file}")
    
    return results


if __name__ == "__main__":
    # Teste standalone
    logging.basicConfig(level=logging.INFO)
    test_api_baseline()
  