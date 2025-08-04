# 🤖 Sistema Multi-Cenário RAG - Construção Civil

Sistema completo para comparação entre três abordagens de NLP aplicadas ao domínio de construção civil.

## 🎯 Cenários Implementados

### 📊 Cenário A: Vector-Only RAG
- **Descrição**: RAG tradicional usando apenas busca vetorial com FAISS.
- **Vantagens**: Rápido, simples, econômico.
- **Limitações**: Não considera relacionamentos complexos ou temporais entre os dados.

### 🔗 Cenário B: Hybrid RAG (Vector + Kùzu Graph)
- **Descrição**: Sistema híbrido inspirado no DyG-RAG, utilizando o banco de dados de grafo **Kùzu** para modelar eventos dinâmicos.
- **Características**:
  - Dynamic Event Units (DEUs) com âncoras temporais.
  - Grafo de eventos em Kùzu com relacionamentos semânticos e temporais.
  - Time Chain-of-Thought para raciocínio temporal.
  - Recuperação multi-hop via consultas **openCypher**.
- **Vantagens**: Raciocínio temporal avançado, alta precisão, e arquitetura escalável (pronta para AWS Neptune).
- **Limitações**: Maior complexidade computacional na construção do grafo.

### 🤖 Cenário C: LLM-Only
- **Descrição**: LLM puro sem recuperação de documentos.
- **Vantagens**: Respostas fluidas, não depende de documentos específicos.
- **Limitações**: Pode gerar informações desatualizadas ou imprecisas (alucinações).

## 🚀 Quick Start

### 1. Configuração do Ambiente

```bash
# Clone o repositório
git clone <repo>
cd otoh-llm-comparison

# Crie o ambiente conda com todas as dependências
conda env create -f environment.yml
conda activate llm-comparison-multi

# Configure a API key da OpenAI (opcional - o sistema funciona com mocks)
echo "OPENAI_API_KEY=sua_key_aqui" > .env
