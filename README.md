# 🤖 LLM Architecture Comparison - MVP

Projeto simplificado para comparação rápida entre arquiteturas RAG local vs APIs externas, focado em viabilidade técnica e econômica para decisão executiva.

## 🎯 Objetivo do MVP

Validar em **2 semanas** (40h) qual arquitetura é mais adequada para o domínio de construção civil:
- **RAG Local**: Embeddings + FAISS + recuperação simples
- **API Externa**: OpenAI/Claude com contexto

## 🚀 Quick Start

### 1. Setup Ambiente (Mac M1 otimizado)

```bash
# Clone o repositório
git clone <repo>
cd otoh-llm-comparison

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Instale dependências essenciais
pip install -r requirements.txt

# Configure API keys (opcional)
echo "OPENAI_API_KEY=sua_key_aqui" > .env
```

### 2. Execução Completa do MVP

```bash
# Execute o pipeline completo
python run_mvp.py
```

Este comando irá:
1. ✅ Verificar ambiente
2. 📝 Criar dados sintéticos de exemplo
3. 🤖 Configurar RAG com embeddings
4. 🧪 Testar RAG com 5 perguntas
5. 🌐 Testar API externa (se configurada)
6. 💰 Calcular estimativas de custo
7. 📊 Gerar relatório de comparação

### 3. Visualizar Resultados

```bash
# Lançar dashboard interativo
streamlit run app/dashboard.py
```

Acesse: http://localhost:8501

## 📁 Estrutura do Projeto

```
otoh-llm-comparison/
├── data/
│   ├── raw/                 # Documentos JSON (criados automaticamente)
│   ├── embeddings/          # Índice FAISS + metadados
│   └── evaluation/          # Resultados e relatórios
├── src/
│   ├── config.py           # Configurações centralizadas
│   ├── simple_rag.py       # RAG simplificado (só embeddings)
│   ├── api_baseline.py     # Cliente APIs externas
│   └── evaluator.py        # Métricas e comparações
├── app/
│   └── dashboard.py        # Interface Streamlit
├── requirements.txt        # Dependências mínimas
└── run_mvp.py             # Script principal
```

## 🎛️ Configuração

### Modelos Otimizados para MVP

- **Embedding**: `all-MiniLM-L6-v2` (rápido, 22MB)
- **Geração**: Apenas retrieval simples (sem LLM local)
- **API Externa**: GPT-3.5-turbo (custo-benefício)

### Dados de Exemplo

O sistema cria automaticamente 5 documentos sobre:
- Normas de ruído (NBR 10151)
- EPIs obrigatórios
- Controle ambiental
- Obras noturnas
- Controle de qualidade

### Perguntas de Teste

5 perguntas cobrindo diferentes categorias:
1. Limites de ruído permitidos
2. Equipamentos de proteção obrigatórios
3. Controle de qualidade do concreto
4. Requisitos para obras noturnas
5. Quando fazer EIA/RIMA

## 📊 Métricas Avaliadas

### Quantitativas
- **Tempo de resposta** (segundos)
- **Custo por query** (USD)
- **Taxa de documentos relevantes** (%)
- **Cobertura de conceitos esperados** (%)

### Qualitativas (Manual)
- **Precisão da resposta** (1-5)
- **Completude da informação** (1-5)
- **Usabilidade geral** (1-5)

### Critérios de Decisão
- ✅ **< $0.10 por query**: Viável economicamente
- ✅ **< 3 segundos**: Experiência aceitável  
- ✅ **> 70% relevância**: Qualidade mínima
- ✅ **> 3.0/5.0 qualidade**: Aprovação usuários

## 💰 Análise de Custos

### Cenários Testados
- **Piloto**: 100 queries/dia
- **Produção Pequena**: 1K queries/dia
- **Produção Média**: 5K queries/dia

### Comparação Estimada
| Arquitetura | Setup | Custo/Query | Custo/Mês (1K/dia) |
|-------------|--------|-------------|---------------------|
| RAG Local   | $0     | ~$0.000     | ~$0                |
| OpenAI API  | $0     | ~$0.002     | ~$60               |

## 🔧 Desenvolvimento

### Executar Componentes Individuais

```bash
# Apenas RAG
python src/simple_rag.py

# Apenas API baseline
python src/api_baseline.py

# Apenas avaliação
python src/evaluator.py

# Validar configuração
python -c "from src.config import Config; Config.validate_setup()"
```

### Estrutura de Dados

**Documento JSON:**
```json
{
  "id": "doc_001",
  "title": "Normas de Ruído",
  "content": "O monitoramento de ruído...",
  "category": "regulamentacao",
  "keywords": ["ruído", "NBR 10151"]
}
```

**Resultado de Query:**
```json
{
  "question": "Quais os limites de ruído?",
  "answer": "Baseado no documento...",
  "response_time": 1.23,
  "relevance_score": 0.85,
  "retrieved_docs": 3
}
```

## 📈 Dashboard

### Abas Disponíveis
1. **📋 Resumo Executivo**: Recomendação + próximos passos
2. **⚡ Performance**: Tempo de resposta + taxa de sucesso
3. **💰 Custos**: Comparação por cenário
4. **🎯 Qualidade**: Análise de relevância + conceitos
5. **🔍 Detalhes**: Resultados completos por pergunta

### Funcionalidades
- Comparação side-by-side
- Filtros por categoria/qualidade
- Métricas em tempo real
- Export de relatórios

## 🚦 Troubleshooting

### Problemas Comuns

**1. Erro no PyTorch/MPS:**
```bash
# Force CPU se MPS der problema
export PYTORCH_ENABLE_MPS_FALLBACK=1
python run_mvp.py
```

**2. OpenAI API não funciona:**
```bash
# Teste sem API (só RAG)
export OPENAI_API_KEY=""
python run_mvp.py
```

**3. Dependências faltando:**
```bash
# Reinstale requirements
pip install --upgrade -r requirements.txt
```

**4. FAISS não instala:**
```bash
# Use versão CPU
pip install faiss-cpu --force-reinstall
```

### Logs Detalhados

```bash
# Debug completo
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python run_mvp.py
```

## 📋 Checklist MVP

### Semana 1 ✅
- [x] Setup ambiente M1 compatível
- [x] RAG básico funcional com 5 documentos
- [x] Testes automáticos com 5 perguntas
- [x] API baseline (OpenAI + mock)
- [x] Métricas essenciais

### Semana 2 🚧
- [x] Dashboard Streamlit
- [x] Comparação automática
- [x] Análise de custos
- [x] Relatório executivo
- [ ] Testes manuais de qualidade

## 🎯 Critério de Sucesso

**✅ Stakeholder consegue decidir** entre RAG local vs API externa baseado em:
- Dados objetivos de performance
- Análise clara de custos
- Recomendação justificada
- Próximos passos definidos

## 🔄 Próximas Iterações

**Se MVP validar viabilidade:**
- Expandir para mais documentos reais
- Implementar fine-tuning com LoRA
- Métricas avançadas (ROUGE, BLEU)
- Pipeline automatizado CI/CD

**Se MVP mostrar inviabilidade:**
- Pivot para apenas APIs externas
- Foco em integração/UX
- Análise detalhada de custos
- Avaliação de outras arquiteturas

## 📞 Suporte

Para problemas técnicos:
1. Verifique logs em `logs/`
2. Execute `Config.validate_setup()`
3. Teste componentes individualmente
4. Consulte troubleshooting acima

---

**Tempo estimado total**: 15-30 minutos para setup + execução completa
**Dados necessários**: Nenhum (usa dados sintéticos)
**Dependências externas**: Apenas OpenAI API (opcional)
