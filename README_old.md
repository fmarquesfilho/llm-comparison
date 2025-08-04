# 🤖 LLM Architecture Comparison - MVP

Projeto simplificado para comparação rápida entre arquiteturas RAG local vs APIs externas, focado em viabilidade técnica e econômica para decisão executiva.

## 🎯 Objetivo do MVP

Validar em **2 semanas** (40h) qual arquitetura é mais adequada para o domínio de construção civil:
- **RAG Local**: Embeddings + FAISS + recuperação simples
- **API Externa**: OpenAI/Claude com contexto (com mock inteligente)

## 🚀 Quick Start

### 1. Setup Ambiente (Mac M1 otimizado)

```bash
# Clone o repositório
git clone <repo>
cd otoh-llm-comparison

# Crie ambiente conda
conda env create -f environment.yml
conda activate llm-comparison

# Atualiza ambiente
conda env update -f environment.yml

# Configure API keys (OPCIONAL - sistema funciona sem)
echo "OPENAI_API_KEY=sua_key_aqui" > .env
```

### 2. Teste Rápido do Sistema

```bash
# Valide ambiente rapidamente
python quick_test.py
```

### 3. Execução Completa do MVP

```bash
# Execute o pipeline completo
python run_mvp.py
```

Este comando irá:
1. ✅ Verificar ambiente
2. 📝 Criar dados sintéticos de exemplo (5 documentos)
3. 🤖 Configurar RAG com embeddings
4. 🧪 Testar RAG com 5 perguntas
5. 🌐 Testar API externa (real ou mock inteligente)
6. 💰 Calcular estimativas de custo
7. 📊 Gerar relatório de comparação

### 4. Visualizar Resultados

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
│   ├── api_baseline.py     # Cliente APIs externas + mock inteligente
│   └── evaluator.py        # Métricas e comparações
├── app/
│   └── dashboard.py        # Interface Streamlit
├── requirements.txt        # Dependências mínimas
├── run_mvp.py             # Script principal
└── quick_test.py          # Validação rápida do ambiente
```

## 🎛️ Configuração

### Modelos Otimizados para MVP

- **Embedding**: `all-MiniLM-L6-v2` (rápido, 22MB)
- **Geração**: Apenas retrieval simples (sem LLM local)
- **API Externa**: GPT-3.5-turbo (real) ou mock inteligente

### Sistema de Fallback Inteligente

O sistema **SEMPRE funciona**, mesmo sem API keys:

```bash
# Cenário 1: Com OpenAI API key válida
OPENAI_API_KEY=sk-... python run_mvp.py
# ✅ Usa API real da OpenAI

# Cenário 2: Sem API key ou quota esgotada
python run_mvp.py  
# ✅ Usa mock inteligente com respostas realistas

# Cenário 3: Problema de conexão
# ✅ Fallback automático para mock
```

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
| Mock API    | $0     | ~$0.002*    | ~$60*              |

*Custo estimativo para comparação - mock é gratuito

## 🔧 Desenvolvimento

### Executar Componentes Individuais

```bash
# Apenas RAG
python src/simple_rag.py

# Apenas API baseline (com fallback automático)
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
  "retrieved_docs": 3,
  "mock_used": false
}
```

## 📈 Dashboard

### Abas Disponíveis
1. **📋 Resumo Executivo**: Recomendação + próximos passos
2. **⚡ Performance**: Tempo de resposta + taxa de sucesso
3. **💰 Custos**: Comparação por cenário (real vs estimado)
4. **🎯 Qualidade**: Análise de relevância + conceitos
5. **🔍 Detalhes**: Resultados completos por pergunta

### Funcionalidades
- Comparação side-by-side RAG vs API
- Indicação clara quando mock é usado
- Filtros por categoria/qualidade
- Métricas em tempo real
- Gráficos interativos com Plotly

## 🚦 Troubleshooting

### Problemas Comuns

**1. Erro no PyTorch/MPS:**
```bash
# Force CPU se MPS der problema
export PYTORCH_ENABLE_MPS_FALLBACK=1
python run_mvp.py
```

**2. Sem quota OpenAI:**
```bash
# Sistema funciona normalmente com mock
unset OPENAI_API_KEY  # Remove key inválida
python run_mvp.py     # Usa mock automaticamente
```

**3. Dependências faltando:**
```bash
# Execute teste completo primeiro
python quick_test.py

# Se falhar, reinstale
pip install --upgrade -r requirements.txt
```

**4. FAISS não instala:**
```bash
# Com conda (melhor para M1 Mac)
conda install -c conda-forge faiss-cpu

# Ou com pip se conda falhar
pip install faiss-cpu --force-reinstall
```

**5. Conflitos de ambiente:**
```bash
# Lista ambientes conda
conda env list

# Remove ambiente se corrupto
conda env remove -n llm-comparison

# Recria ambiente limpo
conda create -n llm-comparison python=3.9 -y
```

### Logs Detalhados

```bash
# Debug completo
export LOG_LEVEL=DEBUG
python run_mvp.py
```

### Validação do Sistema

```bash
# Teste completo do ambiente
python quick_test.py

# Validação específica do RAG
python -c "from src.simple_rag import test_simple_rag; test_simple_rag()"

# Validação da API (com fallback)
python -c "from src.api_baseline import test_api_baseline; test_api_baseline()"
```

## 📋 Checklist MVP

### Funcionalidades ✅
- [x] Setup ambiente M1 compatível
- [x] RAG básico funcional com 5 documentos
- [x] Testes automáticos com 5 perguntas
- [x] API baseline com fallback inteligente
- [x] Sistema de mock realista
- [x] Métricas essenciais
- [x] Dashboard Streamlit completo
- [x] Análise de custos comparativa
- [x] Relatório executivo
- [x] Documentação completa

### Validação de Qualidade ✅
- [x] Funciona sem API keys
- [x] Fallback automático para mock
- [x] Respostas realistas no mock
- [x] Métricas consistentes
- [x] Interface intuitiva

## 🎯 Critério de Sucesso

**✅ Stakeholder consegue decidir** entre RAG local vs API externa baseado em:
- Dados objetivos de performance
- Análise clara de custos (real + estimado)
- Recomendação justificada com trade-offs
- Sistema que sempre funciona (com ou sem APIs)
- Interface visual para análise

## 🔄 Próximas Iterações

**Se MVP validar viabilidade:**
- Expandir para mais documentos reais
- Implementar fine-tuning com LoRA
- Métricas avançadas (ROUGE, BLEU)
- Pipeline automatizado CI/CD
- APIs reais (Anthropic Claude, etc.)

**Se MVP mostrar inviabilidade:**
- Pivot para apenas APIs externas
- Foco em integração/UX
- Análise detalhada de custos
- Avaliação de outras arquiteturas

## 🚀 Funcionalidades Exclusivas

### Mock Inteligente
- **Respostas contextualmente relevantes** baseadas em palavras-chave
- **Simulação realista de latência** de rede + processamento
- **Estimativas precisas de tokens** e custos
- **Fallback automático** quando API real falha

### Dashboard Avançado
- **Gráficos interativos** com Plotly
- **Comparação visual** entre arquiteturas
- **Métricas em tempo real** com caching
- **Indicação clara** de uso de mock vs API real

### Sistema de Avaliação
- **Métricas objetivas** automatizadas
- **Recomendações inteligentes** baseadas em dados
- **Análise de trade-offs** estruturada
- **Relatórios executivos** para tomada de decisão

## 📞 Suporte

Para problemas técnicos:
1. Execute `python quick_test.py` para diagnóstico
2. Verifique logs em `logs/`
3. Execute `Config.validate_setup()` para validação
4. Consulte troubleshooting acima

**Garantia**: O sistema sempre funciona, mesmo sem APIs externas!

---

**Tempo estimado total**: 5-15 minutos para setup + execução completa  
**Dados necessários**: Nenhum (usa dados sintéticos)  
**Dependências externas**: Nenhuma obrigatória (OpenAI opcional)  
**Compatibilidade**: macOS (M1/Intel), Linux, Windows
