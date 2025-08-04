# 🤖 Sistema Multi-Cenário RAG - Construção Civil

Sistema completo para comparação entre três abordagens de processamento de linguagem natural aplicadas ao domínio de construção civil, incorporando as recomendações dos artigos científicos sobre DyG-RAG e raciocínio temporal.

## 🎯 Cenários Implementados

### 📊 Cenário A: Vector-Only RAG
- **Descrição**: RAG tradicional usando apenas busca vetorial com FAISS
- **Características**: 
  - Documentos divididos em chunks com overlap
  - Busca por similaridade semântica
  - Respostas baseadas nos trechos mais relevantes
- **Vantagens**: Rápido, simples, econômico
- **Limitações**: Não considera relacionamentos temporais

### 🔗 Cenário B: Hybrid RAG (Vector + Graph)
- **Descrição**: Sistema híbrido inspirado no DyG-RAG com grafos dinâmicos de eventos
- **Características**:
  - Dynamic Event Units (DEUs) com âncoras temporais
  - Grafo de eventos com relacionamentos semânticos e temporais
  - Time Chain-of-Thought para raciocínio temporal
  - Recuperação multi-hop através do grafo
- **Vantagens**: Raciocínio temporal avançado, alta precisão para queries complexas
- **Limitações**: Maior complexidade computacional

### 🤖 Cenário C: LLM-Only
- **Descrição**: LLM puro sem recuperação de documentos
- **Características**:
  - Base de conhecimento integrada sobre construção civil
  - Respostas baseadas no conhecimento pré-treinado
  - Mock inteligente quando APIs não disponíveis
- **Vantagens**: Respostas fluidas, não depende de documentos específicos
- **Limitações**: Pode gerar informações desatualizadas ou imprecisas

## 🚀 Quick Start

### 1. Configuração do Ambiente

```bash
# Clone o repositório
git clone <repo>
cd otoh-llm-comparison

# Crie ambiente conda com dependências para todos os cenários
conda env create -f environment.yml
conda activate llm-comparison-multi

# Configure API keys (opcional - sistema funciona com mocks)
echo "OPENAI_API_KEY=sua_key_aqui" > .env
```

### 2. Execução Completa

```bash
# Execute o sistema integrado multi-cenário
python run_integrated_mvp.py
```

Este comando executará:
1. ✅ Verificação de ambiente
2. 📝 Criação de dados sintéticos enriquecidos com contexto temporal
3. 🔧 Construção de todos os três cenários
4. 🧪 Teste com perguntas temporais específicas
5. 📊 Comparação de performance
6. 📋 Geração de relatório executivo

### 3. Dashboard Interativo

```bash
# Lance o dashboard para análise visual
streamlit run app/multi_scenario_dashboard.py
```

## 📁 Estrutura do Projeto

```
otoh-llm-comparison/
├── data/
│   ├── raw/                           # Documentos JSON com contexto temporal
│   └── evaluation/                    # Resultados e relatórios
│       ├── multi_scenario_comparison.json    # Resultados detalhados
│       ├── comparative_report.json          # Relatório executivo
│       └── temporal_test_questions.json     # Perguntas de teste
├── src/
│   ├── multi_scenario_system.py      # Sistema principal multi-cenário
│   ├── config.py                     # Configurações
│   └── [outros arquivos originais]
├── app/
│   └── multi_scenario_dashboard.py   # Dashboard Streamlit
├── run_integrated_mvp.py             # Script principal integrado
└── environment.yml                   # Dependências atualizadas
```

## 🔧 Características Técnicas

### Extração Temporal Avançada
O sistema implementa extratores específicos para construção civil:

- **Horários**: "das 7h às 22h", "período noturno"
- **Durações**: "por 28 dias", "a cada 50m³"
- **Limites**: "superior a 3000m²", "não pode exceder 70 dB"
- **Sequências**: "antes de", "após", "simultaneamente"
- **Regulamentações**: "NBR 10151:2019", "aprovado em 2020"

### Dynamic Event Units (DEUs)
Estrutura inspirada no artigo DyG-RAG:

```python
@dataclass
class DynamicEventUnit:
    id: str
    content: str
    entities: List[str]
    temporal_anchor: TemporalAnchor
    event_type: str  # "regulation", "procedure", "measurement"
    source_document: str
    embedding: Optional[np.ndarray]
```

### Grafo de Eventos
- Nós: Dynamic Event Units
- Arestas: Relacionamentos baseados em:
  - Similaridade de entidades (40%)
  - Proximidade temporal (30%)
  - Similaridade semântica (30%)

## 📊 Métricas de Avaliação

### Quantitativas
- **Tempo de resposta** (segundos)
- **Custo por query** (USD)
- **Taxa de relevância** (0-1)
- **Taxa de sucesso** (%)
- **Documentos/eventos recuperados**

### Qualitativas (Dashboard)
- **Cobertura de conceitos esperados**
- **Precisão das informações temporais**
- **Coerência das respostas**
- **Utilidade prática**

## 🎛️ Configurações Avançadas

### Personalização de Parâmetros

```python
# Cenário A: Vector RAG
chunk_size = 512        # Tamanho dos chunks
overlap = 50           # Overlap entre chunks
top_k = 5              # Documentos recuperados

# Cenário B: Hybrid RAG  
max_hops = 2           # Máximo de saltos no grafo
similarity_threshold = 0.25  # Threshold para conexões
use_graph_expansion = True
