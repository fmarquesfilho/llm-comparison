# 🤖 Sistema Multi-Cenário RAG - Construção Civil

Sistema completo para comparação entre três abordagens de NLP aplicadas ao domínio de construção civil, baseado nos artigos científicos mais recentes sobre RAG temporal e grafos dinâmicos.

## 📚 Base Científica

Este projeto implementa conceitos dos seguintes artigos:

1. **DyG-RAG: Dynamic Graph Retrieval-Augmented Generation** - Implementação de grafos dinâmicos com unidades de evento temporal
2. **StreamingRAG: Real-time Contextual Retrieval** - Processamento em tempo real de dados multimodais
3. **It's High Time: Temporal Information Retrieval** - Técnicas avançadas de recuperação temporal
4. **When to use Graphs in RAG** - Análise comparativa GraphRAG vs RAG tradicional

## 🎯 Cenários Implementados

### 📊 Cenário A: Vector-Only RAG
- **Descrição**: RAG tradicional usando apenas busca vetorial com embeddings temporais
- **Tecnologias**: SentenceTransformers + Fourier Time Encoding
- **Vantagens**: Rápido, simples, econômico
- **Limitações**: Não considera relacionamentos complexos entre eventos

### 🔗 Cenário B: Hybrid RAG (DyG-RAG + Kùzu)
- **Descrição**: Sistema híbrido inspirado no DyG-RAG, utilizando **Kùzu Graph Database**
- **Características**:
  - **Dynamic Event Units (DEUs)** com âncoras temporais precisas
  - Grafo de eventos em Kùzu com relacionamentos semânticos e temporais
  - **Time Chain-of-Thought (Time-CoT)** para raciocínio temporal estruturado
  - Recuperação multi-hop via consultas **openCypher**
  - **Fourier Time Encoding** para representação temporal contínua
- **Vantagens**: Raciocínio temporal avançado, alta precisão, arquitetura escalável
- **Limitações**: Maior overhead computacional na construção do grafo

### 🤖 Cenário C: LLM-Only
- **Descrição**: LLM puro sem recuperação de documentos
- **Vantagens**: Respostas fluidas, conhecimento generalista
- **Limitações**: Alucinações, dados desatualizados, sem acesso a contexto específico

## 🏗️ Arquitetura Implementada

```
src/
├── core/
│   ├── events.py           # Dynamic Event Units (DEUs)
│   ├── temporal_rag.py     # NetworkX-based DyG-RAG 
│   ├── vector_rag.py       # Traditional Vector RAG
│   └── router.py           # Query complexity routing
├── utils/
│   ├── time_utils.py       # Fourier Time Encoding
│   └── embeddings.py       # Semantic embeddings
├── evaluation/
│   ├── metrics.py          # Evaluation metrics
│   └── tests.py            # Unit tests
└── pipelines/
    ├── ingestion.py        # Data ingestion pipeline
    └── query.py            # Query processing pipeline

# Integração Kùzu (Nova Implementação)
kuzu_integration.py         # Kùzu Graph Database backend

# Script MVP Integrado
run_integrated_mvp.py       # Comparação automatizada

# Dashboard Streamlit
app/dashboard.py            # Visualização interativa
```

## 🚀 Quick Start

### 1. Configuração do Ambiente

```bash
# Crie o ambiente conda com todas as dependências
conda env create -f environment.yml
conda activate llm-comparison-multi

# Configure a API key da OpenAI (opcional - funciona com mocks)
echo "OPENAI_API_KEY=sua_key_aqui" > .env
```

### 2. Execução do MVP Integrado

```bash
# Executa comparação completa entre os 3 cenários
python run_integrated_mvp.py
```

Este script irá:
- ✅ Gerar 150 eventos sintéticos de construção civil
- ✅ Configurar os sistemas Vector RAG e DyG-RAG
- ✅ Executar 5 perguntas de teste em todos os cenários
- ✅ Gerar métricas comparativas de performance
- ✅ Salvar resultados em `data/evaluation/`

### 3. Dashboard Interativo

```bash
# Inicia o dashboard Streamlit
streamlit run app/dashboard.py
```

Acesse http://localhost:8501 para visualizar:
- 📊 Gráficos comparativos de performance
- 🔍 Análise detalhada por pergunta
- ⚡ Métricas de tempo de resposta e relevância

## 🔬 Implementações Técnicas Avançadas

### Dynamic Event Units (DEUs)

Baseado no paper DyG-RAG, cada evento sonoro é estruturado como:

```python
class DynamicEventUnit:
    event_id: str
    timestamp: datetime
    event_type: str          # martelo, serra, betoneira, etc.
    loudness: float          # Intensity em dB
    sensor_id: str
    description: Optional[str]
    metadata: Dict           # fase_obra, localização, equipamento
    
    def violates_noise_regulations(self) -> bool:
        # Lógica baseada em horários e intensidade
        
    def to_embedding_text(self) -> str:
        # Texto otimizado para busca semântica
```

### Fourier Time Encoding

Implementação da codificação temporal contínua do DyG-RAG:

```python
class FourierTimeEncoder:
    def encode(self, dt: datetime) -> np.ndarray:
        # Gera representação periódica suave do tempo
        # Captura padrões diários, semanais, sazonais
```

### Time Chain-of-Thought (Time-CoT)

Raciocínio temporal estruturado em 6 etapas:

1. **Identificar escopo temporal** da pergunta
2. **Filtrar eventos** no escopo identificado  
3. **Analisar ordem cronológica** dos eventos
4. **Inferir persistência de estados** temporais
5. **Verificar consistência** com regulamentações
6. **Gerar sugestão** baseada na cadeia de raciocínio

### Integração Kùzu Graph Database

Sistema de grafo temporal escalável usando openCypher:

```cypher
# Exemplo: Busca por padrões sequenciais
MATCH (e1:Event)-[r:TemporalRelation]->(e2:Event)
WHERE r.time_diff_seconds <= 1800
RETURN e1.event_type, e2.event_type, COUNT(*) as frequency
ORDER BY frequency DESC
```

## 📊 Resultados Esperados

Com base nos papers de referência, esperamos:

### Performance (Tempo de Resposta)
1. **LLM-Only**: ~50ms (mais rápido)
2. **Vector RAG**: ~200ms (intermediário)  
3. **DyG-RAG**: ~800ms (mais lento, mais preciso)

### Relevância (Score de Qualidade)
1. **DyG-RAG**: ~0.85 (melhor para análises complexas)
2. **Vector RAG**: ~0.72 (bom para buscas simples)
3. **LLM-Only**: ~0.60 (limitado sem contexto)

### Trade-offs Identificados
- **Consultas Simples**: Vector RAG é suficiente e mais eficiente
- **Análises Complexas**: DyG-RAG oferece vantagem significativa
- **Respostas Gerais**: LLM-Only para conhecimento base

## 🔧 Configuração Avançada

### Parâmetros do Sistema

```python
# config/settings.py
TIME_WINDOW = 300           # Janela temporal (segundos)
TIME_EMBEDDING_DIM = 64     # Dimensionalidade temporal
SIMILARITY_THRESHOLD = 0.7  # Threshold semântico
```

### Personalização de Dados

Para usar seus próprios dados de eventos sonoros:

```python
# Formato CSV esperado
# timestamp,event_type,loudness,sensor_id,description,phase,location
```

## 🧪 Testes e Avaliação

```bash
# Executa suite de testes
python -m pytest src/evaluation/tests.py -v

# Análise de coverage
pytest --cov=src src/evaluation/tests.py
```

## 📈 Monitoramento

O sistema inclui métricas detalhadas:
- Tempo de resposta por cenário
- Qualidade da recuperação (Precision/Recall)
- Estatísticas do grafo temporal
- Detecção de violações de ruído

## 🚀 Deploy

O sistema está preparado para:
- **AWS Neptune** (substituir Kùzu para produção)
- **Docker containers** 
- **API REST** via FastAPI
- **Monitoring** com Prometheus

## 📝 Próximos Passos

1. **Integração com dados reais** de sensores IoT
2. **Fine-tuning** de embeddings para domínio específico
3. **Otimização** de consultas Kùzu para grandes volumes
4. **Dashboard** de monitoramento em tempo real
5. **API** para integração com sistemas externos

## 🤝 Contribuição

Este projeto implementa técnicas de ponta em RAG temporal. Contribuições são bem-vindas especialmente em:

- Otimização de performance do grafo dinâmico
- Novos algoritmos de Time-CoT
- Integração com mais tipos de sensores
- Métricas de avaliação mais sofisticadas

## 📄 Licença

MIT License - Veja [LICENSE](LICENSE) para detalhes.

---

*Baseado nos papers: DyG-RAG (Sun et al., 2025), StreamingRAG (Sankaradas et al., 2024), Temporal IR Survey (Piryani et al., 2025), GraphRAG Analysis (Xiang et al., 2025)*