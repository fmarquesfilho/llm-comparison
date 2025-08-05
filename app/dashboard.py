# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="RAG Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_results():
    """Carrega os resultados da comparação do arquivo JSON."""
    results_file = Path("data/evaluation/multi_scenario_comparison.json")
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        st.error(f"Erro ao carregar resultados: {str(e)}")
        return None

@st.cache_data
def load_summary():
    """Carrega o resumo de performance."""
    summary_file = Path("data/evaluation/performance_summary.json")
    if not summary_file.exists():
        return None
    
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        return None

def create_performance_metrics(summary_data):
    """Cria cards de métricas de performance."""
    if not summary_data:
        st.warning("Dados de resumo não disponíveis.")
        return
    
    scenarios = summary_data.get('scenarios', {})
    
    col1, col2, col3 = st.columns(3)
    
    scenario_names = {
        'scenario_a': 'Vector RAG',
        'scenario_b': 'Hybrid RAG (Kùzu)',
        'scenario_c': 'LLM-Only'
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (scenario_key, scenario_data) in enumerate(scenarios.items()):
        with [col1, col2, col3][i]:
            metrics = scenario_data.get('metrics', {})
            name = scenario_names.get(scenario_key, scenario_key)
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {colors[i]}20, {colors[i]}10);
                padding: 1.5rem;
                border-radius: 10px;
                border-left: 4px solid {colors[i]};
                margin-bottom: 1rem;
            ">
                <h3 style="color: {colors[i]}; margin: 0;">{name}</h3>
                <p style="font-size: 2em; font-weight: bold; margin: 0.5rem 0;">
                    {metrics.get('avg_response_time', 0)*1000:.0f}ms
                </p>
                <p style="margin: 0;">Tempo médio de resposta</p>
                <hr style="margin: 1rem 0;">
                <p style="margin: 0;"><strong>Relevância:</strong> {metrics.get('avg_relevance', 0):.2f}</p>
                <p style="margin: 0;"><strong>Itens recuperados:</strong> {metrics.get('avg_retrieved_items', 0):.1f}</p>
            </div>
            """, unsafe_allow_html=True)

def create_comparison_charts(results_data):
    """Cria gráficos comparativos aprimorados."""
    if not results_data:
        st.warning("Não há dados para criar gráficos.")
        return
    
    # Prepara dados para visualização
    chart_data = []
    for comp in results_data:
        q_id = comp['question_metadata']['id']
        q_text = comp['question_metadata']['question']
        q_type = comp['question_metadata'].get('type', 'unknown')
        
        for scenario, result in comp['results'].items():
            scenario_names = {
                'scenario_a': 'Vector RAG',
                'scenario_b': 'Hybrid RAG (Kùzu)', 
                'scenario_c': 'LLM-Only'
            }
            
            chart_data.append({
                'pergunta_id': q_id,
                'pergunta_texto': q_text[:50] + "..." if len(q_text) > 50 else q_text,
                'pergunta_tipo': q_type,
                'cenario': scenario_names.get(scenario, scenario),
                'tempo_resposta_ms': result.get('response_time', 0) * 1000,
                'relevancia': result.get('relevance_score', 0),
                'itens_recuperados': result.get('retrieved_chunks', result.get('retrieved_events', 0))
            })
    
    if not chart_data:
        st.warning("Dados insuficientes para gráficos.")
        return
    
    df = pd.DataFrame(chart_data)
    
    # Gráfico 1: Tempo de resposta
    st.subheader("⚡ Comparação de Tempo de Resposta")
    fig_time = px.bar(
        df, 
        x='pergunta_id', 
        y='tempo_resposta_ms', 
        color='cenario',
        barmode='group',
        labels={
            'tempo_resposta_ms': 'Tempo de Resposta (ms)',
            'pergunta_id': 'ID da Pergunta',
            'cenario': 'Cenário'
        },
        color_discrete_map={
            'Vector RAG': '#FF6B6B',
            'Hybrid RAG (Kùzu)': '#4ECDC4',
            'LLM-Only': '#45B7D1'
        },
        hover_data=['pergunta_texto', 'pergunta_tipo']
    )
    fig_time.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Gráfico 2: Score de relevância
    st.subheader("🎯 Comparação de Relevância")
    fig_rel = px.bar(
        df,
        x='pergunta_id',
        y='relevancia',
        color='cenario',
        barmode='group',
        labels={
            'relevancia': 'Score de Relevância',
            'pergunta_id': 'ID da Pergunta',
            'cenario': 'Cenário'
        },
        color_discrete_map={
            'Vector RAG': '#FF6B6B',
            'Hybrid RAG (Kùzu)': '#4ECDC4',
            'LLM-Only': '#45B7D1'
        },
        hover_data=['pergunta_texto', 'pergunta_tipo']
    )
    fig_rel.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig_rel, use_container_width=True)
    
    # Gráfico 3: Scatter plot Tempo vs Relevância
    st.subheader("📊 Análise Tempo vs Relevância")
    fig_scatter = px.scatter(
        df,
        x='tempo_resposta_ms',
        y='relevancia',
        color='cenario',
        size='itens_recuperados',
        hover_data=['pergunta_id', 'pergunta_texto'],
        labels={
            'tempo_resposta_ms': 'Tempo de Resposta (ms)',
            'relevancia': 'Score de Relevância',
            'cenario': 'Cenário'
        },
        color_discrete_map={
            'Vector RAG': '#FF6B6B',
            'Hybrid RAG (Kùzu)': '#4ECDC4',
            'LLM-Only': '#45B7D1'
        }
    )
    
    # Adiciona linha de trade-off ideal
    fig_scatter.add_shape(
        type="line",
        x0=df['tempo_resposta_ms'].min(),
        y0=0.8,
        x1=df['tempo_resposta_ms'].max(),
        y1=0.8,
        line=dict(color="gray", width=2, dash="dash"),
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

def display_detailed_analysis(results_data):
    """Exibe análise detalhada com filtros."""
    st.header("🔍 Análise Detalhada")
    
    if not results_data:
        st.warning("Dados não disponíveis para análise detalhada.")
        return
    
    # Filtros na sidebar
    st.sidebar.header("Filtros")
    
    # Filtro por pergunta
    questions = [(comp['question_metadata']['id'], comp['question_metadata']['question']) 
                for comp in results_data]
    selected_q = st.sidebar.selectbox(
        "Selecionar Pergunta:",
        options=range(len(questions)),
        format_func=lambda x: f"{questions[x][0]}: {questions[x][1][:40]}..."
    )
    
    # Filtro por cenário
    scenarios = ['Todos', 'Vector RAG', 'Hybrid RAG (Kùzu)', 'LLM-Only']
    selected_scenario = st.sidebar.selectbox("Cenário:", scenarios)
    
    # Exibe análise da pergunta selecionada
    comp = results_data[selected_q]
    q_meta = comp['question_metadata']
    
    st.subheader(f"Pergunta {q_meta['id']}: {q_meta['question']}")
    
    # Informações da pergunta
    col1, col2 = st.columns([1, 3])
    with col1:
        st.info(f"**Tipo:** {q_meta.get('type', 'N/A')}")
        st.info(f"**Complexidade:** {q_meta.get('expected_complexity', 'N/A')}")
    
    # Resultados por cenário
    scenario_mapping = {
        'scenario_a': 'Vector RAG',
        'scenario_b': 'Hybrid RAG (Kùzu)',
        'scenario_c': 'LLM-Only'
    }
    
    with col2:
        for scenario_key, scenario_name in scenario_mapping.items():
            if selected_scenario != 'Todos' and selected_scenario != scenario_name:
                continue
                
            result = comp['results'].get(scenario_key, {})
            if not result:
                continue
            
            with st.expander(f"**{scenario_name}** - {result.get('response_time', 0)*1000:.0f}ms"):
                
                # Métricas
                col_metrics = st.columns(4)
                with col_metrics[0]:
                    st.metric("Tempo", f"{result.get('response_time', 0)*1000:.0f}ms")
                with col_metrics[1]:
                    st.metric("Relevância", f"{result.get('relevance_score', 0):.2f}")
                with col_metrics[2]:
                    items_key = 'retrieved_events' if 'retrieved_events' in result else 'retrieved_chunks'
                    st.metric("Itens", result.get(items_key, 0))
                with col_metrics[3]:
                    st.metric("Método", result.get('method', 'N/A'))
                
                # Resposta
                st.text_area(
                    "Resposta Gerada:", 
                    result.get('answer', 'Resposta não disponível'),
                    height=200,
                    key=f"resp_{q_meta['id']}_{scenario_key}"
                )

def display_insights_and_recommendations(summary_data):
    """Exibe insights automáticos e recomendações."""
    st.header("💡 Insights e Recomendações")
    
    if not summary_data:
        st.warning("Dados de resumo não disponíveis para insights.")
        return
    
    analysis = summary_data.get('analysis', {})
    scenarios = summary_data.get('scenarios', {})
    
    # Cards de insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏆 Ranking de Performance
        """)
        
        # Ranking de velocidade
        st.markdown("**⚡ Velocidade (mais rápido primeiro):**")
        speed_ranking = analysis.get('speed_ranking', [])
        scenario_names = {
            'scenario_a': 'Vector RAG',
            'scenario_b': 'Hybrid RAG (Kùzu)',
            'scenario_c': 'LLM-Only'
        }
        
        for i, scenario in enumerate(speed_ranking, 1):
            name = scenario_names.get(scenario, scenario)
            metrics = scenarios.get(scenario, {}).get('metrics', {})
            time_ms = metrics.get('avg_response_time', 0) * 1000
            st.write(f"{i}. **{name}**: {time_ms:.0f}ms")
        
        # Ranking de relevância
        st.markdown("**🎯 Relevância (melhor primeiro):**")
        relevance_ranking = analysis.get('relevance_ranking', [])
        
        for i, scenario in enumerate(relevance_ranking, 1):
            name = scenario_names.get(scenario, scenario)
            metrics = scenarios.get(scenario, {}).get('metrics', {})
            relevance = metrics.get('avg_relevance', 0)
            st.write(f"{i}. **{name}**: {relevance:.2f}")
    
    with col2:
        st.markdown("""
        ### 🎯 Recomendações de Uso
        """)
        
        fastest = analysis.get('fastest_method', 'N/A')
        most_relevant = analysis.get('most_relevant_method', 'N/A')
        
        if fastest == most_relevant:
            st.success(f"✅ **{fastest}** oferece o melhor equilíbrio entre velocidade e relevância")
        else:
            st.info(f"⚡ **Consultas rápidas:** Use {fastest}")
            st.info(f"🎯 **Análises complexas:** Use {most_relevant}")
        
        # Análise de trade-offs
        st.markdown("**📈 Trade-offs Identificados:**")
        
        # Calcula médias para comparação
        vector_metrics = scenarios.get('scenario_a', {}).get('metrics', {})
        hybrid_metrics = scenarios.get('scenario_b', {}).get('metrics', {})
        llm_metrics = scenarios.get('scenario_c', {}).get('metrics', {})
        
        if vector_metrics and hybrid_metrics:
            speed_diff = (hybrid_metrics.get('avg_response_time', 0) - vector_metrics.get('avg_response_time', 0)) * 1000
            relevance_diff = hybrid_metrics.get('avg_relevance', 0) - vector_metrics.get('avg_relevance', 0)
            
            if speed_diff > 100:  # Diferença significativa
                st.write(f"• Hybrid RAG é {speed_diff:.0f}ms mais lento que Vector RAG")
            if relevance_diff > 0.1:  # Diferença significativa
                st.write(f"• Hybrid RAG oferece {relevance_diff:.2f} pontos a mais em relevância")

def create_system_overview():
    """Cria visão geral do sistema e arquitetura."""
    st.header("🏗️ Visão Geral do Sistema")
    
    # Arquitetura em colunas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Cenário A: Vector RAG
        **Tecnologia:** Embeddings + Busca Vetorial
        
        **Características:**
        - ⚡ Rápido e eficiente
        - 🔍 Busca por similaridade semântica
        - ⏱️ Encoding temporal com Fourier
        - 💰 Baixo custo computacional
        
        **Melhor para:**
        - Consultas factuais simples
        - Busca por eventos específicos
        - Aplicações que priorizam velocidade
        """)
    
    with col2:
        st.markdown("""
        ### 🔗 Cenário B: Hybrid RAG (Kùzu)
        **Tecnologia:** Kùzu Graph DB + DyG-RAG
        
        **Características:**
        - 🧠 Raciocínio temporal avançado
        - 📈 Dynamic Event Units (DEUs)
        - 🔗 Relacionamentos multi-hop
        - ⚡ Time Chain-of-Thought
        
        **Melhor para:**
        - Análises causais complexas
        - Detecção de padrões temporais
        - Perguntas que exigem contexto
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Cenário C: LLM-Only
        **Tecnologia:** LLM Puro (sem RAG)
        
        **Características:**
        - 💬 Respostas fluidas e naturais
        - 🌍 Conhecimento generalista
        - ⚡ Resposta imediata
        - 🔄 Sem necessidade de indexação
        
        **Melhor para:**
        - Perguntas conceituais gerais
        - Explicações e definições
        - Casos sem dados específicos
        """)

def main():
    """Função principal do dashboard."""
    # Header principal
    st.markdown("""
    # 📊 Dashboard de Comparação Multi-Cenário RAG
    ### Análise Comparativa: Vector RAG vs Hybrid RAG (Kùzu) vs LLM-Only
    
    ---
    """)
    
    # Carrega dados
    results = load_results()
    summary = load_summary()
    
    if not results:
        st.error("""
        ❌ **Dados não encontrados!**
        
        Para visualizar o dashboard, execute primeiro:
        ```bash
        python run_integrated_mvp.py
        ```
        
        Isso irá gerar os arquivos necessários em `data/evaluation/`.
        """)
        
        # Mostra arquitetura mesmo sem dados
        create_system_overview()
        return
    
    # Sidebar com informações do dataset
    st.sidebar.markdown("""
    ## 📈 Informações do Dataset
    """)
    
    total_questions = len(results)
    st.sidebar.metric("Total de Perguntas", total_questions)
    
    if summary:
        scenarios_count = len(summary.get('scenarios', {}))
        st.sidebar.metric("Cenários Testados", scenarios_count)
    
    # Menu de navegação
    st.sidebar.markdown("""
    ## 🧭 Navegação
    """)
    
    page = st.sidebar.radio(
        "Selecione a visualização:",
        ["📊 Visão Geral", "📈 Comparação Detalhada", "🔍 Análise por Pergunta", "🏗️ Arquitetura do Sistema"],
        index=0
    )
    
    # Renderiza página selecionada
    if page == "📊 Visão Geral":
        if summary:
            create_performance_metrics(summary)
            
            # Gráfico resumo rápido
            st.subheader("📊 Resumo de Performance")
            scenarios = summary.get('scenarios', {})
            
            if scenarios:
                # Dados para gráfico de radar
                categories = ['Velocidade', 'Relevância', 'Eficiência']
                
                scenario_data = []
                for scenario_key, scenario_info in scenarios.items():
                    metrics = scenario_info.get('metrics', {})
                    name = scenario_info.get('name', scenario_key)
                    
                    # Normaliza métricas para 0-1
                    speed_score = 1 - min(metrics.get('avg_response_time', 1), 1)  # Inverte porque menor é melhor
                    relevance_score = metrics.get('avg_relevance', 0)
                    efficiency_score = (speed_score + relevance_score) / 2
                    
                    scenario_data.append({
                        'name': name,
                        'values': [speed_score, relevance_score, efficiency_score]
                    })
                
                # Cria gráfico de radar
                fig = go.Figure()
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for i, data in enumerate(scenario_data):
                    fig.add_trace(go.Scatterpolar(
                        r=data['values'],
                        theta=categories,
                        fill='toself',
                        name=data['name'],
                        line_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Comparação Multi-dimensional",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
    elif page == "📈 Comparação Detalhada":
        create_comparison_charts(results)
        
    elif page == "🔍 Análise por Pergunta":
        display_detailed_analysis(results)
        
    elif page == "🏗️ Arquitetura do Sistema":
        create_system_overview()
        
        # Estatísticas técnicas se disponíveis
        if summary:
            st.subheader("📊 Estatísticas Técnicas")
            
            scenarios = summary.get('scenarios', {})
            tech_data = []
            
            for scenario_key, scenario_info in scenarios.items():
                metrics = scenario_info.get('metrics', {})
                tech_data.append({
                    'Cenário': scenario_info.get('name', scenario_key),
                    'Tempo Médio (ms)': f"{metrics.get('avg_response_time', 0)*1000:.0f}",
                    'Desvio Padrão (ms)': f"{metrics.get('std_response_time', 0)*1000:.0f}",
                    'Relevância Média': f"{metrics.get('avg_relevance', 0):.3f}",
                    'Itens Recuperados': f"{metrics.get('avg_retrieved_items', 0):.1f}",
                    'Total Processado': metrics.get('total_processed', 0)
                })
            
            if tech_data:
                tech_df = pd.DataFrame(tech_data)
                st.dataframe(tech_df, use_container_width=True)
    
    # Footer com insights
    if summary:
        display_insights_and_recommendations(summary)
    
    # Footer informativo
    st.markdown("""
    ---
    
    **💡 Sobre este dashboard:**
    
    Este dashboard apresenta uma comparação abrangente entre três abordagens de RAG aplicadas ao domínio de construção civil, 
    baseado nos papers científicos mais recentes sobre RAG temporal e grafos dinâmicos.
    
    **📚 Implementações baseadas em:**
    - DyG-RAG: Dynamic Graph Retrieval-Augmented Generation (Sun et al., 2025)
    - StreamingRAG: Real-time Contextual Retrieval (Sankaradas et al., 2024)
    - It's High Time: Temporal Information Retrieval Survey (Piryani et al., 2025)
    - When to use Graphs in RAG (Xiang et al., 2025)
    """)

if __name__ == "__main__":
    main()