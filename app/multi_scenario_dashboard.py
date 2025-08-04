# app/multi_scenario_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import time

# Configuração da página
st.set_page_config(
    page_title="Multi-Scenario RAG Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .scenario-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .performance-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        color: white;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .badge-excellent { background-color: #27ae60; }
    .badge-good { background-color: #f39c12; }
    .badge-poor { background-color: #e74c3c; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_comparison_results():
    """Carrega resultados da comparação multi-cenário"""
    results_file = Path("data/evaluation/multi_scenario_comparison.json")
    
    if not results_file.exists():
        st.error("⚠️ Arquivo de resultados não encontrado. Execute primeiro o MVP integrado.")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Erro ao carregar resultados: {e}")
        return None

@st.cache_data
def load_comparative_report():
    """Carrega relatório comparativo"""
    report_file = Path("data/evaluation/comparative_report.json")
    
    if not report_file.exists():
        return None
    
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Erro ao carregar relatório: {e}")
        return None

def create_performance_badge(value: float, metric_type: str) -> str:
    """Cria badge de performance baseado no valor e tipo de métrica"""
    if metric_type == "time":
        if value < 1.0:
            return "badge-excellent"
        elif value < 3.0:
            return "badge-good"
        else:
            return "badge-poor"
    elif metric_type == "cost":
        if value < 0.001:
            return "badge-excellent"
        elif value < 0.01:
            return "badge-good"
        else:
            return "badge-poor"
    elif metric_type == "relevance":
        if value > 0.8:
            return "badge-excellent"
        elif value > 0.6:
            return "badge-good"
        else:
            return "badge-poor"
    else:
        return "badge-good"

def display_scenario_metrics(scenario_name: str, metrics: Dict, col):
    """Exibe métricas de um cenário"""
    with col:
        st.markdown(f'<div class="scenario-header">{scenario_name}</div>', 
                   unsafe_allow_html=True)
        
        # Tempo de resposta
        time_badge = create_performance_badge(metrics['average_response_time_sec'], "time")
        st.markdown(f"""
        <div class="performance-badge {time_badge}">
            ⚡ {metrics['average_response_time_sec']:.3f}s
        </div>
        """, unsafe_allow_html=True)
        
        # Custo
        cost_badge = create_performance_badge(metrics['total_cost_usd'], "cost")
        st.markdown(f"""
        <div class="performance-badge {cost_badge}">
            💰 ${metrics['total_cost_usd']:.4f}
        </div>
        """, unsafe_allow_html=True)
        
        # Relevância
        relevance_badge = create_performance_badge(metrics['average_relevance_score'], "relevance")
        st.markdown(f"""
        <div class="performance-badge {relevance_badge}">
            🎯 {metrics['average_relevance_score']:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Taxa de sucesso
        st.metric("Taxa de Sucesso", f"{metrics['success_rate']:.1f}%", 
                 delta=f"{metrics['successful_queries']}/{metrics.get('total_queries', '?')}")

def create_comparison_chart(results: Dict):
    """Cria gráfico de comparação entre cenários"""
    metrics = results.get('aggregated_metrics', {})
    
    if not metrics:
        st.warning("⚠️ Métricas não disponíveis")
        return
    
    # Prepara dados para o gráfico
    scenarios = []
    response_times = []
    costs = []
    relevance_scores = []
    
    scenario_names = {
        'scenario_a': 'Vector RAG',
        'scenario_b': 'Hybrid RAG',
        'scenario_c': 'LLM-Only'
    }
    
    for scenario, data in metrics.items():
        if data['success_rate'] > 0:  # Só inclui cenários que funcionaram
            scenarios.append(scenario_names.get(scenario, scenario))
            response_times.append(data['avg_response_time'])
            costs.append(data['total_cost'])
            relevance_scores.append(data['avg_relevance'])
    
    if not scenarios:
        st.warning("⚠️ Nenhum cenário com dados válidos")
        return
    
    # Cria subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tempo de Resposta (s)', 'Custo Total (USD)', 
                       'Relevância Média', 'Comparação Normalizada'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "polar"}]]
    )
    
    # Gráfico de tempo de resposta
    fig.add_trace(
        go.Bar(x=scenarios, y=response_times, name="Tempo (s)", 
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # Gráfico de custo
    fig.add_trace(
        go.Bar(x=scenarios, y=costs, name="Custo (USD)", 
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Gráfico de relevância
    fig.add_trace(
        go.Bar(x=scenarios, y=relevance_scores, name="Relevância", 
               marker_color='lightgreen'),
        row=2, col=1
    )
    
    # Gráfico radar normalizado
    if len(scenarios) > 0:
        # Normaliza valores para 0-1
        norm_time = [1 - (t / max(response_times)) for t in response_times]
        norm_cost = [1 - (c / max(costs)) if max(costs) > 0 else 1 for c in costs]
        norm_relevance = relevance_scores
        
        for i, scenario in enumerate(scenarios):
            fig.add_trace(
                go.Scatterpolar(
                    r=[norm_time[i], norm_cost[i], norm_relevance[i]],
                    theta=['Velocidade', 'Economia', 'Precisão'],
                    fill='toself',
                    name=scenario
                ),
                row=2, col=2
            )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Comparação Multi-Cenário RAG")
    
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_results(results: Dict):
    """Exibe resultados detalhados por pergunta"""
    individual_results = results.get('individual_comparisons', [])
    
    if not individual_results:
        st.warning("⚠️ Resultados individuais não disponíveis")
        return
    
    for i, comparison in enumerate(individual_results):
        question_meta = comparison.get('question_metadata', {})
        question = comparison.get('question', 'Pergunta não disponível')
        
        with st.expander(f"📋 Pergunta {question_meta.get('id', i+1)}: {question}"):
            # Metadados da pergunta
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Categoria:** {question_meta.get('category', 'N/A')}")
            with col2:
                st.info(f"**Foco Temporal:** {question_meta.get('temporal_focus', 'N/A')}")
            with col3:
                st.info(f"**Complexidade:** {question_meta.get('complexity', 'N/A')}")
            
            # Conceitos esperados
            expected = question_meta.get('expected_concepts', [])
            if expected:
                st.write("**Conceitos Esperados:**", ", ".join(expected))
            
            # Resultados por cenário
            results_data = comparison.get('results', {})
            
            for scenario, result in results_data.items():
                scenario_name = {
                    'scenario_a': '🔍 Vector RAG',
                    'scenario_b': '🔗 Hybrid RAG', 
                    'scenario_c': '🤖 LLM-Only'
                }.get(scenario, scenario)
                
                if 'error' not in result:
                    with st.container():
                        st.markdown(f"**{scenario_name}**")
                        
                        # Métricas
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric("Tempo", f"{result.get('response_time', 0):.3f}s")
                        with metric_col2:
                            st.metric("Custo", f"${result.get('estimated_cost_usd', 0):.4f}")
                        with metric_col3:
                            st.metric("Relevância", f"{result.get('relevance_score', 0):.3f}")
                        
                        # Resposta
                        answer = result.get('answer', 'Resposta não disponível')
                        st.text_area("Resposta:", answer, height=150, 
                                   key=f"{scenario}_{i}_answer")
                        
                        # Metadados específicos do cenário
                        if scenario == 'scenario_b':
                            if result.get('seed_events', 0) > 0:
                                st.caption(f"🌱 Eventos semente: {result.get('seed_events', 0)} | "
                                         f"🔗 Eventos expandidos: {result.get('expanded_events', 0)}")
                        elif scenario == 'scenario_a':
                            if 'retrieved_chunks' in result:
                                st.caption(f"📄 Chunks recuperados: {len(result.get('retrieved_chunks', []))}")
                        elif scenario == 'scenario_c':
                            if result.get('using_mock'):
                                st.caption("🎭 Usando mock inteligente")
                            else:
                                st.caption(f"🌐 API: {result.get('model', 'N/A')}")
                        
                        st.divider()
                else:
                    st.error(f"**{scenario_name}**: {result['error']}")

def display_recommendations(report: Dict):
    """Exibe recomendações do relatório"""
    recommendations = report.get('recommendations', {})
    
    if not recommendations:
        st.warning("⚠️ Recomendações não disponíveis")
        return
    
    # Melhores em cada categoria
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_speed = recommendations.get('best_for_speed')
        if best_speed:
            scenario_name = {
                'scenario_a': 'Vector RAG',
                'scenario_b': 'Hybrid RAG',
                'scenario_c': 'LLM-Only'
            }.get(best_speed, best_speed)
            st.success(f"⚡ **Mais Rápido**\n\n{scenario_name}")
    
    with col2:
        best_cost = recommendations.get('best_for_cost')
        if best_cost:
            scenario_name = {
                'scenario_a': 'Vector RAG',
                'scenario_b': 'Hybrid RAG',
                'scenario_c': 'LLM-Only'
            }.get(best_cost, best_cost)
            st.success(f"💰 **Mais Econômico**\n\n{scenario_name}")
    
    with col3:
        best_accuracy = recommendations.get('best_for_accuracy')
        if best_accuracy:
            scenario_name = {
                'scenario_a': 'Vector RAG',
                'scenario_b': 'Hybrid RAG',
                'scenario_c': 'LLM-Only'
            }.get(best_accuracy, best_accuracy)
            st.success(f"🎯 **Mais Preciso**\n\n{scenario_name}")
    
    # Recomendação geral
    overall = recommendations.get('overall_recommendation')
    if overall:
        scenario_name = {
            'scenario_a': 'Vector RAG',
            'scenario_b': 'Hybrid RAG',
            'scenario_c': 'LLM-Only'
        }.get(overall, overall)
        
        st.info(f"🌟 **Recomendação Geral:** {scenario_name}")
    
    # Análise detalhada
    analysis = recommendations.get('detailed_analysis', [])
    if analysis:
        st.markdown("**📋 Análise Detalhada:**")
        for point in analysis:
            st.markdown(f"• {point}")

def interactive_query_tester():
    """Interface para testar consultas interativamente"""
    st.subheader("🧪 Teste Interativo")
    
    # Input da pergunta
    user_question = st.text_input("Digite sua pergunta sobre construção civil:")
    
    if st.button("🚀 Testar nos 3 Cenários") and user_question:
        try:
            # Adiciona path para importar o sistema
            sys.path.append('src')
            from multi_scenario_system import MultiScenarioSystem
            
            # Carrega documentos
            data_dir = Path("data/raw")
            documents = []
            
            for json_file in data_dir.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    documents.append(doc)
            
            if not documents:
                st.error("❌ Nenhum documento encontrado")
                return
            
            # Inicializa sistema
            with st.spinner("Inicializando sistema..."):
                system = MultiScenarioSystem()
                system.load_documents(documents)
                system.build_all_scenarios()
            
            # Executa comparação
            with st.spinner("Executando consulta nos 3 cenários..."):
                comparison = system.compare_all_scenarios(user_question)
            
            # Exibe resultados
            st.success("✅ Consulta executada!")
            
            results = comparison.get('results', {})
            
            # Cria colunas para os resultados
            col1, col2, col3 = st.columns(3)
            
            columns = [col1, col2, col3]
            scenario_names = ['🔍 Vector RAG', '🔗 Hybrid RAG', '🤖 LLM-Only']
            scenario_keys = ['scenario_a', 'scenario_b', 'scenario_c']
            
            for i, (col, name, key) in enumerate(zip(columns, scenario_names, scenario_keys)):
                with col:
                    st.markdown(f"**{name}**")
                    
                    result = results.get(key, {})
                    
                    if 'error' not in result:
                        # Métricas
                        st.metric("Tempo", f"{result.get('response_time', 0):.3f}s")
                        st.metric("Custo", f"${result.get('estimated_cost_usd', 0):.4f}")
                        st.metric("Relevância", f"{result.get('relevance_score', 0):.3f}")
                        
                        # Resposta
                        answer = result.get('answer', 'Sem resposta')
                        st.text_area("Resposta:", answer, height=200, 
                                   key=f"interactive_{key}")
                    else:
                        st.error(f"Erro: {result['error']}")
            
            # Análise da comparação
            analysis = comparison.get('comparison', {})
            if analysis:
                st.subheader("📊 Análise da Comparação")
                
                fastest = analysis.get('fastest_scenario')
                cheapest = analysis.get('cheapest_scenario')
                most_relevant = analysis.get('most_relevant_scenario')
                
                if fastest:
                    st.info(f"⚡ Mais rápido: {fastest}")
                if cheapest:
                    st.info(f"💰 Mais econômico: {cheapest}")
                if most_relevant:
                    st.info(f"🎯 Mais relevante: {most_relevant}")
                    
        except Exception as e:
            st.error(f"❌ Erro ao executar teste: {e}")

def main():
    """Função principal do dashboard"""
    
    # Header
    st.title("🤖 Multi-Scenario RAG Comparison Dashboard")
    st.markdown("Comparação entre **Vector RAG**, **Hybrid RAG** e **LLM-Only** para domínio de construção civil")
    
    # Sidebar
    st.sidebar.title("📊 Navegação")
    
    # Carrega dados
    results = load_comparison_results()
    report = load_comparative_report()
    
    if results is None:
        st.error("❌ Execute primeiro o script `run_integrated_mvp.py` para gerar os resultados.")
        st.stop()
    
    # Menu de navegação
    menu_options = [
        "📋 Resumo Executivo",
        "📊 Comparação de Performance", 
        "🔍 Resultados Detalhados",
        "🎯 Recomendações",
        "🧪 Teste Interativo"
    ]
    
    selected_page = st.sidebar.selectbox("Selecione uma página:", menu_options)
    
    # Informações gerais na sidebar
    if results:
        summary = results.get('test_summary', {})
        st.sidebar.markdown("### 📈 Informações Gerais")
        st.sidebar.metric("Perguntas Testadas", summary.get('total_questions', 0))
        st.sidebar.metric("Documentos Usados", summary.get('documents_used', 0))
        
        execution_time = results.get('execution_timestamp', 'N/A')
        st.sidebar.caption(f"Última execução: {execution_time}")
    
    # Conteúdo principal baseado na página selecionada
    if selected_page == "📋 Resumo Executivo":
        st.header("📋 Resumo Executivo")
        
        if report and 'scenario_comparison' in report:
            st.markdown("### 🎯 Performance por Cenário")
            
            # Métricas principais
            metrics = report['scenario_comparison']
            
            if len(metrics) >= 3:
                col1, col2, col3 = st.columns(3)
                columns = [col1, col2, col3]
                
                for i, (scenario, data) in enumerate(metrics.items()):
                    if i < 3:  # Limita a 3 colunas
                        display_scenario_metrics(scenario, data, columns[i])
            
            # Gráfico de comparação resumido
            st.markdown("### 📊 Visão Geral")
            create_comparison_chart(results)
            
        else:
            st.warning("⚠️ Relatório comparativo não disponível")
    
    elif selected_page == "📊 Comparação de Performance":
        st.header("📊 Comparação de Performance")
        
        # Gráfico principal
        create_comparison_chart(results)
        
        # Tabela de métricas
        if 'aggregated_metrics' in results:
            st.subheader("📋 Métricas Detalhadas")
            
            metrics_data = []
            for scenario, data in results['aggregated_metrics'].items():
                scenario_name = {
                    'scenario_a': 'Vector RAG',
                    'scenario_b': 'Hybrid RAG',
                    'scenario_c': 'LLM-Only'
                }.get(scenario, scenario)
                
                metrics_data.append({
                    'Cenário': scenario_name,
                    'Tempo Médio (s)': f"{data['avg_response_time']:.3f}",
                    'Custo Total (USD)': f"${data['total_cost']:.4f}",
                    'Relevância Média': f"{data['avg_relevance']:.3f}",
                    'Taxa de Sucesso (%)': f"{data['success_rate']*100:.1f}",
                    'Consultas Bem-sucedidas': data['successful_queries']
                })
            
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True)
    
    elif selected_page == "🔍 Resultados Detalhados":
        st.header("🔍 Resultados Detalhados")
        st.markdown("Análise pergunta por pergunta dos três cenários:")
        
        display_detailed_results(results)
    
    elif selected_page == "🎯 Recomendações":
        st.header("🎯 Recomendações")
        
        if report:
            display_recommendations(report)
        else:
            st.warning("⚠️ Relatório de recomendações não disponível")
    
    elif selected_page == "🧪 Teste Interativo":
        st.header("🧪 Teste Interativo")
        st.markdown("Teste seus próprios prompts nos três cenários:")
        
        interactive_query_tester()
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard Multi-Cenário RAG - Construção Civil*")

if __name__ == "__main__":
    main()
    