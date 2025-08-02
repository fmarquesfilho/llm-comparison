# app/streamlit_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.evaluation.cost_analysis import CostAnalyzer
    from src.evaluation.metrics import EvaluationMetrics
    from src.config import Config
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="LLM Architecture Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para Mac-style
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .recommendation-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2196f3;
    }
    .warning-box {
        background: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_evaluation_results():
    """Carrega resultados das avaliações"""
    results_file = Path("data/evaluation/metrics_results.json")
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Dados de exemplo para demonstração
        return {
            'rag_simple': {
                'rouge1': 0.65, 'rouge2': 0.45, 'rougeL': 0.58,
                'bleu': 0.52, 'semantic_similarity': 0.68,
                'factual_accuracy': 0.70, 'overall_score': 0.63,
                'avg_length': 145, 'response_rate': 0.95,
                'avg_response_time_ms': 450
            },
            'rag_optimized': {
                'rouge1': 0.72, 'rouge2': 0.52, 'rougeL': 0.65,
                'bleu': 0.58, 'semantic_similarity': 0.74,
                'factual_accuracy': 0.75, 'overall_score': 0.69,
                'avg_length': 162, 'response_rate': 0.97,
                'avg_response_time_ms': 680
            },
            'fine_tuned': {
                'rouge1': 0.78, 'rouge2': 0.58, 'rougeL': 0.72,
                'bleu': 0.65, 'semantic_similarity': 0.79,
                'factual_accuracy': 0.82, 'overall_score': 0.76,
                'avg_length': 178, 'response_rate': 0.98,
                'avg_response_time_ms': 1200
            },
            'hybrid': {
                'rouge1': 0.82, 'rouge2': 0.64, 'rougeL': 0.76,
                'bleu': 0.71, 'semantic_similarity': 0.83,
                'factual_accuracy': 0.85, 'overall_score': 0.80,
                'avg_length': 195, 'response_rate': 0.99,
                'avg_response_time_ms': 950
            },
            'api_external': {
                'rouge1': 0.85, 'rouge2': 0.68, 'rougeL': 0.79,
                'bleu': 0.74, 'semantic_similarity': 0.87,
                'factual_accuracy': 0.88, 'overall_score': 0.83,
                'avg_length': 210, 'response_rate': 0.99,
                'avg_response_time_ms': 2100
            }
        }

@st.cache_data
def load_cost_analysis():
    """Carrega análise de custos"""
    cost_file = Path("data/evaluation/cost_analysis.csv")
    
    if cost_file.exists():
        return pd.read_csv(cost_file)
    else:
        # Gera análise de custos em tempo real
        analyzer = CostAnalyzer()
        scenarios = [
            {'name': 'Piloto', 'queries_per_day': 100, 'environment': 'local', 'model_size': '7b'},
            {'name': 'Produção Pequena', 'queries_per_day': 1000, 'environment': 'local', 'model_size': '7b'},
            {'name': 'Produção Média', 'queries_per_day': 5000, 'environment': 'cloud_aws', 'model_size': '7b'},
            {'name': 'Produção Grande', 'queries_per_day': 20000, 'environment': 'cloud_aws', 'model_size': '13b'}
        ]
        return analyzer.compare_architectures(scenarios)

def create_radar_chart(metrics_data):
    """Cria gráfico radar para comparação de métricas"""
    
    # Métricas principais para o radar
    radar_metrics = ['rouge1', 'rougeL', 'bleu', 'semantic_similarity', 'factual_accuracy']
    metric_labels = ['ROUGE-1', 'ROUGE-L', 'BLEU', 'Similaridade Semântica', 'Acurácia Factual']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, (arch_name, metrics) in enumerate(metrics_data.items()):
        values = [metrics.get(metric, 0) for metric in radar_metrics]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels,
            fill='toself',
            name=arch_name.replace('_', ' ').title(),
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)],
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.2f'
            )
        ),
        showlegend=True,
        height=500,
        title="Comparação de Performance por Arquitetura"
    )
    
    return fig

def create_cost_comparison_chart(cost_df, scenario_filter):
    """Cria gráfico de comparação de custos"""
    
    filtered_df = cost_df[cost_df['scenario'] == scenario_filter]
    
    if filtered_df.empty:
        return go.Figure().add_annotation(text="Dados não disponíveis", x=0.5, y=0.5)
    
    # Gráfico de barras com custos
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Custo Total 12 meses (USD)', 'Custo por 1K queries (USD)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    architectures = filtered_df['architecture'].tolist()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Custo total
    fig.add_trace(
        go.Bar(
            x=architectures,
            y=filtered_df['total_cost_12m_usd'],
            name='Custo Total',
            marker_color=colors[:len(architectures)],
            text=[f'${x:,.0f}' for x in filtered_df['total_cost_12m_usd']],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Custo por 1K queries
    fig.add_trace(
        go.Bar(
            x=architectures,
            y=filtered_df['cost_per_1k_queries_usd'],
            name='Custo/1K queries',
            marker_color=colors[:len(architectures)],
            text=[f'${x:.3f}' for x in filtered_df['cost_per_1k_queries_usd']],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def create_roi_analysis_chart(cost_df, scenario_filter):
    """Cria análise de ROI"""
    
    filtered_df = cost_df[cost_df['scenario'] == scenario_filter]
    
    if filtered_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Scatter plot: Setup Cost vs ROI
    fig.add_trace(go.Scatter(
        x=filtered_df['setup_cost_usd'],
        y=filtered_df['roi_annual_percent'],
        mode='markers+text',
        text=filtered_df['architecture'],
        textposition='top center',
        marker=dict(
            size=filtered_df['monthly_cost_usd'] / 10,  # Tamanho baseado no custo mensal
            color=filtered_df['payback_months'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Payback (meses)")
        ),
        name='Arquiteturas'
    ))
    
    fig.update_layout(
        title='Análise ROI: Investimento vs Retorno',
        xaxis_title='Custo de Setup (USD)',
        yaxis_title='ROI Anual (%)',
        height=400
    )
    
    return fig

def main():
    """Função principal do dashboard"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🤖 LLM Architecture Comparison Dashboard</h1>
        <p style="color: white; margin: 0; opacity: 0.8;">Análise comparativa de arquiteturas para construção civil</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configurações")
    
    # Filtros
    scenario_options = ['Piloto', 'Produção Pequena', 'Produção Média', 'Produção Grande']
    selected_scenario = st.sidebar.selectbox(
        "📊 Cenário de Análise:",
        scenario_options,
        index=1  # Produção Pequena como padrão
    )
    
    # Parâmetros personalizados
    st.sidebar.subheader("🎛️ Parâmetros Customizados")
    
    custom_queries = st.sidebar.slider(
        "Consultas por dia:",
        min_value=10,
        max_value=50000,
        value=1000,
        step=100
    )
    
    custom_budget = st.sidebar.slider(
        "Orçamento mensal (USD):",
        min_value=50,
        max_value=10000,
        value=500,
        step=50
    )
    
    tech_expertise = st.sidebar.selectbox(
        "Expertise técnica da equipe:",
        ["Baixa", "Média", "Alta"],
        index=1
    )
    
    # Carrega dados
    try:
        metrics_data = load_evaluation_results()
        cost_df = load_cost_analysis()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()
    
    # Métricas principais
    st.header("📈 Resumo Executivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_performance = max(metrics_data.items(), key=lambda x: x[1]['overall_score'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏆 Melhor Performance</h4>
            <h3>{best_performance[0].replace('_', ' ').title()}</h3>
            <p>Score: {best_performance[1]['overall_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        scenario_costs = cost_df[cost_df['scenario'] == selected_scenario]
        if not scenario_costs.empty:
            best_cost = scenario_costs.loc[scenario_costs['total_cost_12m_usd'].idxmin()]
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Menor Custo</h4>
                <h3>{best_cost['architecture'].replace('_', ' ').title()}</h3>
                <p>${best_cost['total_cost_12m_usd']:,.0f}/ano</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if not scenario_costs.empty:
            best_roi = scenario_costs.loc[scenario_costs['roi_annual_percent'].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Melhor ROI</h4>
                <h3>{best_roi['architecture'].replace('_', ' ').title()}</h3>
                <p>{best_roi['roi_annual_percent']:.1f}% ao ano</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        avg_response_time = np.mean([m['avg_response_time_ms'] for m in metrics_data.values()])
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Tempo Médio</h4>
            <h3>{avg_response_time:.0f}ms</h3>
            <p>Resposta média</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Performance Comparativa")
        radar_fig = create_radar_chart(metrics_data)
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with col2:
        st.subheader("💹 Análise de Custos")
        cost_fig = create_cost_comparison_chart(cost_df, selected_scenario)
        st.plotly_chart(cost_fig, use_container_width=True)
    
    # ROI Analysis
    st.subheader("📊 Análise de Retorno sobre Investimento")
    roi_fig = create_roi_analysis_chart(cost_df, selected_scenario)
    st.plotly_chart(roi_fig, use_container_width=True)
    
    # Tabela detalhada
    st.subheader("📋 Comparação Detalhada")
    
    # Combina dados de performance e custo
    if not scenario_costs.empty:
        detailed_data = []
        
        for _, cost_row in scenario_costs.iterrows():
            arch = cost_row['architecture']
            metrics = metrics_data.get(arch, {})
            
            detailed_data.append({
                'Arquitetura': arch.replace('_', ' ').title(),
                'Score Geral': metrics.get('overall_score', 0),
                'ROUGE-L': metrics.get('rougeL', 0),
                'Acurácia Factual': metrics.get('factual_accuracy', 0),
                'Tempo Resposta (ms)': metrics.get('avg_response_time_ms', 0),
                'Custo Setup (USD)': cost_row['setup_cost_usd'],
                'Custo Mensal (USD)': cost_row['monthly_cost_usd'],
                'Custo Total/Ano (USD)': cost_row['total_cost_12m_usd'],
                'Custo/Query (USD)': cost_row['cost_per_query_usd'],
                'ROI Anual (%)': cost_row['roi_annual_percent'],
                'Payback (meses)': cost_row['payback_months']
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        
        # Formata colunas numéricas
        format_dict = {
            'Score Geral': '{:.3f}',
            'ROUGE-L': '{:.3f}',
            'Acurácia Factual': '{:.3f}',
            'Tempo Resposta (ms)': '{:.0f}',
            'Custo Setup (USD)': '${:,.0f}',
            'Custo Mensal (USD)': '${:,.0f}',
            'Custo Total/Ano (USD)': '${:,.0f}',
            'Custo/Query (USD)': '${:.4f}',
            'ROI Anual (%)': '{:.1f}%',
            'Payback (meses)': '{:.1f}'
        }
        
        st.dataframe(
            detailed_df.style.format(format_dict),
            use_container_width=True
        )
    
    # Recomendações personalizadas
    st.subheader("🎯 Recomendações Personalizadas")
    
    try:
        analyzer = CostAnalyzer()
        
        custom_scenario = {
            'name': 'Custom',
            'queries_per_day': custom_queries,
            'budget_monthly': custom_budget,
            'technical_expertise': tech_expertise.lower(),
            'environment': 'local',
            'model_size': '7b'
        }
        
        recommendations = analyzer.get_architecture_recommendations(custom_scenario)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            primary = recommendations.get('primary_recommendation', 'N/A')
            alternative = recommendations.get('alternative', 'N/A')
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>🥇 Recomendação Principal</h4>
                <h3>{primary.replace('_', ' ').title()}</h3>
                <h4>🥈 Alternativa</h4>
                <h3>{alternative.replace('_', ' ').title()}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            reasoning = recommendations.get('reasoning', [])
            if reasoning:
                st.markdown("**💡 Justificativa:**")
                for reason in reasoning:
                    st.write(f"• {reason}")
    
    except Exception as e:
        st.warning(f"Não foi possível gerar recomendações: {e}")
    
    # Análise de cenários
    st.subheader("📈 Análise de Cenários")
    
    scenario_analysis = st.expander("Ver análise detalhada por cenário")
    
    with scenario_analysis:
        for scenario in scenario_options:
            scenario_data = cost_df[cost_df['scenario'] == scenario]
            
            if not scenario_data.empty:
                st.write(f"**{scenario}:**")
                
                queries = scenario_data['queries_per_day'].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cheapest = scenario_data.loc[scenario_data['total_cost_12m_usd'].idxmin()]
                    st.metric(
                        "💰 Mais Econômico",
                        cheapest['architecture'].replace('_', ' ').title(),
                        f"${cheapest['total_cost_12m_usd']:,.0f}/ano"
                    )
                
                with col2:
                    best_roi = scenario_data.loc[scenario_data['roi_annual_percent'].idxmax()]
                    st.metric(
                        "📊 Melhor ROI",
                        best_roi['architecture'].replace('_', ' ').title(),
                        f"{best_roi['roi_annual_percent']:.1f}%"
                    )
                
                with col3:
                    best_efficiency = scenario_data.loc[scenario_data['cost_per_query_usd'].idxmin()]
                    st.metric(
                        "⚡ Mais Eficiente",
                        best_efficiency['architecture'].replace('_', ' ').title(),
                        f"${best_efficiency['cost_per_query_usd']:.4f}/query"
                    )
                
                st.write("---")
    
    # Insights e alertas
    st.subheader("💡 Insights e Alertas")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        # Alerta de custo por query
        if not scenario_costs.empty:
            max_cost_per_query = scenario_costs['cost_per_query_usd'].max()
            min_cost_per_query = scenario_costs['cost_per_query_usd'].min()
            
            if max_cost_per_query > min_cost_per_query * 3:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Diferença Significativa de Custos</h4>
                    <p>A diferença entre a arquitetura mais cara e mais barata é de <strong>{max_cost_per_query/min_cost_per_query:.1f}x</strong> por query.</p>
                    <p>Considere cuidadosamente o volume esperado antes de escolher.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with insights_col2:
        # Alerta de ROI
        if not scenario_costs.empty:
            negative_roi_count = len(scenario_costs[scenario_costs['roi_annual_percent'] < 0])
            
            if negative_roi_count > 0:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>📉 ROI Negativo</h4>
                    <p><strong>{negative_roi_count}</strong> arquitetura(s) apresentam ROI negativo no primeiro ano.</p>
                    <p>Considere aumentar o valor por query ou reduzir custos operacionais.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; opacity: 0.7;">
        <p>🤖 LLM Architecture Comparison Dashboard | Atualizado em {}</p>
        <p>Dados baseados em cenários simulados para construção civil</p>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    