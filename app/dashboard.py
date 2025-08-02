import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="LLM Architecture Comparison - MVP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar dados
@st.cache_data
def load_evaluation_data():
    """Carrega dados de avaliação"""
    data_dir = Path("data/evaluation")
    
    # Dados padrão vazios
    rag_results = []
    api_results = []
    cost_estimates = []
    evaluation_report = {}
    summary_report = {}
    
    # Carrega RAG results
    rag_file = data_dir / "rag_results.json"
    if rag_file.exists():
        with open(rag_file, 'r', encoding='utf-8') as f:
            rag_results = json.load(f)
    
    # Carrega API results  
    api_file = data_dir / "api_baseline_results.json"
    if api_file.exists():
        with open(api_file, 'r', encoding='utf-8') as f:
            api_results = json.load(f)
    
    # Carrega estimates
    cost_file = data_dir / "cost_estimates.json"
    if cost_file.exists():
        with open(cost_file, 'r', encoding='utf-8') as f:
            cost_estimates = json.load(f)
    
    # Carrega evaluation report
    eval_file = data_dir / "evaluation_report.json"
    if eval_file.exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            evaluation_report = json.load(f)
    
    # Carrega summary report
    summary_file = data_dir / "mvp_summary_report.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_report = json.load(f)
    
    return rag_results, api_results, cost_estimates, evaluation_report, summary_report

def create_performance_chart(rag_results, api_results):
    """Cria gráfico de performance (tempo de resposta)"""
    data = []
    
    if rag_results:
        for r in rag_results:
            data.append({
                'Architecture': 'RAG Local',
                'Question': r.get('question_id', 'N/A'),
                'Response Time (s)': r.get('response_time', 0),
                'Relevance Score': r.get('relevance_score', 0)
            })
    
    if api_results:
        for r in api_results:
            data.append({
                'Architecture': 'API Externa',
                'Question': r.get('question_id', 'N/A'),
                'Response Time (s)': r.get('response_time', 0),
                'Relevance Score': r.get('relevance_score', 0)
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = px.box(df, x='Architecture', y='Response Time (s)', 
                 title='Tempo de Resposta por Arquitetura',
                 color='Architecture')
    
    fig.update_layout(showlegend=False)
    return fig

def create_cost_comparison_chart(cost_estimates):
    """Cria gráfico de comparação de custos"""
    if not cost_estimates:
        return None
    
    df = pd.DataFrame(cost_estimates)
    
    fig = px.bar(df, x='scenario', y='monthly_cost_usd', 
                 color='architecture',
                 title='Comparação de Custos Mensais por Cenário',
                 labels={'monthly_cost_usd': 'Custo Mensal (USD)', 'scenario': 'Cenário'})
    
    return fig

def create_quality_metrics_chart(rag_results, api_results):
    """Cria gráfico de métricas de qualidade"""
    if not rag_results and not api_results:
        return None
    
    metrics = []
    
    if rag_results:
        relevance_scores = [r.get('relevance_score', 0) for r in rag_results]
        metrics.append({
            'Architecture': 'RAG Local',
            'Avg Relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'Min Relevance': np.min(relevance_scores) if relevance_scores else 0,
            'Max Relevance': np.max(relevance_scores) if relevance_scores else 0
        })
    
    if api_results:
        # Para API, simulamos relevância baseada na qualidade da resposta
        relevance_scores = [0.8 if not r.get('answer', '').startswith('❌') else 0.2 for r in api_results]
        metrics.append({
            'Architecture': 'API Externa',
            'Avg Relevance': np.mean(relevance_scores) if relevance_scores else 0,
            'Min Relevance': np.min(relevance_scores) if relevance_scores else 0,
            'Max Relevance': np.max(relevance_scores) if relevance_scores else 0
        })
    
    df = pd.DataFrame(metrics)
    
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            name=row['Architecture'],
            x=['Relevância Média'],
            y=[row['Avg Relevance']],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[row['Max Relevance'] - row['Avg Relevance']],
                arrayminus=[row['Avg Relevance'] - row['Min Relevance']]
            )
        ))
    
    fig.update_layout(title='Métricas de Qualidade', yaxis_title='Score')
    return fig

def display_detailed_results(results, title):
    """Exibe resultados detalhados em formato expandível"""
    st.subheader(title)
    
    if not results:
        st.warning("Nenhum resultado disponível")
        return
    
    for i, result in enumerate(results):
        with st.expander(f"Pergunta {result.get('question_id', i+1)}: {result.get('question', 'N/A')[:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Tempo de Resposta", f"{result.get('response_time', 0):.2f}s")
                if 'relevance_score' in result:
                    st.metric("Relevância", f"{result.get('relevance_score', 0):.3f}")
                if 'estimated_cost_usd' in result:
                    st.metric("Custo Estimado", f"${result.get('estimated_cost_usd', 0):.4f}")
            
            with col2:
                if 'retrieved_docs' in result:
                    st.metric("Docs Recuperados", result['retrieved_docs'])
                if 'tokens_used' in result:
                    st.metric("Tokens Usados", result['tokens_used'])
            
            st.write("**Resposta:**")
            st.write(result.get('answer', 'N/A'))

def show_recommendations(evaluation_report):
    """Mostra recomendações do sistema"""
    if not evaluation_report or 'recommendations' not in evaluation_report:
        st.warning("Recomendações não disponíveis")
        return
    
    rec = evaluation_report['recommendations']
    
    # Recomendação principal
    primary = rec.get('primary_recommendation', 'N/A')
    
    if primary == 'rag_local':
        st.success("🎯 **Recomendação: RAG Local**")
    elif primary == 'api_external':
        st.info("🎯 **Recomendação: API Externa**")
    elif primary == 'volume_dependent':
        st.warning("🎯 **Recomendação: Depende do Volume**")
    else:
        st.error("🎯 **Recomendação: Análise Adicional Necessária**")
    
    # Reasoning
    reasoning = rec.get('reasoning', [])
    if reasoning:
        st.write("**Justificativa:**")
        for reason in reasoning:
            st.write(f"• {reason}")
    
    # Trade-offs
    trade_offs = rec.get('trade_offs', {})
    if trade_offs:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Vantagens RAG Local:**")
            for advantage in trade_offs.get('rag_advantages', []):
                st.write(f"• {advantage}")
            
            st.write("**❌ Desvantagens RAG Local:**")
            for disadvantage in trade_offs.get('rag_disadvantages', []):
                st.write(f"• {disadvantage}")
        
        with col2:
            st.write("**✅ Vantagens API Externa:**")
            for advantage in trade_offs.get('api_advantages', []):
                st.write(f"• {advantage}")
            
            st.write("**❌ Desvantagens API Externa:**")
            for disadvantage in trade_offs.get('api_disadvantages', []):
                st.write(f"• {disadvantage}")
    
    # Próximos passos
    next_steps = rec.get('next_steps', [])
    if next_steps:
        st.write("**📋 Próximos Passos:**")
        for step in next_steps:
            st.write(f"• {step}")

# Interface principal
def main():
    st.title("🤖 LLM Architecture Comparison - MVP")
    st.markdown("**Análise comparativa entre RAG Local vs APIs Externas**")
    
    # Carrega dados
    try:
        rag_results, api_results, cost_estimates, evaluation_report, summary_report = load_evaluation_data()
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()
    
    # Sidebar com informações do sistema
    with st.sidebar:
        st.header("ℹ️ Informações do Sistema")
        
        if summary_report:
            exec_date = summary_report.get('mvp_summary', {}).get('execution_date', 'N/A')
            st.write(f"**Última execução:** {exec_date}")
            
            perf = summary_report.get('mvp_summary', {}).get('rag_performance', {})
            if perf:
                st.metric("Tempo Médio RAG", f"{perf.get('avg_response_time_sec', 0):.2f}s")
                st.metric("Relevância Média", f"{perf.get('avg_relevance_score', 0):.3f}")
                st.metric("Taxa de Sucesso", f"{perf.get('success_rate', 0):.1%}")
        
        st.markdown("---")
        st.write("**Status dos Dados:**")
        st.write(f"✅ RAG: {len(rag_results)} consultas" if rag_results else "❌ RAG: Sem dados")
        st.write(f"✅ API: {len(api_results)} consultas" if api_results else "❌ API: Sem dados")
        st.write(f"✅ Custos: {len(cost_estimates)} cenários" if cost_estimates else "❌ Custos: Sem dados")
        
        # Botão para recarregar dados
        if st.button("🔄 Recarregar Dados"):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Resumo Executivo", 
        "⚡ Performance", 
        "💰 Custos", 
        "🎯 Qualidade", 
        "🔍 Detalhes"
    ])
    
    with tab1:
        st.header("📋 Resumo Executivo")
        
        if not rag_results and not api_results:
            st.warning("⚠️ Execute primeiro `python run_mvp.py` para gerar dados!")
            st.code("python run_mvp.py")
            st.stop()
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_questions = len(set(
                [r.get('question_id') for r in rag_results] + 
                [r.get('question_id') for r in api_results]
            ))
            st.metric("Perguntas Testadas", total_questions)
        
        with col2:
            architectures = []
            if rag_results: architectures.append("RAG")
            if api_results: architectures.append("API")
            st.metric("Arquiteturas", len(architectures))
        
        with col3:
            if rag_results:
                avg_time = sum(r.get('response_time', 0) for r in rag_results) / len(rag_results)
                st.metric("Tempo Médio RAG", f"{avg_time:.2f}s")
        
        with col4:
            if api_results:
                total_cost = sum(r.get('estimated_cost_usd', 0) for r in api_results)
                st.metric("Custo Total API", f"${total_cost:.4f}")
        
        st.markdown("---")
        
        # Recomendações
        show_recommendations(evaluation_report)
    
    with tab2:
        st.header("⚡ Performance")
        
        # Gráfico de tempo de resposta
        perf_chart = create_performance_chart(rag_results, api_results)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        else:
            st.warning("Dados de performance não disponíveis")
        
        # Estatísticas detalhadas
        if rag_results or api_results:
            st.subheader("📊 Estatísticas Detalhadas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if rag_results:
                    st.write("**RAG Local:**")
                    times = [r.get('response_time', 0) for r in rag_results]
                    st.write(f"• Média: {np.mean(times):.2f}s")
                    st.write(f"• Mediana: {np.median(times):.2f}s")
                    st.write(f"• Min/Max: {np.min(times):.2f}s / {np.max(times):.2f}s")
                    st.write(f"• Abaixo de 3s: {sum(1 for t in times if t < 3.0) / len(times):.1%}")
            
            with col2:
                if api_results:
                    st.write("**API Externa:**")
                    times = [r.get('response_time', 0) for r in api_results]
                    st.write(f"• Média: {np.mean(times):.2f}s")
                    st.write(f"• Mediana: {np.median(times):.2f}s")
                    st.write(f"• Min/Max: {np.min(times):.2f}s / {np.max(times):.2f}s")
                    st.write(f"• Abaixo de 3s: {sum(1 for t in times if t < 3.0) / len(times):.1%}")
    
    with tab3:
        st.header("💰 Análise de Custos")
        
        # Gráfico de custos
        cost_chart = create_cost_comparison_chart(cost_estimates)
        if cost_chart:
            st.plotly_chart(cost_chart, use_container_width=True)
        else:
            st.warning("Dados de custo não disponíveis")
        
        # Tabela de custos
        if cost_estimates:
            st.subheader("📊 Tabela de Custos")
            
            df_costs = pd.DataFrame(cost_estimates)
            
            # Reformata para melhor visualização
            pivot_df = df_costs.pivot(index='scenario', columns='architecture', values='monthly_cost_usd')
            
            st.dataframe(pivot_df.style.format("${:.2f}"), use_container_width=True)
            
            # Economia estimada
            if 'rag_local' in pivot_df.columns and 'openai_api' in pivot_df.columns:
                st.subheader("💡 Economia Estimada (RAG vs API)")
                
                for scenario in pivot_df.index:
                    rag_cost = pivot_df.loc[scenario, 'rag_local']
                    api_cost = pivot_df.loc[scenario, 'openai_api']
                    savings = api_cost - rag_cost
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{scenario}", f"${savings:.2f}/mês")
                    with col2:
                        st.write(f"Economia anual: ${savings * 12:.2f}")
                    with col3:
                        if api_cost > 0:
                            pct_savings = (savings / api_cost) * 100
                            st.write(f"Redução: {pct_savings:.1f}%")
    
    with tab4:
        st.header("🎯 Análise de Qualidade")
        
        # Gráfico de qualidade
        quality_chart = create_quality_metrics_chart(rag_results, api_results)
        if quality_chart:
            st.plotly_chart(quality_chart, use_container_width=True)
        else:
            st.warning("Dados de qualidade não disponíveis")
        
        # Análise por categoria
        if rag_results:
            st.subheader("📂 Análise por Categoria")
            
            categories = {}
            for result in rag_results:
                cat = result.get('category', 'unknown')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result.get('relevance_score', 0))
            
            cat_data = []
            for cat, scores in categories.items():
                cat_data.append({
                    'Categoria': cat,
                    'Média': np.mean(scores),
                    'Consultas': len(scores),
                    'Min/Max': f"{np.min(scores):.3f} / {np.max(scores):.3f}"
                })
            
            if cat_data:
                df_cats = pd.DataFrame(cat_data)
                st.dataframe(df_cats, use_container_width=True)
    
    with tab5:
        st.header("🔍 Resultados Detalhados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_detailed_results(rag_results, "🤖 RAG Local")
        
        with col2:
            display_detailed_results(api_results, "🌐 API Externa")
        
        # Dados brutos (opcional)
        if st.checkbox("Mostrar dados brutos JSON"):
            st.subheader("📄 Dados Brutos")
            
            tab_rag, tab_api, tab_costs = st.tabs(["RAG", "API", "Custos"])
            
            with tab_rag:
                if rag_results:
                    st.json(rag_results)
                else:
                    st.write("Sem dados RAG")
            
            with tab_api:
                if api_results:
                    st.json(api_results)
                else:
                    st.write("Sem dados API")
            
            with tab_costs:
                if cost_estimates:
                    st.json(cost_estimates)
                else:
                    st.write("Sem dados de custo")

    # Footer
    st.markdown("---")
    st.markdown("**MVP LLM Architecture Comparison** - Desenvolvido para análise rápida de viabilidade técnica e econômica")

if __name__ == "__main__":
    main()
