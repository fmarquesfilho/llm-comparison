# app/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="RAG Comparison Dashboard",
    page_icon="🤖",
    layout="wide"
)

@st.cache_data
def load_results():
    """Carrega os resultados da comparação do arquivo JSON."""
    results_file = Path("data/evaluation/multi_scenario_comparison.json")
    if not results_file.exists():
        st.error(f"Arquivo de resultados '{results_file}' não encontrado. Execute 'run_integrated_mvp.py' primeiro.")
        return None
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_summary_chart(results_data):
    """Cria um gráfico de barras comparando as métricas de cada cenário."""
    perf_data = []
    for comp in results_data:
        for scenario, result in comp['results'].items():
            perf_data.append({
                'pergunta': comp['question_metadata']['id'],
                'cenario': scenario,
                'tempo_resposta': result.get('response_time', 0),
                'relevancia': result.get('relevance_score', 0)
            })
    
    if not perf_data:
        st.warning("Não há dados de performance para exibir.")
        return

    df = pd.DataFrame(perf_data)
    
    # Gráfico de Tempo de Resposta
    st.subheader("Tempo de Resposta por Pergunta (segundos)")
    fig_time = px.bar(df, x='pergunta', y='tempo_resposta', color='cenario', barmode='group',
                      labels={'tempo_resposta': 'Tempo (s)', 'pergunta': 'ID da Pergunta'})
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Gráfico de Relevância
    st.subheader("Score de Relevância por Pergunta")
    fig_rel = px.bar(df, x='pergunta', y='relevancia', color='cenario', barmode='group',
                     labels={'relevancia': 'Relevância', 'pergunta': 'ID da Pergunta'})
    st.plotly_chart(fig_rel, use_container_width=True)

def display_detailed_results(results_data):
    """Exibe os resultados detalhados de cada pergunta e cenário."""
    st.header("🔍 Análise Detalhada por Pergunta")
    
    for comp in results_data:
        q_meta = comp['question_metadata']
        with st.expander(f"**Pergunta {q_meta['id']}:** {q_meta['question']}"):
            
            cols = st.columns(3)
            scenarios = ['scenario_a', 'scenario_b', 'scenario_c']
            titles = ['A: Vector RAG', 'B: Hybrid RAG (Kùzu)', 'C: LLM-Only']
            
            for col, scenario_key, title in zip(cols, scenarios, titles):
                with col:
                    st.subheader(title)
                    result = comp['results'].get(scenario_key, {})
                    
                    if 'answer' in result:
                        st.metric("Tempo (s)", f"{result.get('response_time', 0):.4f}")
                        st.metric("Relevância", f"{result.get('relevance_score', 0):.2f}")
                        st.text_area("Resposta", result['answer'], height=250, key=f"ans_{q_meta['id']}_{scenario_key}")
                        
                        if 'retrieved_events' in result:
                             st.caption(f"Eventos recuperados: {len(result['retrieved_events'])}")
                        if 'retrieved_chunks' in result:
                             st.caption(f"Chunks recuperados: {len(result['retrieved_chunks'])}")
                    else:
                        st.error("Falha ao processar a resposta.")

def main():
    """Função principal do dashboard."""
    st.title("📊 Dashboard de Comparação de Cenários RAG")
    st.markdown("Análise de performance entre **Vector RAG**, **Hybrid RAG com Kùzu** e **LLM-Only**.")

    results = load_results()
    
    if results:
        st.header("📈 Resumo da Performance")
        create_summary_chart(results)
        display_detailed_results(results)
    else:
        st.info("Aguardando a execução do script 'run_integrated_mvp.py' para gerar os dados.")

if __name__ == "__main__":
    main()
    