# src/evaluation/cost_analysis.py
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class CostAnalyzer:
    def __init__(self):
        """Inicializa analisador de custos otimizado para diferentes arquiteturas"""
        
        # Custos base em USD (valores aproximados de 2024)
        self.costs = {
            'hardware': {
                # Cloud Computing (por hora)
                'aws_t3_medium': 0.0416,      # CPU: 2 vCPUs, 4GB RAM
                'aws_t3_large': 0.0832,       # CPU: 2 vCPUs, 8GB RAM
                'aws_g4dn_xlarge': 0.526,     # GPU: NVIDIA T4, 4 vCPUs, 16GB RAM
                'aws_g5_xlarge': 1.006,       # GPU: NVIDIA A10G, 4 vCPUs, 16GB RAM
                'aws_p3_2xlarge': 3.06,       # GPU: NVIDIA V100, 8 vCPUs, 61GB RAM
                
                # Google Cloud (por hora)
                'gcp_n1_standard_2': 0.095,   # CPU: 2 vCPUs, 7.5GB RAM
                'gcp_n1_standard_4': 0.190,   # CPU: 4 vCPUs, 15GB RAM
                'gcp_t4_gpu': 0.35,           # GPU: NVIDIA T4
                'gcp_v100_gpu': 2.48,         # GPU: NVIDIA V100
                
                # Local/Mac Mini M1 equivalents (depreciation per hour)
                'mac_mini_m1': 0.05,          # Amortização Mac Mini M1 (3 anos)
                'mac_studio_m1_ultra': 0.15,  # Amortização Mac Studio M1 Ultra
                
                # Storage (por GB/mês)
                'storage_ssd': 0.10,          # SSD storage
                'storage_object': 0.023,      # Object storage (S3/GCS)
                'bandwidth_gb': 0.09          # Bandwidth por GB
            },
            
            'apis_external': {
                # OpenAI (por 1K tokens)
                'openai_gpt4_turbo': 0.01,
                'openai_gpt4': 0.03,
                'openai_gpt35_turbo': 0.0015,
                
                # Anthropic Claude (por 1K tokens)
                'anthropic_claude_3_opus': 0.015,
                'anthropic_claude_3_sonnet': 0.003,
                'anthropic_claude_3_haiku': 0.00025,
                
                # Google Gemini (por 1K tokens)
                'google_gemini_pro': 0.00035,
                'google_gemini_ultra': 0.01,
                
                # Cohere (por 1K tokens)
                'cohere_command': 0.0015,
                'cohere_command_r_plus': 0.003
            },
            
            'development_tools': {
                # Ferramentas de desenvolvimento (por mês)
                'github_copilot': 10,
                'wandb_team': 50,
                'huggingface_pro': 20,
                'colab_pro': 12,
                'monitoring_basic': 30,
                'ci_cd_pipeline': 25
            },
            
            'operational': {
                # Custos operacionais (por mês)
                'domain_ssl': 15,
                'backup_service': 10,
                'monitoring_alerts': 20,
                'logging_service': 35,
                'security_scanning': 25,
                'team_communication': 8  # Slack, etc.
            }
        }
        
        # Configurações de arquiteturas
        self.architecture_configs = {
            'rag_simple': {
                'description': 'RAG com embeddings + modelo local',
                'requires_training': False,
                'requires_gpu': False,  # Pode rodar em CPU
                'model_size': 'small',
                'complexity': 'low'
            },
            'rag_optimized': {
                'description': 'RAG otimizado com reranking',
                'requires_training': False,
                'requires_gpu': True,
                'model_size': 'medium',
                'complexity': 'medium'
            },
            'fine_tuned': {
                'description': 'Modelo fine-tunado com LoRA/DoRA', 
                'requires_training': True,
                'requires_gpu': True,
                'model_size': 'large',
                'complexity': 'high'
            },
            'hybrid': {
                'description': 'RAG + Fine-tuning combinados',
                'requires_training': True,
                'requires_gpu': True,
                'model_size': 'large',
                'complexity': 'high'
            },
            'api_external': {
                'description': 'APIs externas (OpenAI, Claude, etc)',
                'requires_training': False,
                'requires_gpu': False,
                'model_size': 'external',
                'complexity': 'low'
            }
        }
    
    def calculate_hardware_costs(self, architecture: str, environment: str = "local", 
                               hours_per_month: int = 730) -> Dict[str, float]:
        """
        Calcula custos de hardware/infraestrutura
        
        Args:
            architecture: Nome da arquitetura
            environment: 'local', 'cloud_aws', 'cloud_gcp', 'hybrid'
            hours_per_month: Horas de operação por mês
            
        Returns:
            Dict com breakdown de custos de hardware
        """
        config = self.architecture_configs.get(architecture, {})
        costs = {}
        
        if environment == "local":
            # Custos para Mac Mini M1 / desenvolvimento local
            if config.get('requires_gpu', False):
                # Para modelos que precisam de GPU, assume Mac Studio ou similar
                costs['compute'] = self.costs['hardware']['mac_studio_m1_ultra'] * hours_per_month
            else:
                # Para CPU apenas, Mac Mini M1 é suficiente
                costs['compute'] = self.costs['hardware']['mac_mini_m1'] * hours_per_month
            
            # Storage local (SSD)
            storage_gb = 100 if config.get('model_size') == 'large' else 50
            costs['storage'] = storage_gb * self.costs['hardware']['storage_ssd']
            
            # Bandwidth mínimo
            costs['bandwidth'] = 10 * self.costs['hardware']['bandwidth_gb']  # 10GB/mês
            
        elif environment == "cloud_aws":
            # Custos AWS baseados na arquitetura
            if config.get('requires_gpu', False):
                if config.get('model_size') == 'large':
                    costs['compute'] = self.costs['hardware']['aws_g5_xlarge'] * hours_per_month
                else:
                    costs['compute'] = self.costs['hardware']['aws_g4dn_xlarge'] * hours_per_month
            else:
                costs['compute'] = self.costs['hardware']['aws_t3_large'] * hours_per_month
            
            # Storage baseado no tamanho do modelo
            storage_gb = {'small': 20, 'medium': 50, 'large': 100, 'external': 10}
            costs['storage'] = storage_gb.get(config.get('model_size', 'small'), 50) * self.costs['hardware']['storage_object']
            
            # Bandwidth estimado
            costs['bandwidth'] = 50 * self.costs['hardware']['bandwidth_gb']
            
        elif environment == "cloud_gcp":
            # Custos Google Cloud
            if config.get('requires_gpu', False):
                compute_cost = self.costs['hardware']['gcp_n1_standard_4'] * hours_per_month
                if config.get('model_size') == 'large':
                    compute_cost += self.costs['hardware']['gcp_v100_gpu'] * hours_per_month
                else:
                    compute_cost += self.costs['hardware']['gcp_t4_gpu'] * hours_per_month
                costs['compute'] = compute_cost
            else:
                costs['compute'] = self.costs['hardware']['gcp_n1_standard_2'] * hours_per_month
            
            # Storage similar ao AWS
            storage_gb = {'small': 20, 'medium': 50, 'large': 100, 'external': 10}
            costs['storage'] = storage_gb.get(config.get('model_size', 'small'), 50) * self.costs['hardware']['storage_object']
            costs['bandwidth'] = 50 * self.costs['hardware']['bandwidth_gb']
        
        # APIs externas
        if architecture == 'api_external':
            costs['compute'] = 0  # Sem custos de compute próprio
            costs['storage'] = 5 * self.costs['hardware']['storage_object']  # Armazenamento mínimo
            costs['bandwidth'] = 20 * self.costs['hardware']['bandwidth_gb']
        
        return costs
    
    def calculate_training_costs(self, architecture: str, environment: str = "local",
                               training_hours: int = 8, model_size: str = "7b") -> Dict[str, float]:
        """
        Calcula custos de treinamento/fine-tuning
        
        Args:
            architecture: Nome da arquitetura
            environment: Ambiente de treinamento
            training_hours: Horas de treinamento
            model_size: Tamanho do modelo (7b, 13b, 70b)
            
        Returns:
            Dict com custos de treinamento
        """
        config = self.architecture_configs.get(architecture, {})
        costs = {}
        
        if not config.get('requires_training', False):
            # Sem treinamento necessário
            return {'training_compute': 0, 'training_storage': 0, 'data_preparation': 0}
        
        # Multiplier baseado no tamanho do modelo
        size_multipliers = {'7b': 1.0, '13b': 1.8, '70b': 8.0}
        multiplier = size_multipliers.get(model_size, 1.0)
        
        if environment == "local":
            # Treinamento local com M1/M2
            if model_size in ['7b', '13b']:
                # M1 pode treinar modelos menores com quantização
                base_cost = self.costs['hardware']['mac_mini_m1'] * training_hours
            else:
                # Modelos grandes precisam de Mac Studio ou similar
                base_cost = self.costs['hardware']['mac_studio_m1_ultra'] * training_hours
            
            costs['training_compute'] = base_cost * multiplier
            
        elif environment in ["cloud_aws", "cloud_gcp"]:
            # Treinamento em cloud
            if model_size == '7b':
                hourly_cost = self.costs['hardware']['aws_g4dn_xlarge']
            elif model_size == '13b':
                hourly_cost = self.costs['hardware']['aws_g5_xlarge']
            else:  # 70b
                hourly_cost = self.costs['hardware']['aws_p3_2xlarge']
            
            costs['training_compute'] = hourly_cost * training_hours * multiplier
        
        # Storage para checkpoints e modelos treinados
        storage_gb = {'7b': 15, '13b': 30, '70b': 150}
        costs['training_storage'] = storage_gb.get(model_size, 15) * self.costs['hardware']['storage_ssd']
        
        # Preparação de dados (custo fixo)
        costs['data_preparation'] = 10  # Assumindo preparação manual/semi-automática
        
        return costs
    
    def calculate_inference_costs(self, architecture: str, queries_per_day: int = 1000,
                                avg_tokens_per_response: int = 200, 
                                days_per_month: int = 30) -> Dict[str, float]:
        """
        Calcula custos de inferência/operação
        
        Args:
            architecture: Nome da arquitetura
            queries_per_day: Consultas por dia
            avg_tokens_per_response: Tokens médios por resposta
            days_per_month: Dias por mês
            
        Returns:
            Dict com custos de inferência
        """
        monthly_queries = queries_per_day * days_per_month
        monthly_tokens = monthly_queries * avg_tokens_per_response
        
        costs = {}
        
        if architecture == 'api_external':
            # Uso de APIs externas
            # Assume GPT-3.5 Turbo como padrão
            api_cost_per_1k_tokens = self.costs['apis_external']['openai_gpt35_turbo']
            costs['api_calls'] = (monthly_tokens / 1000) * api_cost_per_1k_tokens
            costs['hosting'] = 0  # Sem hospedagem própria
            
        else:
            # Hospedagem própria
            config = self.architecture_configs.get(architecture, {})
            
            # Custo base de hospedagem (assumindo 24/7)
            hardware_costs = self.calculate_hardware_costs(architecture, "local", 730)
            costs['hosting'] = sum(hardware_costs.values())
            
            # Ajuste baseado no volume de consultas
            volume_multiplier = min(2.0, 1.0 + (queries_per_day / 10000))  # Máximo 2x para alto volume
            costs['hosting'] *= volume_multiplier
            
            costs['api_calls'] = 0  # Sem custos de API externa
        
        # Custos fixos mensais
        costs['monitoring'] = self.costs['operational']['monitoring_alerts']
        costs['backup'] = self.costs['operational']['backup_service']
        costs['security'] = self.costs['operational']['security_scanning']
        
        return costs
    
    def calculate_development_costs(self, architecture: str, development_months: int = 3) -> Dict[str, float]:
        """
        Calcula custos de desenvolvimento e ferramentas
        
        Args:
            architecture: Nome da arquitetura
            development_months: Meses de desenvolvimento
            
        Returns:
            Dict com custos de desenvolvimento
        """
        config = self.architecture_configs.get(architecture, {})
        costs = {}
        
        # Ferramentas básicas (por mês)
        monthly_tools = 0
        
        if config.get('complexity') in ['medium', 'high']:
            monthly_tools += self.costs['development_tools']['wandb_team']  # Tracking experimentos
            monthly_tools += self.costs['development_tools']['huggingface_pro']  # Modelos
        
        if config.get('requires_training'):
            monthly_tools += self.costs['development_tools']['colab_pro']  # Backup para treinamento
            
        monthly_tools += self.costs['development_tools']['github_copilot']  # Produtividade
        monthly_tools += self.costs['development_tools']['monitoring_basic']  # Monitoramento
        
        costs['development_tools'] = monthly_tools * development_months
        
        # Custo de desenvolvimento (tempo de desenvolvedor)
        complexity_hours = {'low': 40, 'medium': 80, 'high': 160}
        dev_hours = complexity_hours.get(config.get('complexity', 'medium'), 80)
        
        # Assumindo $50/hora para desenvolvedor (freelance/consultoria)
        costs['developer_time'] = dev_hours * 50
        
        return costs
    
    def calculate_total_costs(self, architecture: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula custos totais para uma arquitetura em um cenário específico
        
        Args:
            architecture: Nome da arquitetura
            scenario: Dict com parâmetros do cenário
                     (queries_per_day, environment, model_size, etc.)
            
        Returns:
            Dict com breakdown completo de custos
        """
        queries_per_day = scenario.get('queries_per_day', 1000)
        environment = scenario.get('environment', 'local')
        model_size = scenario.get('model_size', '7b')
        development_months = scenario.get('development_months', 3)
        months_analysis = scenario.get('months', 12)
        
        # Custos uma vez (setup)
        training_costs = self.calculate_training_costs(architecture, environment, model_size=model_size)
        setup_cost = sum(training_costs.values())
        
        development_costs = self.calculate_development_costs(architecture, development_months)
        setup_cost += sum(development_costs.values())
        
        # Custos mensais (operação)
        monthly_costs = self.calculate_inference_costs(
            architecture, queries_per_day, 
            avg_tokens_per_response=scenario.get('avg_tokens', 200)
        )
        monthly_cost = sum(monthly_costs.values())
        
        # Custos totais
        total_cost_period = setup_cost + (monthly_cost * months_analysis)
        cost_per_query = total_cost_period / (queries_per_day * 365) if queries_per_day > 0 else 0
        
        return {
            'architecture': architecture,
            'scenario': scenario.get('name', 'default'),
            'setup_cost': setup_cost,
            'monthly_cost': monthly_cost,
            'total_cost_period': total_cost_period,
            'cost_per_query': cost_per_query,
            'cost_per_1k_queries': cost_per_query * 1000,
            'months_analyzed': months_analysis,
            'queries_per_day': queries_per_day,
            'breakdown': {
                'training': training_costs,
                'development': development_costs,
                'monthly_operation': monthly_costs
            },
            'roi_metrics': self._calculate_roi_metrics(setup_cost, monthly_cost, queries_per_day)
        }
    
    def _calculate_roi_metrics(self, setup_cost: float, monthly_cost: float, 
                              queries_per_day: int) -> Dict[str, float]:
        """Calcula métricas de ROI"""
        
        # Payback period em meses (assumindo valor de $0.10 por query)
        value_per_query = 0.10
        monthly_value = queries_per_day * 30 * value_per_query
        monthly_profit = monthly_value - monthly_cost
        
        payback_months = setup_cost / monthly_profit if monthly_profit > 0 else float('inf')
        
        # ROI anual
        annual_profit = monthly_profit * 12
        roi_annual = (annual_profit / setup_cost) * 100 if setup_cost > 0 else 0
        
        return {
            'payback_months': min(payback_months, 999),  # Cap para evitar valores infinitos
            'roi_annual_percent': roi_annual,
            'monthly_profit': monthly_profit,
            'break_even_queries_per_day': (monthly_cost / (value_per_query * 30)) if value_per_query > 0 else 0
        }
    
    def compare_architectures(self, scenarios: List[Dict]) -> pd.DataFrame:
        """
        Compara custos entre arquiteturas para diferentes cenários
        
        Args:
            scenarios: Lista de cenários para analisar
            
        Returns:
            DataFrame com comparação completa
        """
        results = []
        
        architectures = ['rag_simple', 'rag_optimized', 'fine_tuned', 'hybrid', 'api_external']
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            
            for arch in architectures:
                try:
                    costs = self.calculate_total_costs(arch, scenario)
                    
                    # Flatten para DataFrame
                    result_row = {
                        'scenario': scenario_name,
                        'architecture': arch,
                        'queries_per_day': costs['queries_per_day'],
                        'setup_cost_usd': costs['setup_cost'],
                        'monthly_cost_usd': costs['monthly_cost'],
                        'total_cost_12m_usd': costs['total_cost_period'],
                        'cost_per_query_usd': costs['cost_per_query'],
                        'cost_per_1k_queries_usd': costs['cost_per_1k_queries'],
                        'payback_months': costs['roi_metrics']['payback_months'],
                        'roi_annual_percent': costs['roi_metrics']['roi_annual_percent'],
                        'break_even_queries_day': costs['roi_metrics']['break_even_queries_per_day']
                    }
                    
                    results.append(result_row)
                    
                except Exception as e:
                    logger.error(f"Erro ao calcular custos para {arch} em {scenario_name}: {e}")
        
        df = pd.DataFrame(results)
        return df
    
    def generate_cost_report(self, output_path: Path = None, 
                           custom_scenarios: List[Dict] = None) -> pd.DataFrame:
        """
        Gera relatório completo de custos com cenários padrão
        
        Args:
            output_path: Caminho para salvar CSV
            custom_scenarios: Cenários customizados
            
        Returns:
            DataFrame com análise completa
        """
        
        # Cenários padrão
        default_scenarios = [
            {
                'name': 'Piloto_Local',
                'queries_per_day': 100,
                'environment': 'local',
                'model_size': '7b',
                'avg_tokens': 150,
                'months': 6
            },
            {
                'name': 'Producao_Pequena',
                'queries_per_day': 1000,
                'environment': 'local',
                'model_size': '7b',
                'avg_tokens': 200,
                'months': 12
            },
            {
                'name': 'Producao_Media',
                'queries_per_day': 5000,
                'environment': 'cloud_aws',
                'model_size': '7b',
                'avg_tokens': 200,
                'months': 12
            },
            {
                'name': 'Producao_Grande',
                'queries_per_day': 20000,
                'environment': 'cloud_aws',
                'model_size': '13b',
                'avg_tokens': 250,
                'months': 12
            },
            {
                'name': 'Enterprise',
                'queries_per_day': 50000,
                'environment': 'cloud_aws',
                'model_size': '13b',
                'avg_tokens': 300,
                'months': 24
            }
        ]
        
        scenarios = custom_scenarios if custom_scenarios else default_scenarios
        
        logger.info(f"Gerando relatório de custos para {len(scenarios)} cenários...")
        
        df = self.compare_architectures(scenarios)
        
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Relatório de custos salvo em {output_path}")
        
        # Log sumário
        self._log_cost_summary(df)
        
        return df
    
    def _log_cost_summary(self, df: pd.DataFrame):
        """Log sumário dos custos"""
        if df.empty:
            logger.warning("DataFrame de custos vazio")
            return
        
        logger.info("📊 SUMÁRIO DE CUSTOS:")
        
        for scenario in df['scenario'].unique():
            scenario_df = df[df['scenario'] == scenario]
            logger.info(f"\n🎯 Cenário: {scenario}")
            
            # Melhor custo-benefício
            best_cost = scenario_df.loc[scenario_df['total_cost_12m_usd'].idxmin()]
            best_roi = scenario_df.loc[scenario_df['roi_annual_percent'].idxmax()]
            
            logger.info(f"   💰 Menor custo: {best_cost['architecture']} (${best_cost['total_cost_12m_usd']:.0f}/ano)")
            logger.info(f"   📈 Melhor ROI: {best_roi['architecture']} ({best_roi['roi_annual_percent']:.1f}%)")
            
            # APIs vs Self-hosted
            api_costs = scenario_df[scenario_df['architecture'] == 'api_external']
            self_hosted = scenario_df[scenario_df['architecture'] != 'api_external']
            
            if not api_costs.empty and not self_hosted.empty:
                api_cost = api_costs['total_cost_12m_usd'].iloc[0]
                min_self_cost = self_hosted['total_cost_12m_usd'].min()
                
                if api_cost < min_self_cost:
                    savings = min_self_cost - api_cost
                    logger.info(f"   🤖 APIs externas são {savings:.0f}$ mais baratas que self-hosted")
                else:
                    savings = api_cost - min_self_cost
                    logger.info(f"   🏠 Self-hosted é {savings:.0f}$ mais barato que APIs externas")
    
    def get_architecture_recommendations(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera recomendações baseadas em cenário específico
        
        Args:
            scenario: Parâmetros do cenário
            
        Returns:
            Dict com recomendações
        """
        queries_per_day = scenario.get('queries_per_day', 1000)
        budget_monthly = scenario.get('budget_monthly', 1000)
        technical_expertise = scenario.get('technical_expertise', 'medium')  # low, medium, high
        
        recommendations = {
            'primary_recommendation': None,
            'alternative': None,
            'reasoning': [],
            'cost_comparison': None
        }
        
        # Calcula custos para todas arquiteturas neste cenário
        df = self.compare_architectures([scenario])
        
        if df.empty:
            recommendations['reasoning'].append("Erro ao calcular custos")
            return recommendations
        
        # Filtra por orçamento
        within_budget = df[df['monthly_cost_usd'] <= budget_monthly]
        
        if within_budget.empty:
            recommendations['reasoning'].append(f"⚠️ Nenhuma arquitetura dentro do orçamento de ${budget_monthly}/mês")
            # Recomenda a mais barata
            cheapest = df.loc[df['monthly_cost_usd'].idxmin()]
            recommendations['primary_recommendation'] = cheapest['architecture']
            recommendations['reasoning'].append(f"Recomenda-se {cheapest['architecture']} (mais barata: ${cheapest['monthly_cost_usd']:.0f}/mês)")
        else:
            # Recomendação baseada em volume e expertise
            if queries_per_day <= 500:
                # Baixo volume - prioriza simplicidade
                if technical_expertise == 'low':
                    recommendations['primary_recommendation'] = 'api_external'
                    recommendations['reasoning'].append("Volume baixo + baixa expertise técnica → APIs externas")
                else:
                    recommendations['primary_recommendation'] = 'rag_simple'
                    recommendations['reasoning'].append("Volume baixo → RAG simples e econômico")
                    
            elif queries_per_day <= 5000:
                # Volume médio - balanceia custo e performance
                best_value = within_budget.loc[within_budget['roi_annual_percent'].idxmax()]
                recommendations['primary_recommendation'] = best_value['architecture']
                recommendations['reasoning'].append(f"Volume médio → {best_value['architecture']} (melhor ROI: {best_value['roi_annual_percent']:.1f}%)")
                
            else:
                # Alto volume - prioriza eficiência
                if 'fine_tuned' in within_budget['architecture'].values:
                    recommendations['primary_recommendation'] = 'fine_tuned'
                    recommendations['reasoning'].append("Alto volume → Fine-tuning para eficiência")
                else:
                    best_cost_per_query = within_budget.loc[within_budget['cost_per_query_usd'].idxmin()]
                    recommendations['primary_recommendation'] = best_cost_per_query['architecture']
                    recommendations['reasoning'].append("Alto volume → Menor custo por query")
        
        # Alternativa
        remaining = df[df['architecture'] != recommendations['primary_recommendation']]
        if not remaining.empty:
            alternative = remaining.loc[remaining['total_cost_12m_usd'].idxmin()]
            recommendations['alternative'] = alternative['architecture']
        
        recommendations['cost_comparison'] = df.to_dict('records')
        
        return recommendations
    