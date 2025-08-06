"""
Pipeline de ingestão de dados para eventos sonoros
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ..core.events import DynamicEventUnit

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """
    Pipeline para ingestão e processamento de dados de eventos sonoros
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapeamento de colunas esperadas
        self.column_mapping = {
            'timestamp': ['timestamp', 'time', 'datetime', 'date_time'],
            'event_type': ['event_type', 'type', 'event', 'sound_type'],
            'loudness': ['loudness', 'volume', 'decibels', 'db', 'intensity'],
            'sensor_id': ['sensor_id', 'sensor', 'device_id', 'source'],
            'description': ['description', 'desc', 'notes', 'comments']
        }
        
        logger.info(f"DataIngestionPipeline initialized with data_dir: {data_dir}")
    
    def process_csv(self, csv_path: str, encoding: str = 'utf-8') -> List[DynamicEventUnit]:
        """
        Processa arquivo CSV e retorna lista de eventos
        
        Args:
            csv_path: Caminho para o arquivo CSV
            encoding: Codificação do arquivo
            
        Returns:
            Lista de DynamicEventUnit
        """
        try:
            # Carrega CSV
            df = pd.read_csv(csv_path, encoding=encoding)
            logger.info(f"Loaded CSV with {len(df)} rows from {csv_path}")
            
            # Mapeia colunas
            df_mapped = self._map_columns(df)
            
            # Valida dados obrigatórios
            df_validated = self._validate_data(df_mapped)
            
            # Converte para eventos
            events = self._dataframe_to_events(df_validated)
            
            logger.info(f"Successfully processed {len(events)} events from CSV")
            return events
            
        except Exception as e:
            logger.error(f"Error processing CSV {csv_path}: {str(e)}")
            raise
    
    def process_json(self, json_path: str) -> List[DynamicEventUnit]:
        """
        Processa arquivo JSON e retorna lista de eventos
        """
        try:
            df = pd.read_json(json_path)
            logger.info(f"Loaded JSON with {len(df)} records from {json_path}")
            
            # Aplica mesmo processamento do CSV
            df_mapped = self._map_columns(df)
            df_validated = self._validate_data(df_mapped)
            events = self._dataframe_to_events(df_validated)
            
            logger.info(f"Successfully processed {len(events)} events from JSON")
            return events
            
        except Exception as e:
            logger.error(f"Error processing JSON {json_path}: {str(e)}")
            raise
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapeia colunas do DataFrame para nomes padrão"""
        df_mapped = df.copy()
        
        for standard_name, possible_names in self.column_mapping.items():
            for col_name in df.columns:
                if col_name.lower() in [name.lower() for name in possible_names]:
                    if standard_name not in df_mapped.columns:
                        df_mapped[standard_name] = df_mapped[col_name]
                    break
        
        return df_mapped
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida e limpa dados obrigatórios"""
        initial_count = len(df)
        
        # Remove linhas com campos obrigatórios nulos
        required_fields = ['timestamp', 'event_type', 'loudness']
        for field in required_fields:
            if field in df.columns:
                df = df.dropna(subset=[field])
        
        # Validação de tipos
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        
        if 'loudness' in df.columns:
            df['loudness'] = pd.to_numeric(df['loudness'], errors='coerce')
            df = df.dropna(subset=['loudness'])
            # Filtra valores de ruído válidos
            df = df[(df['loudness'] >= 0) & (df['loudness'] <= 150)]
        
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid records during validation")
        
        return df
    
    def _dataframe_to_events(self, df: pd.DataFrame) -> List[DynamicEventUnit]:
        """Converte DataFrame para lista de DynamicEventUnit"""
        events = []
        
        for idx, row in df.iterrows():
            try:
                # Campos obrigatórios
                event_data = {
                    'timestamp': row['timestamp'] if 'timestamp' in row else datetime.now(),
                    'event_type': str(row['event_type']) if 'event_type' in row else 'unknown',
                    'loudness': float(row['loudness']) if 'loudness' in row else 0.0,
                    'sensor_id': str(row['sensor_id']) if 'sensor_id' in row and pd.notna(row['sensor_id']) else f'sensor_{idx}'
                }
                
                # Campos opcionais
                if 'description' in row and pd.notna(row['description']):
                    event_data['description'] = str(row['description'])
                
                # Metadados adicionais
                metadata = {}
                for col in df.columns:
                    if col not in ['timestamp', 'event_type', 'loudness', 'sensor_id', 'description']:
                        if pd.notna(row[col]):
                            metadata[col] = row[col]
                
                if metadata:
                    event_data['metadata'] = metadata
                
                # Cria evento
                event = DynamicEventUnit(**event_data)
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Error creating event from row {idx}: {str(e)}")
                continue
        
        return events
    
    def save_processed_events(self, events: List[DynamicEventUnit], filename: str = "processed_events.json"):
        """Salva eventos processados em arquivo JSON"""
        output_path = self.data_dir.parent / "processed" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte eventos para dicionários
        events_data = [event.to_dict() for event in events]
        
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(events_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Saved {len(events)} processed events to {output_path}")
        return output_path
    
    def get_data_statistics(self, events: List[DynamicEventUnit]) -> Dict[str, Any]:
        """Gera estatísticas dos dados processados"""
        if not events:
            return {"error": "No events to analyze"}
        
        # Estatísticas básicas
        loudness_values = [e.loudness for e in events]
        timestamps = [e.timestamp for e in events]
        event_types = [e.event_type for e in events]
        
        stats = {
            "total_events": len(events),
            "time_range": {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "span_days": (max(timestamps) - min(timestamps)).days
            },
            "loudness_stats": {
                "min": min(loudness_values),
                "max": max(loudness_values),
                "mean": sum(loudness_values) / len(loudness_values),
                "violations": sum(1 for e in events if e.violates_noise_regulations())
            },
            "event_types": {
                "unique_types": len(set(event_types)),
                "distribution": {t: event_types.count(t) for t in set(event_types)}
            },
            "sensors": {
                "unique_sensors": len(set(e.sensor_id for e in events)),
                "sensor_distribution": {s: sum(1 for e in events if e.sensor_id == s) 
                                     for s in set(e.sensor_id for e in events)}
            }
        }
        
        return stats