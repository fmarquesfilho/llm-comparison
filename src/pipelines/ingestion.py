import pandas as pd
from typing import Union
from pathlib import Path
from ..core.events import DynamicEventUnit
from ..core.temporal_rag import TemporalGraphRAG
from ..core.vector_rag import TemporalVectorRAG

class DataIngestionPipeline:
    def __init__(self, graph_rag: TemporalGraphRAG, vector_rag: TemporalVectorRAG):
        self.graph_rag = graph_rag
        self.vector_rag = vector_rag
    
    def ingest_from_dataframe(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            event = DynamicEventUnit(
                event_id=row['id'],
                timestamp=row['timestamp'],
                event_type=row['event_type'],
                loudness=row['loudness'],
                sensor_id=row['sensor_id'],
                description=row.get('description'),
                metadata={
                    'phase': row.get('phase'),
                    'location': row.get('location')
                }
            )
            self.graph_rag.add_event(event)
            self.vector_rag.add_event(event)
    
    def ingest_from_csv(self, file_path: Union[str, Path]):
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        self.ingest_from_dataframe(df)