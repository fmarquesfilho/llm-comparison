from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel

class DynamicEventUnit(BaseModel):
    """Unidade de Evento Dinâmico (DEU) conforme DyG-RAG"""
    event_id: str
    timestamp: datetime
    event_type: str
    loudness: float
    sensor_id: str
    description: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_embedding_text(self) -> str:
        """Texto base para geração de embeddings"""
        base_text = f"{self.event_type} {self.description or ''}"
        if self.metadata:
            base_text += f" {self.metadata.get('phase', '')} {self.metadata.get('location', '')}"
        return base_text.strip()