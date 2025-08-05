from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import hashlib
import json

class DynamicEventUnit(BaseModel):
    """
    Unidade de Evento Dinâmico (DEU) conforme DyG-RAG.
    Implementa a estrutura básica para eventos temporais com âncoras precisas.
    """
    event_id: str
    timestamp: datetime
    event_type: str = Field(..., description="Tipo do evento sonoro (martelo, serra, betoneira, etc.)")
    loudness: float = Field(..., ge=0, le=150, description="Intensidade em decibéis")
    sensor_id: str = Field(..., description="Identificador do sensor que captou o evento")
    description: Optional[str] = Field(None, description="Descrição textual opcional do evento")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadados contextuais")
    
    # Campos derivados para análise temporal
    duration_seconds: Optional[float] = Field(None, description="Duração estimada do evento")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confiança na classificação")
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.event_id:
            self.event_id = self._generate_event_id()
    
    def _generate_event_id(self) -> str:
        """Gera ID único baseado no timestamp e sensor"""
        content = f"{self.timestamp.isoformat()}_{self.sensor_id}_{self.event_type}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_embedding_text(self) -> str:
        """
        Texto base para geração de embeddings semânticos.
        Combina informações relevantes para busca semântica.
        """
        base_text = f"{self.event_type}"
        
        if self.description:
            base_text += f" {self.description}"
            
        if self.metadata:
            # Adiciona contexto da fase da obra
            if 'phase' in self.metadata:
                base_text += f" fase {self.metadata['phase']}"
            
            # Adiciona localização
            if 'location' in self.metadata:
                base_text += f" local {self.metadata['location']}"
                
            # Adiciona equipamento se disponível
            if 'equipment' in self.metadata:
                base_text += f" equipamento {self.metadata['equipment']}"
        
        # Adiciona informação de intensidade categorizada
        loudness_category = self._categorize_loudness()
        base_text += f" intensidade {loudness_category}"
        
        return base_text.strip()
    
    def _categorize_loudness(self) -> str:
        """Categoriza a intensidade sonora para melhor busca semântica"""
        if self.loudness < 50:
            return "baixa"
        elif self.loudness < 70:
            return "moderada"
        elif self.loudness < 90:
            return "alta"
        else:
            return "muito_alta"
    
    def to_temporal_context(self) -> Dict[str, Any]:
        """
        Extrai contexto temporal para análise de padrões.
        Usado pelo TemporalGraphRAG para construir relacionamentos.
        """
        return {
            'hour': self.timestamp.hour,
            'day_of_week': self.timestamp.weekday(),
            'is_weekend': self.timestamp.weekday() >= 5,
            'is_work_hours': 7 <= self.timestamp.hour <= 18,
            'is_noise_restricted': self.timestamp.hour < 7 or self.timestamp.hour > 22,
            'month': self.timestamp.month,
            'year': self.timestamp.year
        }
    
    def violates_noise_regulations(self) -> bool:
        """
        Verifica se o evento viola regulamentações de ruído.
        Baseado em horários típicos de construção civil.
        """
        temporal_context = self.to_temporal_context()
        
        # Ruído em horário proibido
        if temporal_context['is_noise_restricted']:
            return True
            
        # Ruído muito alto mesmo em horário permitido
        if self.loudness > 85 and temporal_context['is_work_hours']:
            return True
            
        # Ruído moderado em fins de semana
        if temporal_context['is_weekend'] and self.loudness > 70:
            return True
            
        return False
    
    def get_severity_level(self) -> str:
        """Retorna nível de severidade do evento para priorização"""
        if self.violates_noise_regulations():
            if self.loudness > 90:
                return "CRÍTICO"
            else:
                return "ALTO"
        elif self.loudness > 80:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para armazenamento"""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'loudness': self.loudness,
            'sensor_id': self.sensor_id,
            'description': self.description,
            'metadata': self.metadata,
            'duration_seconds': self.duration_seconds,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DynamicEventUnit':
        """Cria instância a partir de dicionário"""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def __hash__(self):
        return hash(self.event_id)
    
    def __eq__(self, other):
        if not isinstance(other, DynamicEventUnit):
            return False
        return self.event_id == other.event_id