from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class DEU:
    id_evento: str
    timestamp: datetime
    tipo_som: str
    loudness: float
    id_sensor: str
    texto: Optional[str] = ""
    metadados: Optional[Dict] = None

    def to_dict(self):
        return {
            "id_evento": self.id_evento,
            "timestamp": self.timestamp.isoformat(),
            "tipo_som": self.tipo_som,
            "loudness": self.loudness,
            "id_sensor": self.id_sensor,
            "texto": self.texto,
            "metadados": self.metadados or {},
        }
        
    def build_deu_from_row(row):
        return {
            "event_id": row["event_id"],
            "anchor_time": row["anchor_time"],
            "type": row["type"],
            "description": row["description"],
            "related_entities": row["related_entities"].split(",")
            if row["related_entities"]
            else [],
        }

    def print_deu(deu):
        print("[DEU]", deu["event_id"], "(", deu["anchor_time"], ")", deu["description"])
