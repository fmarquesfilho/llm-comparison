from datetime import datetime

class Event:
    def __init__(self, timestamp, tipo, loudness, sensor_id, desc="", metadata=None):
        self.timestamp = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp)
        self.tipo = tipo
        self.loudness = loudness
        self.sensor_id = sensor_id
        self.desc = desc
        self.metadata = metadata or {}

    def to_dict(self):
        return self.__dict__
