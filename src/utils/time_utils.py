import numpy as np
from datetime import datetime

class FourierTimeEncoder:
    """Codificador temporal baseado em Fourier"""
    def __init__(self, dim=64, max_period=365*24*3600):
        self.dim = dim
        self.max_period = max_period
        self.frequencies = 1 / np.power(10000, 2 * (np.arange(dim // 2) / dim))
        
    def encode(self, dt: datetime):
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
            
        t = dt.timestamp()
        embedding = np.zeros(self.dim)
        
        for i in range(self.dim // 2):
            omega = 2 * np.pi * self.frequencies[i]
            embedding[2*i] = np.sin(omega * t / self.max_period)
            embedding[2*i + 1] = np.cos(omega * t / self.max_period)
            
        return embedding

def normalize_timestamp(timestamp: datetime, min_time: datetime, max_time: datetime) -> float:
    """Normaliza timestamp para valor entre 0 e 1"""
    if min_time == max_time:
        return 0.5
    return (timestamp - min_time) / (max_time - min_time)