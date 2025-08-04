import numpy as np

def fourier_time_embedding(timestamp, period=86400):
    t_sec = timestamp.hour*3600 + timestamp.minute*60 + timestamp.second
    return np.array([np.sin(2*np.pi*t_sec/period), np.cos(2*np.pi*t_sec/period)])
