from typing import List
from datetime import datetime

def evaluate_retrieval(retrieved_events: List, relevant_events: List) -> dict:
    """Calcula métricas de avaliação para recuperação"""
    tp = len(set(retrieved_events) & set(relevant_events))
    precision = tp / len(retrieved_events) if retrieved_events else 0
    recall = tp / len(relevant_events) if relevant_events else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'retrieved': len(retrieved_events),
        'relevant': len(relevant_events)
    }

def evaluate_response_quality(response: str, ground_truth: str) -> dict:
    """Avalia qualidade da resposta gerada"""
    # Implementação simplificada - pode ser expandida
    response_words = set(response.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    overlap = len(response_words & truth_words)
    return {
        'word_overlap': overlap / len(truth_words) if truth_words else 0,
        'response_length': len(response.split())
    }