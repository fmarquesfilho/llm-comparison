import unittest
from datetime import datetime, timedelta
from ..core.events import DynamicEventUnit
from ..core.temporal_rag import TemporalGraphRAG
from ..core.vector_rag import TemporalVectorRAG

class TestTemporalRAG(unittest.TestCase):
    def setUp(self):
        self.rag = TemporalGraphRAG(None)
        self.events = [
            DynamicEventUnit(
                event_id=f"event_{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                event_type=f"type_{i % 3}",
                loudness=50 + i,
                sensor_id=f"sensor_{i % 2}"
            ) for i in range(10)
        ]
        
    def test_event_addition(self):
        for event in self.events:
            self.rag.add_event(event)
        self.assertEqual(len(self.rag.graph.nodes), 10)

class TestVectorRAG(unittest.TestCase):
    def setUp(self):
        self.rag = TemporalVectorRAG(None)
        # Similar setup as above
        
    def test_retrieval(self):
        # Test retrieval logic
        pass