import kuzu
import os

class KuzuConnector:
    def __init__(self, db_path="kuzu_db"):
        os.makedirs(db_path, exist_ok=True)
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def create_schema(self):
        self.conn.execute("CREATE NODE TABLE Event(id_evento STRING PRIMARY KEY, timestamp DATETIME, tipo_som STRING, loudness FLOAT, id_sensor STRING, texto STRING);")
        self.conn.execute("CREATE REL TABLE RelatesTo(FROM Event TO Event, peso DOUBLE);")

    def insert_deu(self, deu):
        self.conn.execute(f"""
            INSERT INTO Event VALUES (
                "{deu.id_evento}",
                "{deu.timestamp.isoformat()}",
                "{deu.tipo_som}",
                {deu.loudness},
                "{deu.id_sensor}",
                "{deu.texto or ''}"
            );
        """)

    def insert_edge(self, id1, id2, peso):
        self.conn.execute(f"""
            INSERT INTO RelatesTo VALUES ("{id1}", "{id2}", {peso});
        """)

class KuzuInterface:
    def __init__(self, db_path="graph_kuzu"):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def query(self, cypher):
        result = self.conn.execute(cypher)
        return [dict(row) for row in result]

    def close(self):
        self.conn.close()
        