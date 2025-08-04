from shared.deu import DEU
from shared.kuzu_utils import KuzuConnector
from datetime import datetime
import json

# 1. Carrega os DEUs
with open("data/deus.json", "r") as f:
    deus_data = json.load(f)

deus = [DEU(**d) for d in deus_data]

# 2. Conecta ao Kuzu e cria esquema
kz = KuzuConnector()
kz.create_schema()

# 3. Insere nós
for deu in deus:
    kz.insert_deu(deu)

# 4. Insere arestas por proximidade temporal (60s)
for i in range(len(deus)):
    for j in range(i + 1, len(deus)):
        delta = abs((deus[i].timestamp - deus[j].timestamp).total_seconds())
        if delta < 60:
            peso = 1.0 / (delta + 1e-6)
            kz.insert_edge(deus[i].id_evento, deus[j].id_evento, peso)
