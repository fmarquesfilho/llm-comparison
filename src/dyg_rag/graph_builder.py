import networkx as nx

def build_event_graph(deus, window_seconds=300):
    G = nx.Graph()
    for i, e1 in enumerate(deus):
        G.add_node(i, **e1)
        for j in range(i+1, len(deus)):
            e2 = deus[j]
            delta_t = abs((e1['timestamp'] - e2['timestamp']).total_seconds())
            context_ok = e1['metadata'].get("fase_obra") == e2['metadata'].get("fase_obra")
            if delta_t<=window_seconds or context_ok:
                weight = 1/(delta_t+1)
                G.add_edge(i, j, weight=weight)
    return G
