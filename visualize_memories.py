import networkx as nx
from interconnectedness import RelationalNetworkRegistry
registry = RelationalNetworkRegistry()
memories = [24, 25, 26, 27]
cycle_id_0 = registry.add_cycle(memories)
graph = registry.visualize_cycle(cycle_id_0)
print(f"Nodes: {list(graph.nodes())}")
print(f"Edges: {list(graph.edges())}")