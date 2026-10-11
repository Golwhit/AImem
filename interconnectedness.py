# Interconnectedness - A system for modeling and visualizing philosophical
# interconnectedness cycles between entities.
from typing import List, Dict
import networkx as nx
import hashlib


class RelationalNetworkRegistry:
    """Registry for storing and managing relational networks."""

    def __init__(self):
        self.cycles: Dict[str, List[str]] = {}

    def add_cycle(self, entities: List[str], auto_generate_id=True) -> str:
        """Add a cycle of interconnected entities to the registry.

        Args:
            entities: List of entity names forming a cycle.
            auto_generate_id: If True, generate unique ID for this cycle; 
                            if False, raise ValueError when no ID provided.
        Returns:
            The unique identifier assigned to this cycle."""
        # Generate a unique ID using timestamp and hash
        import time
        timestamp = int(time.time() * 10**6) % (10**12)
        hash_value = hashlib.md5(str(entities).encode()).hexdigest()
        cycle_id = f"cycle_{hash_value}"
        self.cycles[cycle_id] = entities
        return cycle_id

    def get_cycle(self, cycle_id: str) -> List[str]:
        """Retrieve the entity list for a given cycle ID."""
        if cycle_id not in self.cycles:
            raise KeyError(f'Cycle {cycle_id} not found')
        return self.cycles[cycle_id]

    def visualize_cycle(self, cycle_id: str) -> nx.Graph:
        """Visualize a single cycle as an undirected graph."""
        if cycle_id not in self.cycles:
            raise KeyError(f'Cycle {cycle_id} not found')
        G = nx.complete_graph(self.cycles[cycle_id])
        return G

    def visualize_cycles(self, cycle_ids: List[str], directed: bool = False) -> nx.Graph:
        """Visualize all cycles as a combined graph.

        Args:
            cycle_ids: List of cycle IDs to include in the visualization.
            directed: If True, create a directed graph; otherwise undirected.
        Returns:
            The resulting networkx Graph object."""
        G = nx.DiGraph() if directed else nx.Graph()
        for cycle_id in cycle_ids:
            entities = self.get_cycle(cycle_id)
            G.add_edges_from([(e, e) for e in set(entities)])
            # Add edges between consecutive entities
            for i in range(len(entities)):
                G.add_edge(entities[i], entities[(i + 1) % len(entities)])
        return G

    def get_all_cycles(self) -> List[str]:
        """Get all cycle IDs stored in the registry."""
        return list(self.cycles.keys())
