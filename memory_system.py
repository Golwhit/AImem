"""
AI Memory System v2
====================
Implements:
  - Importance weighting (Salience / Value / Consolidation)
  - Forgetting / decay curves (exponential + interference-based)
  - Connection strength between memory nodes (Hebbian strengthening, consolidation, pruning)

Design inspired by:
  - ACT-R declarative memory model (Anderson & Lebiere)
  - Ebbinghaus forgetting curve
  - Hebbian learning ("cells that fire together wire together")
  - Consolidation theory (Hippocampus → Neocortex transfer)
  - BDI agent architectures (Rao & Georgeff, Shoham)
"""

from typing import Dict, List, Optional, Tuple, Set
import math
import time
import hashlib
import json
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants / Defaults
# ---------------------------------------------------------------------------
DEFAULT_DECAY_RATE = 0.15          # per activation cycle
DEFAULT_INTERFERENCE_WEIGHT = 0.05 # overlap-induced decay boost
DEFAULT_HEBBIAN_STRENGTH = 0.1     # co-activation boosts connection
DEFAULT_CONSOLIDATION_THRESHOLD = 0.7
DEFAULT_IMPORTANCE_DECAY_FACTOR = 0.98


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class MemoryNode:
    """A single memory unit with importance, decay state, and connections."""

    id: str
    content: str
    timestamp: float = field(default_factory=time.time)
    kind: str = "message"

    # --- Importance (Salience) ------------------------------------------
    raw_importance: float = 1.0          # user / system-assigned base value
    activation_history: List[float] = field(default_factory=list)
    _consolidated: bool = False

    @property
    def importance(self) -> float:
        """Current effective importance — decays with inactivity."""
        if not self.activation_history:
            return self.raw_importance
        # Recent activations weigh more (recency boost)
        weights = [2 ** (i / max(len(self.activation_history), 1))
                   for i in range(len(self.activation_history))]
        total_w = sum(weights)
        weighted_avg = sum(a * w / total_w for a, w in zip(self.activation_history, weights))
        return self.raw_importance * weighted_avg

    @property
    def is_consolidated(self) -> bool:
        return self._consolidated

    def consolidate(self):
        """Mark as consolidated (hippocampal → neocortical transfer)."""
        self._consolidated = True
        self.raw_importance *= 1.2  # consolidation bump

    def activate(self, strength: float = 1.0):
        """Register an activation event — drives recall & strengthens links."""
        self.activation_history.append(strength)
        # Keep a rolling window to avoid unbounded growth
        if len(self.activation_history) > 50:
            self.activation_history = self.activation_history[-50:]

    def decay(self, cycles: int = 1) -> float:
        """Apply exponential forgetting. Returns remaining importance."""
        for _ in range(cycles):
            # Standard exponential decay — higher importance decays slower
            decay_factor = DEFAULT_DECAY_RATE / max(self.importance, 0.1)
            self.raw_importance *= max(0.0, 1.0 - decay_factor)
            # Interference: more connections → faster forgetting (cue competition)
            if hasattr(self, '_connections'):
                overlap = sum(c.strength for c in self._connections.values())
                self.raw_importance *= max(0.0, 1.0 - DEFAULT_INTERFERENCE_WEIGHT * overlap)
        return self.raw_importance

    def forget_threshold(self) -> bool:
        """Returns True if memory has decayed below recall threshold."""
        return self.raw_importance < 0.05

    def __repr__(self):
        status = "consolidated" if self._consolidated else "unconsolidated"
        return f"MemoryNode({self.id!r}, imp={self.importance:.3f}, {status})"


@dataclass
class MemoryLink:
    """A weighted edge between two MemoryNodes — Hebbian connection."""

    source_id: str
    target_id: str
    strength: float = 0.0
    last_accessed: float = field(default_factory=time.time)

    def strengthen(self, amount: float = DEFAULT_HEBBIAN_STRENGTH):
        """Hebbian co-activation strengthens the link."""
        self.strength = min(1.0, self.strength + amount)
        self.last_accessed = time.time()

    def weaken(self, amount: float = 0.02):
        """Synaptic depression — unused links fade."""
        self.strength = max(0.0, self.strength - amount)

    def decay(self, cycles: int = 1):
        """Unused connections drift toward zero (use-it-or-lose-it)."""
        for _ in range(cycles):
            if self.strength > 0.01:
                self.strength *= 0.95  # slow drift
            else:
                self.strength = 0.0

    def __repr__(self):
        return f"MemoryLink({self.source_id}->{self.target_id}, w={self.strength:.3f})"

# ---------------------------------------------------------------------------
# Memory Graph — the core data structure
# ---------------------------------------------------------------------------
class MemoryGraph:
    """
    A graph of interconnected memories with forgetting, importance dynamics,
    and Hebbian connection strengthening/weakening.

    Usage:
        g = MemoryGraph()
        n1 = g.add_node("hello_world", "Learned Python basics")
        n2 = g.add_node("loops",     "For/while loop patterns")
        g.connect(n1.id, n2.id)          # establish link
        g.activate(n1.id, strength=0.8)  # recall / review
        g.decay_all(cycles=5)            # simulate forgetting over time
    """

    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.links: Dict[Tuple[str, str], MemoryLink] = {}
        self._adjacency: Dict[str, Set[str]] = {}

    # ---- Node management ------------------------------------------------
    def add_node(
        self,
        node_id: str,
        content: str,
        importance: float = 1.0,
        kind: str = "message",
    ) -> MemoryNode:
        """Create and store a new memory node."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} already exists")
        node = MemoryNode(id=node_id, content=content, raw_importance=importance, kind=kind)
        self.nodes[node_id] = node
        self._adjacency.setdefault(node_id, set())
        return node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    # ---- Link management (Hebbian) --------------------------------------
    def connect(self, src: str, tgt: str, strength: float = 0.5) -> MemoryLink:
        """Create or update a directed link between two nodes."""
        key = (src, tgt)
        if src not in self.nodes or tgt not in self.nodes:
            raise KeyError("Both nodes must exist")
        link = MemoryLink(source_id=src, target_id=tgt, strength=strength)
        self.links[key] = link
        self._adjacency.setdefault(src, set()).add(tgt)
        self._adjacency.setdefault(tgt, set()).add(src)
        return link

    def get_link(self, src: str, tgt: str) -> Optional[MemoryLink]:
        return self.links.get((src, tgt))

    def remove_link(self, src: str, tgt: str):
        """Prune a weak or obsolete connection."""
        key = (src, tgt)
        if key in self.links:
            del self.links[key]
            self._adjacency.setdefault(src, set()).discard(tgt)
            self._adjacency.setdefault(tgt, set()).discard(src)

    # ---- Activation / Recall --------------------------------------------
    def activate(self, node_id: str, strength: float = 1.0):
        """
        Activate a node — drives recall probability and triggers Hebbian
        strengthening of all inbound/outbound links (spreading activation).
        """
        node = self.nodes.get(node_id)
        if not node:
            raise KeyError(f"Node {node_id!r} not found")
        node.activate(strength)

        # Strengthen adjacent links (Hebbian: fire together → wire together)
        for neighbor in list(self._adjacency.get(node_id, set())):
            link = self.links.get((node_id, neighbor)) or self.links.get((neighbor, node_id))
            if link:
                link.strengthen(strength * 0.5)

    # ---- Forgetting / Decay ---------------------------------------------
    def decay_all(self, cycles: int = 1):
        """Apply forgetting to every node and weaken unused links."""
        for node in self.nodes.values():
            node.decay(cycles)
        for link in self.links.values():
            link.decay(cycles)

    def prune_forgotten(self, threshold: float = 0.05) -> List[str]:
        """Remove nodes below recall threshold and their incident links."""
        to_remove = [nid for nid, n in self.nodes.items() if n.forget_threshold()]
        for nid in to_remove:
            neighbors = list(self._adjacency.get(nid, set()))
            for nb in neighbors:
                self.remove_link(nid, nb)
            del self.nodes[nid]
            self._adjacency.pop(nid, None)
        return to_remove

    def consolidate_all(self):
        """Consolidate all unconsolidated nodes (hippocampal → neocortical)."""
        for node in self.nodes.values():
            if not node.is_consolidated and node.importance >= DEFAULT_CONSOLIDATION_THRESHOLD:
                node.consolidate()

    # ---- Spreading Activation / Recall Simulation -----------------------
    def recall(self, seed_id: str, depth: int = 2, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Simulate spreading activation from a seed node. Returns list of
        (node_id, activation_level) for all reachable nodes above threshold.
        """
        if seed_id not in self.nodes:
            raise KeyError(f"Seed node {seed_id!r} not found")

        activated: Dict[str, float] = {seed_id: 1.0}
        frontier = [(seed_id, 1.0)]
        visited = {seed_id}

        while frontier and depth > 0:
            next_frontier = []
            for nid, level in frontier:
                node = self.nodes[nid]
                # Propagate to neighbors weighted by link strength & node importance
                for neighbor in self._adjacency.get(nid, set()):
                    link = self.links.get((nid, neighbor)) or self.links.get((neighbor, nid))
                    if not link:
                        continue
                    new_level = level * link.strength * (node.importance / max(node.raw_importance, 0.01))
                    if neighbor not in visited and new_level > threshold:
                        visited.add(neighbor)
                        activated[neighbor] = new_level
                        next_frontier.append((neighbor, new_level))
            frontier = next_frontier
            depth -= 1

        return sorted(activated.items(), key=lambda x: -x[1])

    # ---- Network Statistics ---------------------------------------------
    def stats(self) -> Dict:
        """Return summary statistics about the memory graph."""
        importances = [n.importance for n in self.nodes.values()]
        link_strengths = [l.strength for l in self.links.values()]
        return {
            "node_count": len(self.nodes),
            "link_count": len(self.links),
            "avg_importance": sum(importances) / len(importances) if importances else 0,
            "max_importance": max(importances) if importances else 0,
            "avg_link_strength": sum(link_strengths) / len(link_strengths) if link_strengths else 0,
            "consolidated_count": sum(1 for n in self.nodes.values() if n.is_consolidated),
            "forgotten_count": sum(1 for n in self.nodes.values() if n.forget_threshold()),
        }

    def to_dict(self) -> Dict:
        """Serialize the graph to a JSON-friendly dict."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "importance": round(n.importance, 4),
                    "consolidated": n.is_consolidated,
                    "timestamp": n.timestamp,
                    "kind": n.kind,
                }
                for n in self.nodes.values()
            ],
            "links": [
                {
                    "source": l.source_id,
                    "target": l.target_id,
                    "strength": round(l.strength, 4),
                }
                for l in self.links.values()
            ],
            "stats": self.stats(),
        }

    def __repr__(self):
        return f"MemoryGraph(nodes={len(self.nodes)}, links={len(self.links)}, {self.stats()})"


# ---------------------------------------------------------------------------
# Forgetting Curve Utilities
# ---------------------------------------------------------------------------
def ebbinghaus_curve(t: float, half_life: float = 1.0) -> float:
    """Classic Ebbinghaus exponential forgetting curve."""
    return math.exp(-t / half_life)


def interference_forgetting(base_importance: float, overlap_score: float, cycles: int = 1) -> float:
    """
    Forgetting with cue competition — memories that share many cues decay
    faster (similar to sibling interference in ACT-R).
    """
    decay = DEFAULT_DECAY_RATE / max(base_importance, 0.1)
    decay += DEFAULT_INTERFERENCE_WEIGHT * overlap_score
    return base_importance * ((1.0 - decay) ** cycles)


def compute_relevance_score(node: MemoryNode, connections: List[MemoryLink], recency: float = 1.0) -> float:
    """
    Composite relevance score used for prioritizing recall / consolidation.
    Combines importance, connection density, and recency.
    """
    conn_weight = sum(c.strength for c in connections) / max(len(connections), 1)
    activation_avg = (sum(node.activation_history) / len(node.activation_history)) if node.activation_history else 0
    return node.importance * (0.5 + 0.3 * conn_weight + 0.2 * activation_avg) * recency
