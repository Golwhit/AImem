#!/usr/bin/env python3
"""
Memory System Demo — Importance, Forgetting Curves & Connection Dynamics
=========================================================================
Visualizes:
  1. Ebbinghaus-style forgetting over activation cycles
  2. Hebbian link strengthening through repeated co-activation
  3. Consolidation threshold behavior
  4. Spreading activation / recall simulation
  5. Interference-based accelerated forgetting
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from memory_system import (
    MemoryGraph, MemoryNode, MemoryLink,
    ebbinghaus_curve, interference_forgetting, compute_relevance_score
)


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def bar(value: float, width: int = 40, char: str = "█") -> str:
    """Render a horizontal bar for importance/link strength visualization."""
    return f"{char * int(value * width):>{width}} {value:.3f}"


def demo_forgetting_curve():
    """Show how memory importance decays over activation cycles."""
    section("FORGETTING CURVE — Exponential Decay")
    g = MemoryGraph()
    nodes = []
    for name, imp in [("High Salience", 1.0), ("Medium", 0.6), ("Low Salience", 0.3)]:
        nodes.append(g.add_node(name, f"Memory: {name}", importance=imp))

    print(f"{'Cycle':>6}  |  {'High Salience':>14}  {'Medium':>10}  {'Low':>8}")
    print(f"-------  |  {'-'*14}  {'-'*10}  {'-'*8}")
    for cycle in range(0, 21, 2):
        if cycle > 0:
            g.decay_all(cycles=1)
        vals = [n.raw_importance for n in nodes]
        print(f"  {cycle:>4}  |  {bar(vals[0], 12):>16}  {bar(vals[1], 8):>12}  {bar(vals[2], 6):>10}")

    # Ebbinghaus formula comparison
    print(f"\n  Ebbinghaus reference (half_life=5.0 cycles):")
    for t in [0, 2, 5, 10, 20, 50]:
        print(f"    t={t:>3}  →  {ebbinghaus_curve(t, half_life=5.0):.4f}")


def demo_hebbian_strengthening():
    """Show co-activation strengthening links between nodes."""
    section("HEBBIAN LEARNING — Co-activation Strengthens Links")
    g = MemoryGraph()
    n1 = g.add_node("Concept A", "Frequently reviewed material", importance=0.8)
    n2 = g.add_node("Concept B", "Related concept", importance=0.7)
    link = g.connect(n1.id, n2.id, strength=0.3)

    print(f"  Initial link strength: {link.strength:.3f}")
    print(f"{'# activations':>12}  {'Link Strength':>14}  {'Bar'}")
    print(f"------------  {'-'*14}  {'-'*30}")
    for i in range(10):
        g.activate(n1.id, strength=0.9)
        g.activate(n2.id, strength=0.8)
        print(f"  {i+1:>10}  {link.strength:>14.3f}  {bar(link.strength, 25)}")

    # Now show decay on unused links
    print(f"\n  After 10 decay cycles (no activation):")
    g.decay_all(cycles=10)
    print(f"  Link strength: {link.strength:.3f}   {bar(link.strength, 25)}")


def demo_consolidation():
    """Demonstrate consolidation threshold behavior."""
    section("CONSOLIDATION — Hippocampal → Neocortical Transfer")
    g = MemoryGraph()
    for name, imp in [("Daily fact", 0.3), ("Repeated study", 0.7), ("Mastery topic", 1.0)]:
        n = g.add_node(name, f"Content: {name}", importance=imp)
        # Simulate some activations to boost
        for _ in range(int(imp * 5)):
            g.activate(n.id, strength=0.9)

    print(f"{'Node':>14}  {'Imp':>6}  {'Consol?':>8}  {'After Consolidate'}")
    print(f"{'-'*14}  {'-'*6}  {'-'*8}  {'-'*20}")
    for n in g.nodes.values():
        print(f"  {n.id:>14}  {n.importance:>6.3f}  {'✗' if not n.is_consolidated else '✓':>8}")

    g.consolidate_all()
    for n in g.nodes.values():
        print(f"  {n.id:>14}  {n.importance:>6.3f}  {'✓' if n.is_consolidated else '✗':>8}   ← {'consolidated (+20% imp)' if n.is_consolidated else 'below threshold'}")


def demo_spreading_activation():
    """Simulate recall via spreading activation."""
    section("SPREADING ACTIVATION — Recall Simulation")
    g = MemoryGraph()
    # Build a knowledge graph
    concepts = [
        ("python",      "Programming language",     0.9),
        ("oop",         "Object-oriented patterns",  0.8),
        ("decorators",   "Function wrapping tech",    0.7),
        ("data_struct",  "Lists, dicts, sets",       0.6),
        ("algorithms",   "Sorting/searching",         0.85),
        ("graphs",       "Node-edge structures",      0.75),
    ]
    nodes = [g.add_node(c[0], c[1], importance=c[2]) for c in concepts]

    # Add realistic connections
    edges = [
        ("python", "oop",         0.8),
        ("python", "decorators",   0.7),
        ("python", "data_struct", 0.6),
        ("oop",    "decorators",   0.9),
        ("algorithms", "graphs",   0.85),
        ("algorithms", "data_struct", 0.5),
        ("graphs",   "data_struct", 0.6),
    ]
    for s, t, w in edges:
        g.connect(s, t, strength=w)

    # Activate key nodes to simulate study
    g.activate("python", strength=1.0)
    g.activate("algorithms", strength=0.9)

    seed = "python"
    results = g.recall(seed, depth=2, threshold=0.05)
    print(f"  Seeding recall from '{seed}':")
    for nid, level in results:
        node = g.get_node(nid)
        print(f"    {nid:>14}  activation={level:.3f}   {bar(level, 20)}   [{node.content}]")


def demo_interference():
    """Show how competing cues accelerate forgetting."""
    section("INTERFERENCE FORGETTING — Cue Competition")
    print("  Comparing plain exponential decay vs interference-augmented decay:")
    base_imp = 0.8
    print(f"  {'Cycles':>6}  {'Plain Decay':>12}  {'+Interference':>14}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*14}")
    imp_plain = base_imp
    imp_interf = base_imp
    for c in range(10):
        decay_factor = 0.15 / max(imp_plain, 0.1)
        imp_plain *= max(0.0, 1.0 - decay_factor)
        interp_decay = 0.15 / max(imp_interf, 0.1) + 0.05 * 0.6  # overlap=0.6
        imp_interf *= max(0.0, 1.0 - interp_decay)
        print(f"    {c:>4}     {imp_plain:>10.4f}       {imp_interf:>12.4f}")


def demo_pruning():
    """Show forgotten memory pruning."""
    section("MEMORY PRUNING — Removing Forgotten Nodes")
    g = MemoryGraph()
    for name, imp in [("recent", 0.9), ("old fact", 0.2), ("vivid", 0.85), ("faded", 0.15)]:
        n = g.add_node(name, f"Content: {name}", importance=imp)
    g.connect("recent", "vivid", 0.5)
    g.connect("old fact", "faded", 0.3)

    print(f"  Before decay: {len(g.nodes)} nodes, {len(g.links)} links")
    for _ in range(15):
        g.decay_all(cycles=1)
    removed = g.prune_forgotten()
    print(f"  After 15 decay cycles + prune: {len(g.nodes)} nodes, {len(g.links)} links")
    print(f"  Removed (forgotten): {removed}")


def demo_relevance_score():
    """Show composite relevance scoring for recall prioritization."""
    section("RELEVANCE SCORING — Prioritizing What to Recall")
    g = MemoryGraph()
    n1 = g.add_node("math", "Mathematics", importance=0.9)
    n2 = g.add_node("cs",   "Computer Science", importance=0.7)
    n3 = g.add_node("hist",  "History",          importance=0.4)
    g.connect("math", "cs", strength=0.8)
    g.activate("math", strength=1.0)
    g.activate("cs", strength=0.9)

    print(f"  {'Node':>8}  {'Importance':>10}  {'Conn Weight':>12}  {'Relevance Score':>15}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*15}")
    for n in g.nodes.values():
        conns = []
        for link in g.links.values():
            if link.source_id == n.id or link.target_id == n.id:
                conns.append(link)
        score = compute_relevance_score(n, conns, recency=1.0)
        print(f"  {n.id:>8}  {n.importance:>10.3f}  {sum(c.strength for c in conns)/max(len(conns),1):>12.3f}  {score:>15.3f}")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AI MEMORY SYSTEM — Importance, Forgetting,       ║")
    print("║        Connection Strength & Consolidation Demo         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo_forgetting_curve()
    demo_hebbian_strengthening()
    demo_consolidation()
    demo_spreading_activation()
    demo_interference()
    demo_pruning()
    demo_relevance_score()

    # Final graph summary
    section("GRAPHSERIALIZATION EXAMPLE")
    g = MemoryGraph()
    n1 = g.add_node("ai", "Artificial Intelligence", importance=0.95)
    n2 = g.add_node("ml",  "Machine Learning",       importance=0.9)
    n3 = g.add_node("llm", "Large Language Models",  importance=0.85)
    g.connect("ai", "ml", strength=0.85)
    g.connect("ml", "llm", strength=0.9)
    for _ in range(5):
        g.activate("ai"); g.activate("ml"); g.activate("llm")
    import json
    print(json.dumps(g.to_dict(), indent=2))


if __name__ == "__main__":
    main()
