"""
Memory Data Cubes
=================
Extends MemoryGraph with multi-dimensional aggregation (a "data cube" over memories).

Dimensions:
  - time_bucket   : day / week / month bins of timestamp
  - importance_bin: low / mid / high based on importance score
  - consolidated  : True / False flag
  - tag           : note tags (multi-valued)

Operations (OLAP-style):
  - slice   : fix one dimension to a value, return sub-cube
  - dice    : filter multiple dimensions simultaneously
  - roll_up : aggregate a dimension upward (day -> week -> month)
  - drill_down: split a bucket into finer granularity
  - pivot   : swap two dimensions for tabular output

Inspired by:
  - Obsidian Dataview plugin (natural-language queries over note metadata)
  - Multidimensional OLAP cubes (slice/dice/roll-up/pivot)
  - Your existing MemoryGraph + RelationalNetworkRegistry
"""
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import math
import networkx as nx


def bin_time(timestamp: float, granularity: str = "day") -> str:
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    if granularity == "day":
        return dt.strftime("%Y-%m-%d")
    elif granularity == "week":
        start = dt - datetime.timedelta(days=dt.weekday())
        return start.strftime("%Y-W%W")
    elif granularity == "month":
        return dt.strftime("%Y-%m")
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def bin_importance(score: float) -> str:
    if score < 0.3:
        return "low"
    elif score < 0.7:
        return "mid"
    else:
        return "high"


@dataclass
class CubeCell:
    key: Tuple[str, ...]
    nodes: List[str] = field(default_factory=list)
    total_importance: float = 0.0
    link_count: int = 0

    @property
    def density(self) -> float:
        return self.total_importance / max(len(self.nodes), 1)

class MemoryDataCube:
    """
    Multi-dimensional aggregation over a MemoryGraph.

    Dimensions (configurable at construction):
      - time_bucket   : granularity for timestamp bins ("day", "week", "month")
      - importance_bin: bucket thresholds (auto low/mid/high)
      - consolidated  : boolean split
      - tag           : note tags, split per unique tag value
    """

    def __init__(self, memory_graph, time_granularity: str = "week"):
        self.graph = memory_graph
        self.time_granularity = time_granularity
        self.cells: Dict[Tuple[str, ...], CubeCell] = {}
        self.dimension_names = ["time_bucket", "importance_bin", "consolidated", "tag"]
        self._build()

    def _build(self):
        for node in self.graph.nodes.values():
            key_parts = [
                bin_time(node.timestamp, self.time_granularity),
                bin_importance(node.importance),
                str(node.is_consolidated).lower(),
            ]
            tags = getattr(node, "tags", []) or []
            if not tags:
                tags = ["untagged"]
            for tag in tags:
                key = (*key_parts, str(tag))
                if key not in self.cells:
                    self.cells[key] = CubeCell(key=key)
                cell = self.cells[key]
                if node.id not in cell.nodes:
                    cell.nodes.append(node.id)
                cell.total_importance += node.importance
            # Also add an "all-tags" summary cell
            key_all = (*key_parts, "_all_tags_")
            if key_all not in self.cells:
                self.cells[key_all] = CubeCell(key=key_all)
            cell_all = self.cells[key_all]
            for n_id in node.id if node.id not in cell_all.nodes else []:
                cell_all.nodes.append(n_id)
            cell_all.total_importance += node.importance

        # Compute link counts per cell (bidirectional)
        for link in self.graph.links.values():
            src_key = self._node_to_key(link.source_id)
            tgt_key = self._node_to_key(link.target_id)
            for k in (src_key, tgt_key):
                if k and k in self.cells:
                    self.cells[k].link_count += 1



def nl_query_over_graph(graph, query: str, granularity: str = "week") -> Dict:
    cube = MemoryDataCube(graph, time_granularity=granularity)
    sub = cube.query_nl(query)
    return {
        "subcube": sub,
        "summary": sub.summarize(),
        "pivot": sub.visualize_pivot_table("time_bucket", "importance_bin"),
        "graph": sub.visualize_slice("importance_bin", sub._most_common_key_dim(0)),
    }


if __name__ == "__main__":
    from memory_system import MemoryGraph, MemoryNode, MemoryLink

    mg = MemoryGraph()
    notes = [
        ("mem_1", "Hebbian learning fundamentals", 1704067200, 0.9, True, ["neuroscience"]),
        ("mem_2", "Ebbinghaus forgetting curve derivation", 1704153600, 0.75, False, ["psychology"]),
        ("mem_3", "ACT-R declarative model overview", 1704326400, 0.85, True, ["cognitive_model"]),
        ("mem_4", "Obsidian graph view mechanics", 1704412800, 0.6, False, ["tools"]),
        ("mem_5", "Data cube OLAP operations deep dive", 1704499200, 0.95, True, ["data_engineering"]),
        ("mem_6", "Natural language query parsing", 1704585600, 0.55, False, ["nlp"]),
    ]
    for nid, content, ts, imp, cons, tags in notes:
        node = MemoryNode(id=nid, content=content, timestamp=ts)
        node.raw_importance = imp
        node._consolidated = cons
        node.tags = tags
        mg.add_node(node)

    mg.add_link("mem_1", "mem_3", strength=0.8)
    mg.add_link("mem_2", "mem_6", strength=0.5)
    mg.add_link("mem_4", "mem_5", strength=0.6)
    mg.add_link("mem_1", "mem_2", strength=0.3)

    print("=" * 60)
    print("Memory Data Cube - Full Demo")
    print("=" * 60)

    cube = MemoryDataCube(mg, time_granularity="day")
    print("\n[Summary]")
    for k, v in cube.summarize().items():
        print(f"  {k}: {v}")

    print("\n[Pivot: time_bucket x importance_bin]")
    print(cube.visualize_pivot_table("time_bucket", "importance_bin"))

    print("\n[NL Query: 'high importance"]")
    sub = cube.query_nl("high importance")
    print(f"  Cells matched: {len(sub.cells)}")
    for k, c in sub.cells.items():
        print(f"    {k} -> {len(c.nodes)} nodes, density={c.density:.3f}")

    print("\n[NL Query: 'unconsolidated"]")
    sub2 = cube.query_nl("unconsolidated")
    for k, c in sub2.cells.items():
        print(f"    {k} -> {len(c.nodes)} nodes")

    print("\n[Pivot after roll-up (week->month)]")
    rolled = cube.roll_up("time_bucket")
    print(rolled.visualize_pivot_table("time_bucket", "importance_bin"))

    print("\n[Visualizing slice: importance_bin=high]")
    G = cube.visualize_slice("importance_bin", "high")
    print(f"  Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

    print("\n[Demonstration complete.]")
            matrix[row] = {}
            matrix[row][col] = cell
        return matrix

    # ---- Natural-Language Query ------------------------------------------
    def query_nl(self, natural_language: str) -> "MemoryDataCube":
        text = natural_language.lower().strip()
        filters: Dict[str, str] = {}

        if "high" in text and "low" not in text and "mid" not in text:
            filters["importance_bin"] = "high"
        elif "mid" in text:
            filters["importance_bin"] = "mid"
        elif "low" in text:
            filters["importance_bin"] = "low"

        if "consolidated" in text and "un" not in text:
            filters["consolidated"] = "true"
        elif "unconsolidated" in text:
            filters["consolidated"] = "false"

        import datetime, re
        now = datetime.datetime.now()
        if "this week" in text:
            start = now - datetime.timedelta(days=now.weekday())
            filters["time_bucket"] = start.strftime("%Y-W%W")
        elif "last week" in text:
            start = now - datetime.timedelta(days=now.weekday() + 7)
            filters["time_bucket"] = start.strftime("%Y-W%W")
        elif "this month" in text:
            filters["time_bucket"] = now.strftime("%Y-%m")
        elif "last month" in text:
            last = now.replace(day=1) - datetime.timedelta(days=1)
            filters["time_bucket"] = last.strftime("%Y-%m")

        tag_match = re.search(r"(?:tag:#?)\s*(\w+)", text)
        if tag_match:
            filters["tag"] = tag_match.group(1)

        return self.dice(filters) if filters else self

    def visualize_slice(self, dimension: str, value: str) -> nx.Graph:
        sub = self.slice(dimension, value)
        G = nx.Graph()
        for cell in sub.cells.values():
            for n_id in cell.nodes:
                G.add_node(n_id, importance=self._node_importance(n_id))
            for i, a in enumerate(cell.nodes):
                for b in cell.nodes[i + 1:]:
                    G.add_edge(a, b, weight=0.3)
        return G

    def _node_importance(self, node_id: str) -> float:
        for node in self.graph.nodes.values():
            if node.id == node_id:
                return node.importance
        return 0.0

    def _most_common_key_dim(self, dim_idx: int) -> str:
        counts: Dict[str, int] = {}
        for key in self.cells.keys():
            v = key[dim_idx]
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    def visualize_pivot_table(self, swap_a: str, swap_b: str) -> str:
        matrix = self.pivot(swap_a, swap_b)
        rows = sorted(matrix.keys())
        cols_set = set()
        for row in matrix.values():
            cols_set.update(row.keys())
        cols = sorted(cols_set)
        header = "          | " + " | ".join(str(c) for c in cols) + " | total"
        sep = "----------+" + "+".join("-" * max(len(str(c)) for c in cols) + "+")
        lines = [header, sep]
        grand_total = 0
        for row in rows:
            vals = []
            row_sum = 0
            for col in cols:
                cell = matrix.get(row, {}).get(col)
                val_str = f"{cell.density:.2f}" if cell else "-"
                vals.append(val_str)
                row_sum += cell.total_importance if cell else 0
            line = f"  {row:<9} | " + " | ".join(vals) + f" | {row_sum:.2f}"
            lines.append(line)
            grand_total += row_sum
        lines.append(sep)
        lines.append(f"{'TOTAL':<11} | {'':>25} | {grand_total:.2f}")
        return "\n".join(lines)

    def summarize(self) -> Dict:
        by_time = {}
        by_imp = {"low": 0, "mid": 0, "high": 0}
        by_con = {"true": 0, "false": 0}
        for cell in self.cells.values():
            t, imp, con = cell.key[0], cell.key[1], cell.key[2]
            by_time[cell.key[0]] = by_time.get(cell.key[0], 0) + len(cell.nodes)
            by_imp[imp] = by_imp.get(imp, 0) + len(cell.nodes)
            by_con[con] = by_con.get(con, 0) + len(cell.nodes)
        return {
            "total_cells": len(self.cells),
            "total_nodes_indexed": sum(len(c.nodes) for c in self.cells.values()),
            "by_time_bucket": by_time,
            "by_importance": by_imp,
            "by_consolidated": by_con,
            "time_granularity": self.time_granularity,
        }

    def _from_cells(self, cells: Dict[Tuple[str, ...], CubeCell]) -> "MemoryDataCube":
        stub = MemoryDataCube.__new__(MemoryDataCube)
        stub.graph = self.graph
        stub.time_granularity = self.time_granularity
        stub.cells = cells
        stub.dimension_names = self.dimension_names
        return stub

    def _timestamp_from_key(self, key_time: str) -> float:
        import datetime
        try:
            if "-" in key_time and "W" not in key_time:
                return datetime.datetime.strptime(key_time, "%Y-%m").replace(day=1).timestamp()
            elif "W" in key_time:
                yr, w = key_time.split("W")
                return datetime.datetime.fromisocalendar(int(yr), int(w), 1).timestamp()
        except Exception:
            pass
        return time.time()

    def __repr__(self):
        s = self.summarize()
        return f"MemoryDataCube({s['total_cells']} cells, {s['total_nodes_indexed']} node-entries, gran={s['time_granularity']})"
