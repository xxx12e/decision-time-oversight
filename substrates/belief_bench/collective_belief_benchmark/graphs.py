from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx
import numpy as np


def _ensure_connected(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    if graph.number_of_nodes() <= 1 or nx.is_connected(graph):
        return graph
    components = [list(component) for component in nx.connected_components(graph)]
    for left, right in zip(components, components[1:]):
        graph.add_edge(int(rng.choice(left)), int(rng.choice(right)))
    return graph


def _cluster_source_groups(n_agents: int, clusters: int) -> dict[int, int]:
    clusters = max(1, clusters)
    return {node: min(clusters - 1, node * clusters // n_agents) for node in range(n_agents)}


def _degree_source_groups(graph: nx.Graph, groups: int) -> dict[int, int]:
    groups = max(1, groups)
    nodes_by_degree = sorted(graph.nodes, key=lambda node: (graph.degree[node], node))
    result: dict[int, int] = {}
    for rank, node in enumerate(nodes_by_degree):
        result[int(node)] = min(groups - 1, rank * groups // max(1, graph.number_of_nodes()))
    return result


def build_graph(
    n_agents: int,
    topology: str,
    graph_config: Mapping[str, Any],
    seed: int,
) -> tuple[nx.Graph, dict[int, int]]:
    rng = np.random.default_rng(seed)
    topology = topology.lower()
    target_degree = int(graph_config.get("target_degree", 8))
    source_groups = int(graph_config.get("source_groups", 4))

    if topology == "fully_mixed":
        graph = nx.complete_graph(n_agents)
        groups = _cluster_source_groups(n_agents, source_groups)
        return graph, groups

    if topology == "random":
        edge_prob = float(graph_config.get("random_edge_prob", target_degree / max(1, n_agents - 1)))
        graph = nx.fast_gnp_random_graph(n_agents, edge_prob, seed=seed)
        graph = _ensure_connected(graph, rng)
        return graph, _degree_source_groups(graph, source_groups)

    if topology == "small_world":
        k = min(n_agents - 1, max(2, target_degree))
        if k % 2:
            k -= 1
        k = max(2, k)
        rewire = float(graph_config.get("small_world_rewire_prob", 0.18))
        graph = nx.watts_strogatz_graph(n_agents, k, rewire, seed=seed)
        graph = _ensure_connected(graph, rng)
        return graph, _cluster_source_groups(n_agents, source_groups)

    if topology == "hub":
        attachment = min(n_agents - 1, max(1, int(graph_config.get("hub_attachment", target_degree // 2))))
        graph = nx.barabasi_albert_graph(n_agents, attachment, seed=seed)
        graph = _ensure_connected(graph, rng)
        return graph, _degree_source_groups(graph, source_groups)

    if topology == "clustered":
        clusters = max(1, int(graph_config.get("clusters", source_groups)))
        base = n_agents // clusters
        sizes = [base for _ in range(clusters)]
        for index in range(n_agents - sum(sizes)):
            sizes[index % clusters] += 1
        intra = float(graph_config.get("clustered_intra_prob", 0.42))
        inter = float(graph_config.get("clustered_inter_prob", 0.04))
        probs = [[intra if i == j else inter for j in range(clusters)] for i in range(clusters)]
        graph = nx.stochastic_block_model(sizes, probs, seed=seed)
        graph = nx.Graph(graph)
        graph = _ensure_connected(graph, rng)
        groups = {}
        start = 0
        for group, size in enumerate(sizes):
            for node in range(start, start + size):
                groups[node] = group
            start += size
        return graph, groups

    raise ValueError(f"Unknown graph topology: {topology}")
