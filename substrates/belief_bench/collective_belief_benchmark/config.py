from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260601,
    "model_type": "scripted",
    "n_agents": 32,
    "rounds": 8,
    "trials": 2,
    "consensus_threshold": 0.8,
    "minority_ratios": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "evidence_strengths": [0.58, 0.66, 0.74],
    "memory_lengths": [1, 3, 6],
    "graph_topologies": ["random", "small_world", "hub", "clustered", "fully_mixed"],
    "defenses": [
        "no_defense",
        "majority_vote",
        "confidence_weighted_update",
        "prompt_only_caution",
        "evidence_anchoring",
        "source_diversity_constraint",
        "memory_decay",
        "unsupported_claim_quarantine",
        "combined_inoculation_policy",
    ],
    "defense_strengths": [0.0, 0.5, 1.0],
    "graph": {
        "target_degree": 8,
        "random_edge_prob": 0.13,
        "small_world_rewire_prob": 0.18,
        "hub_attachment": 4,
        "clusters": 4,
        "clustered_intra_prob": 0.42,
        "clustered_inter_prob": 0.04,
        "source_groups": 4,
    },
    "committed": {
        "selection": "high_degree",
        "confidence": 0.99,
        "logodds": 4.0,
    },
    "update": {
        "evidence_anchor": 0.12,
        "social_weight": 2.1,
        "max_logodds": 6.0,
        "memory_decay": 0.55,
        "confidence_cap": 0.95,
    },
    "local_llm": {
        "model_path": None,
        "max_new_tokens": 96,
        "temperature": 0.0,
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = deep_update(config, loaded)
    return config


def save_config_snapshot(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
