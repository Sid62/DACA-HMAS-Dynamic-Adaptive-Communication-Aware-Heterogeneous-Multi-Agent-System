"""Coalition feasibility functions (Eqs 23-27)."""

from __future__ import annotations

import numpy as np


def pairwise_feasibility(
    dist_ij: float,
    cqi_ij: float,
    c1: float,
) -> float:
    """Eq 23: Psi_ij(t) = [dist <= C1] * CQI_ij.
    For dist > C1 up to multi-hop range (100m), multi-hop relay factor preserves link feasibility.
    """
    max_range = max(c1 * 2.0, 100.0)
    if dist_ij > max_range:
        return 0.0
    link_factor = max(1.0 - (dist_ij / max_range), 0.2) if dist_ij > c1 else 1.0
    return float(cqi_ij * link_factor)


def build_psi_matrix(
    distance_matrix: np.ndarray,
    cqi_matrix: np.ndarray,
    c1: float,
) -> np.ndarray:
    n = distance_matrix.shape[0]
    psi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                psi[i, j] = pairwise_feasibility(
                    distance_matrix[i, j], cqi_matrix[i, j], c1
                )
            else:
                psi[i, j] = 1.0
    return psi


def coalition_feasibility_score(
    member_indices: list[int],
    psi_matrix: np.ndarray,
) -> float:
    """Eq 24: Gamma_k(t) = min-pair over coalition members."""
    if len(member_indices) <= 1:
        return 1.0
    min_psi = 1.0
    for i in member_indices:
        for j in member_indices:
            if i < j:
                min_psi = min(min_psi, psi_matrix[i, j])
    return float(min_psi)


def coalition_feasibility_rate(
    coalitions: list[list[int]],
    psi_matrix: np.ndarray,
    gamma_min: float,
) -> float:
    """Eqs 26-27: CFR(t)."""
    if not coalitions:
        return 1.0
    feasible = sum(
        1
        for members in coalitions
        if coalition_feasibility_score(members, psi_matrix) >= gamma_min
    )
    return feasible / len(coalitions)


def validate_coalition_members(
    member_ids: list[str],
    agent_id_to_idx: dict[str, int],
    psi_matrix: np.ndarray,
    gamma_min: float,
) -> bool:
    indices = [agent_id_to_idx[mid] for mid in member_ids if mid in agent_id_to_idx]
    return coalition_feasibility_score(indices, psi_matrix) >= gamma_min
