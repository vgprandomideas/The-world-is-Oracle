"""
Portfolio analytics for event probability swaps.
"""

from dataclasses import dataclass

import numpy as np

from oracle.models import SwapContract


@dataclass
class PortfolioPosition:
    position_id: str
    event_id: str
    notional: float
    fixed_probability: float
    long_floating: bool = True
    label: str = ""

    def contract(self):
        return SwapContract(
            contract_id=self.position_id,
            event_id=self.event_id,
            notional=self.notional,
            fixed_probability=self.fixed_probability,
        )


def _submatrix(correlation_matrix, event_ids):
    size = len(event_ids)
    matrix = np.eye(size)
    if not event_ids:
        return matrix

    if hasattr(correlation_matrix, "is_positive_definite") and not correlation_matrix.is_positive_definite():
        correlation_matrix.make_positive_definite()

    source_events = getattr(correlation_matrix, "_events", [])
    source_matrix = getattr(correlation_matrix, "_matrix", np.eye(len(source_events)))

    for i, event_a in enumerate(event_ids):
        for j, event_b in enumerate(event_ids):
            if i == j:
                matrix[i, j] = 1.0
            elif event_a in source_events and event_b in source_events:
                a_idx = source_events.index(event_a)
                b_idx = source_events.index(event_b)
                matrix[i, j] = float(source_matrix[a_idx, b_idx])
    return matrix


def summarize_positions(positions, current_probabilities):
    rows = []
    totals = {"current_mtm": 0.0, "margin": 0.0, "gross_notional": 0.0}

    for position in positions:
        current_p = current_probabilities.get(position.event_id, 0.5)
        direction = 1.0 if position.long_floating else -1.0
        mtm = (current_p - position.fixed_probability) * position.notional * direction
        margin = position.contract().total_margin
        row = {
            "position_id": position.position_id,
            "label": position.label or position.position_id,
            "event_id": position.event_id,
            "direction": "Long floating" if position.long_floating else "Short floating",
            "current_probability": round(current_p, 4),
            "fixed_probability": round(position.fixed_probability, 4),
            "notional": round(position.notional, 2),
            "current_mtm": round(mtm, 2),
            "margin": round(margin, 2),
        }
        rows.append(row)
        totals["current_mtm"] += mtm
        totals["margin"] += margin
        totals["gross_notional"] += position.notional

    totals = {key: round(value, 2) for key, value in totals.items()}
    return {"positions": rows, "totals": totals}


def analyze_portfolio_risk(
    positions,
    current_probabilities,
    daily_vols,
    correlation_matrix,
    n_paths=5000,
    n_days=30,
):
    if not positions:
        return {
            "summary": summarize_positions([], current_probabilities),
            "risk": None,
            "histogram": [],
            "tail_contributors": [],
        }

    unique_events = []
    for position in positions:
        if position.event_id not in unique_events:
            unique_events.append(position.event_id)

    corr = _submatrix(correlation_matrix, unique_events)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    scaling = np.sqrt(np.diag(corr))
    corr = corr / np.outer(scaling, scaling)
    chol = np.linalg.cholesky(corr)

    base_probs = np.array([current_probabilities.get(event_id, 0.5) for event_id in unique_events], dtype=float)
    vols = np.array([daily_vols.get(event_id, 0.08) for event_id in unique_events], dtype=float)

    rng = np.random.default_rng(42)
    shocks = rng.standard_normal((n_paths, n_days, len(unique_events)))
    correlated = np.einsum("pde,ef->pdf", shocks, chol.T)

    logits = np.log(np.clip(base_probs, 0.02, 0.98) / (1.0 - np.clip(base_probs, 0.02, 0.98)))
    final_logits = np.repeat(logits[np.newaxis, :], n_paths, axis=0)

    for day in range(n_days):
        final_logits += correlated[:, day, :] * vols * 3.0

    final_probs = 1.0 / (1.0 + np.exp(-np.clip(final_logits, -10.0, 10.0)))
    event_index = {event_id: idx for idx, event_id in enumerate(unique_events)}

    position_paths = []
    per_position = []
    for position in positions:
        idx = event_index[position.event_id]
        direction = 1.0 if position.long_floating else -1.0
        pnl = (final_probs[:, idx] - current_probabilities.get(position.event_id, 0.5)) * position.notional * direction
        position_paths.append(pnl)
        per_position.append(
            {
                "position_id": position.position_id,
                "event_id": position.event_id,
                "mean_pnl": round(float(np.mean(pnl)), 2),
                "std_pnl": round(float(np.std(pnl)), 2),
            }
        )

    stacked = np.vstack(position_paths)
    book_pnl = np.sum(stacked, axis=0)
    var_95 = float(np.percentile(book_pnl, 5))
    expected_shortfall_95 = float(book_pnl[book_pnl <= var_95].mean()) if np.any(book_pnl <= var_95) else var_95

    tail_mask = book_pnl <= var_95
    tail_contributors = []
    for idx, position in enumerate(positions):
        tail_mean = float(stacked[idx, tail_mask].mean()) if np.any(tail_mask) else 0.0
        tail_contributors.append(
            {
                "position_id": position.position_id,
                "event_id": position.event_id,
                "tail_mean_pnl": round(tail_mean, 2),
            }
        )
    tail_contributors.sort(key=lambda item: item["tail_mean_pnl"])

    hist_counts, hist_edges = np.histogram(book_pnl, bins=20)
    histogram = []
    for idx, count in enumerate(hist_counts):
        histogram.append(
            {
                "bin_start": round(float(hist_edges[idx]), 2),
                "bin_end": round(float(hist_edges[idx + 1]), 2),
                "count": int(count),
            }
        )

    risk = {
        "mean_pnl": round(float(np.mean(book_pnl)), 2),
        "std_pnl": round(float(np.std(book_pnl)), 2),
        "var_95": round(var_95, 2),
        "expected_shortfall_95": round(expected_shortfall_95, 2),
        "best_case": round(float(np.max(book_pnl)), 2),
        "worst_case": round(float(np.min(book_pnl)), 2),
    }

    return {
        "summary": summarize_positions(positions, current_probabilities),
        "risk": risk,
        "histogram": histogram,
        "tail_contributors": tail_contributors,
        "positions": per_position,
    }
