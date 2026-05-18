from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon

ONESTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ONESTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onestep.src import (
    AnsweredQuery,
    AlternativenMatrix,
    Query,
    QueryInfo,
    TerminationResult,
    build_W,
    build_no_informative_query_termination_result,
    build_one_remaining_candidate_termination_result,
    build_optimal_region_in_W,
    build_same_utility_termination_result,
    build_zielpaar_intervalle_lookup,
    compute_all_query_kandidaten,
    compute_all_ratio_intervals,
    compute_query_info,
    estimate_optimality_shares,
    filter_already_answered_queries,
    filter_informative_query_infos,
    sample_points_from_ungleichungssystem,
)
from onestep.src.termination import all_candidates_have_same_utility_values_in_W


SQRT3 = np.sqrt(3.0)
QUERY_EPSILON = 1e-3
VISUALIZATION_EQUALITY_TOL = 0.0025
DEFAULT_GRID_STEP = 0.002
DEFAULT_GRID_POINT_SIZE = 0.4
DEFAULT_TARGET_POINT_SIZE = 20.0
DEFAULT_SAMPLE_COUNT = 10000
DEFAULT_SAMPLE_BURN_IN = 300
DEFAULT_SAMPLE_THINNING = 4
DEFAULT_SAMPLE_POINT_SIZE = 1
GOAL_LABELS = ["Spaß", "Finanzen", "Sicherheit"]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--load-tradeoffs", action="store_true")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def resolve_data_file_path(data_file: str) -> Path:
    path = Path(data_file).expanduser()
    if not path.is_absolute():
        path = ONESTEP_ROOT / path
    return path.resolve()


def load_problem_from_json(
    data_file: str,
    load_tradeoffs: bool,
) -> tuple[list[str], np.ndarray, np.ndarray, list[AnsweredQuery], str]:
    path = resolve_data_file_path(data_file)
    if not path.is_file():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    matrix_data = payload.get("handlungsalternativenmatrix")
    if not isinstance(matrix_data, list) or not matrix_data:
        raise ValueError("JSON-Datei enthaelt keine gueltige handlungsalternativenmatrix.")

    utilities = np.array(matrix_data, dtype=float)
    if utilities.ndim != 2 or utilities.shape[1] != 3:
        raise ValueError("Die handlungsalternativenmatrix muss die Form (m, 3) haben.")

    alternative_names = payload.get("alternative_names")
    if not isinstance(alternative_names, list) or len(alternative_names) != len(utilities):
        alternative_names = [f"A{i}" for i in range(1, len(utilities) + 1)]
    else:
        alternative_names = [str(name) for name in alternative_names]

    target_weights_data = payload.get("target_weights", {}).get("vector")
    if not isinstance(target_weights_data, list) or len(target_weights_data) != 3:
        raise ValueError("JSON-Datei enthaelt keinen gueltigen target_weights.vector-Eintrag.")

    target_weights = np.array(target_weights_data, dtype=float)

    answered_queries: list[AnsweredQuery] = []
    if load_tradeoffs:
        tradeoffs = payload.get("tradeoffs", [])
        if not isinstance(tradeoffs, list):
            raise ValueError("tradeoffs muss in der JSON-Datei eine Liste sein.")

        for entry in tradeoffs:
            answered_queries.append(
                AnsweredQuery(
                    ziel_index_a=int(entry["ziel_index_a"]),
                    ziel_index_b=int(entry["ziel_index_b"]),
                    value=float(entry["value"]),
                    operator=str(entry["operator"]),
                )
            )

    return alternative_names, utilities, target_weights, answered_queries, str(path)


def barycentric_to_cartesian(w_1: float, w_2: float, w_3: float) -> tuple[float, float]:
    x_value = w_2 + 0.5 * w_3
    y_value = (SQRT3 / 2.0) * w_3
    return x_value, y_value


def cartesian_to_barycentric(x_value: float, y_value: float) -> np.ndarray:
    w_3 = (2.0 / SQRT3) * y_value
    w_2 = x_value - 0.5 * w_3
    w_1 = 1.0 - w_2 - w_3
    return np.array([w_1, w_2, w_3], dtype=float)


def generate_random_alternatives(n_alternatives: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng()
    alternative_names = [f"A{i}" for i in range(1, n_alternatives + 1)]
    utilities = np.round(rng.uniform(0.0, 1.0, size=(n_alternatives, 3)), 2)
    return alternative_names, utilities


def generate_random_target_weights() -> np.ndarray:
    rng = np.random.default_rng()
    return rng.dirichlet(alpha=[1.0, 1.0, 1.0]).astype(float)


@st.cache_data(show_spinner=False)
def generate_simplex_grid(step: float) -> np.ndarray:
    grid: list[tuple[float, float, float]] = []
    values = np.arange(0.0, 1.0 + step / 2.0, step)

    for w_1 in values:
        for w_2 in values:
            w_3 = 1.0 - w_1 - w_2
            if w_3 >= -1e-12:
                grid.append((w_1, w_2, max(0.0, w_3)))

    return np.array(grid, dtype=float)


def sample_visualization_weights(
    answered_queries: list[AnsweredQuery],
    sample_count: int,
    sample_burn_in: int,
    sample_thinning: int,
) -> np.ndarray:
    visualization_W = build_W(3, answered_queries)
    if not visualization_W.is_feasible():
        raise ValueError("W(T) ist infeasible.")

    samples = sample_points_from_ungleichungssystem(
        visualization_W,
        num_samples=sample_count,
        burn_in=sample_burn_in,
        thinning=sample_thinning,
    )
    return np.array(samples, dtype=float)


def satisfies_answered_query_for_visualization(
    weights: np.ndarray,
    answered_query: AnsweredQuery,
) -> bool:
    left_value = float(weights[answered_query.ziel_index_a])
    right_value = float(answered_query.value * weights[answered_query.ziel_index_b])

    if answered_query.operator == ">":
        return left_value >= right_value - 1e-9
    if answered_query.operator == "<":
        return left_value <= right_value + 1e-9
    if answered_query.operator == "=":
        return abs(left_value - right_value) <= VISUALIZATION_EQUALITY_TOL

    raise ValueError(f"unknown operator: {answered_query.operator}")


def satisfies_all_answered_queries_for_visualization(
    weights: np.ndarray,
    answered_queries: list[AnsweredQuery],
) -> bool:
    return all(
        satisfies_answered_query_for_visualization(weights, answered_query)
        for answered_query in answered_queries
    )


def build_problem_signature(
    utilities: np.ndarray,
    answered_queries: list[AnsweredQuery],
) -> tuple[Any, ...]:
    utilities_signature = tuple(
        tuple(float(value) for value in row)
        for row in utilities.tolist()
    )
    answered_queries_signature = tuple(
        (
            int(answered_query.ziel_index_a),
            int(answered_query.ziel_index_b),
            float(answered_query.value),
            answered_query.operator,
        )
        for answered_query in answered_queries
    )
    return utilities_signature, answered_queries_signature


def build_sampling_signature(
    utilities: np.ndarray,
    answered_queries: list[AnsweredQuery],
    sample_count: int,
    sample_burn_in: int,
    sample_thinning: int,
    sampling_version: int,
) -> tuple[Any, ...]:
    return build_problem_signature(utilities, answered_queries) + (
        sample_count,
        sample_burn_in,
        sample_thinning,
        sampling_version,
    )


def get_alternative_colors(n_alternatives: int) -> list[str]:
    if n_alternatives == 1:
        return ["#4C78A8"]

    cmap = plt.get_cmap("nipy_spectral")
    return [
        mcolors.to_hex(cmap(position))
        for position in np.linspace(0.03, 0.97, n_alternatives)
    ]


def query_to_text(query: Query) -> str:
    left_label = GOAL_LABELS[query.ziel_index_a]
    right_label = GOAL_LABELS[query.ziel_index_b]
    return f"{left_label} ? {query.value:g} * {right_label}"


def answered_query_to_text(answered_query: AnsweredQuery) -> str:
    left_label = GOAL_LABELS[answered_query.ziel_index_a]
    right_label = GOAL_LABELS[answered_query.ziel_index_b]
    return f"{left_label} {answered_query.operator} {answered_query.value:g} * {right_label}"


def linear_result_to_text(result: Any) -> str:
    if result.status == "optimal":
        if result.optimal_value is None:
            raise RuntimeError("optimal result has no value")
        return f"{result.optimal_value:g}"
    return result.status


def compute_best_alternatives_on_grid(grid: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    total_utilities = utilities @ grid.T
    return np.argmax(total_utilities, axis=0).astype(int)


def point_is_in_simplex(weights: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.all(weights >= -tol) and abs(float(np.sum(weights)) - 1.0) <= 1e-6)


def query_is_on_greater_side(query: Query, weights: np.ndarray) -> bool:
    return float(weights[query.ziel_index_a]) >= float(query.value * weights[query.ziel_index_b]) - 1e-9


def get_query_line_segment(query: Query) -> tuple[tuple[float, float], tuple[float, float]] | None:
    simplex_vertices = np.eye(3)
    edge_indices = [(0, 1), (1, 2), (2, 0)]
    barycentric_points: list[np.ndarray] = []

    def boundary_value(weights: np.ndarray) -> float:
        return float(weights[query.ziel_index_a] - query.value * weights[query.ziel_index_b])

    def add_point(candidate: np.ndarray) -> None:
        for existing in barycentric_points:
            if np.allclose(existing, candidate, atol=1e-9):
                return
        barycentric_points.append(candidate)

    for start_idx, end_idx in edge_indices:
        start = simplex_vertices[start_idx]
        end = simplex_vertices[end_idx]
        start_value = boundary_value(start)
        end_value = boundary_value(end)

        if abs(start_value) <= 1e-9 and abs(end_value) <= 1e-9:
            add_point(start)
            add_point(end)
            continue

        if abs(start_value) <= 1e-9:
            add_point(start)
            continue

        if abs(end_value) <= 1e-9:
            add_point(end)
            continue

        if start_value * end_value < 0:
            interpolation = start_value / (start_value - end_value)
            add_point(start + interpolation * (end - start))

    if len(barycentric_points) != 2:
        return None

    first = barycentric_to_cartesian(*barycentric_points[0].tolist())
    second = barycentric_to_cartesian(*barycentric_points[1].tolist())
    return first, second


def get_query_side_arrow(
    query: Query,
    segment: tuple[tuple[float, float], tuple[float, float]],
    want_greater_side: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    point_a = np.array(segment[0], dtype=float)
    point_b = np.array(segment[1], dtype=float)
    midpoint = 0.5 * (point_a + point_b)
    direction = point_b - point_a
    norm = np.linalg.norm(direction)

    if norm <= 1e-12:
        return None

    normal = np.array([-direction[1], direction[0]], dtype=float) / norm

    for arrow_length in (0.11, 0.08, 0.05):
        for sign in (1.0, -1.0):
            arrow_end = midpoint + sign * arrow_length * normal
            weights = cartesian_to_barycentric(float(arrow_end[0]), float(arrow_end[1]))
            if not point_is_in_simplex(weights):
                continue

            on_greater_side = query_is_on_greater_side(query, weights)
            if on_greater_side == want_greater_side:
                return midpoint, arrow_end

    return None


def draw_triangle(
    grid: np.ndarray,
    utilities: np.ndarray,
    alternative_names: list[str],
    answered_queries: list[AnsweredQuery],
    selected_query: Query | None,
    target_weights: np.ndarray | None,
    grid_point_size: float,
    target_point_size: float,
    target_point_is_black: bool,
) -> plt.Figure:
    colors = get_alternative_colors(len(alternative_names))
    fig, ax = plt.subplots(figsize=(4.7, 4.0))

    triangle_vertices = np.array(
        [
            barycentric_to_cartesian(1.0, 0.0, 0.0),
            barycentric_to_cartesian(0.0, 1.0, 0.0),
            barycentric_to_cartesian(0.0, 0.0, 1.0),
        ]
    )

    triangle = Polygon(
        triangle_vertices,
        closed=True,
        facecolor="#f8f9fb",
        edgecolor="#1f1f1f",
        linewidth=1.8,
    )
    ax.add_patch(triangle)

    grid_points = np.array([barycentric_to_cartesian(*weights.tolist()) for weights in grid], dtype=float)
    feasible_mask = np.array(
        [satisfies_all_answered_queries_for_visualization(weights, answered_queries) for weights in grid],
        dtype=bool,
    )
    best_indices = compute_best_alternatives_on_grid(grid, utilities)

    excluded_points = grid_points[~feasible_mask]
    feasible_points = grid_points[feasible_mask]

    if excluded_points.size > 0:
        ax.scatter(
            excluded_points[:, 0],
            excluded_points[:, 1],
            s=grid_point_size,
            c="#d0d3da",
            marker="s",
            linewidths=0,
            zorder=1,
            rasterized=True,
        )

    if feasible_points.size > 0:
        ax.scatter(
            feasible_points[:, 0],
            feasible_points[:, 1],
            c=best_indices[feasible_mask],
            cmap=mcolors.ListedColormap(colors),
            s=grid_point_size,
            marker="s",
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    goal_positions = [
        barycentric_to_cartesian(1.0, 0.0, 0.0),
        barycentric_to_cartesian(0.0, 1.0, 0.0),
        barycentric_to_cartesian(0.0, 0.0, 1.0),
    ]
    goal_offsets = [(-0.06, -0.04), (0.01, -0.04), (-0.04, 0.02)]

    for (x_value, y_value), (x_offset, y_offset), label in zip(goal_positions, goal_offsets, GOAL_LABELS):
        ax.text(x_value + x_offset, y_value + y_offset, label, fontsize=11, fontweight="bold")

    legend_handles: list[plt.Line2D] = []
    if len(alternative_names) <= 12:
        legend_handles.extend(
            [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=alternative_name,
                    markerfacecolor=colors[index],
                    markersize=7,
                )
                for index, alternative_name in enumerate(alternative_names)
            ]
        )

    if excluded_points.size > 0:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Ausgeschlossen",
                markerfacecolor="#d0d3da",
                markersize=7,
            )
        )

    if selected_query is not None:
        segment = get_query_line_segment(selected_query)
        if segment is not None:
            (x_1, y_1), (x_2, y_2) = segment
            ax.plot(
                [x_1, x_2],
                [y_1, y_2],
                color="#111111",
                linewidth=2.2,
                linestyle="--",
                zorder=3,
            )
            legend_handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#111111",
                    linewidth=2.2,
                    linestyle="--",
                    label="Ausgewaehlte Query",
                )
            )

            greater_arrow = get_query_side_arrow(selected_query, segment, want_greater_side=True)
            if greater_arrow is not None:
                arrow_start, arrow_end = greater_arrow
                ax.annotate(
                    "",
                    xy=(float(arrow_end[0]), float(arrow_end[1])),
                    xytext=(float(arrow_start[0]), float(arrow_start[1])),
                    arrowprops=dict(arrowstyle="-|>", color="#111111", linewidth=1.8),
                    zorder=4,
                )
                label_position = arrow_end + 0.02 * (arrow_end - arrow_start)
                ax.text(
                    float(label_position[0]),
                    float(label_position[1]),
                    ">",
                    fontsize=11,
                    fontweight="bold",
                    color="#111111",
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="#111111",
                        alpha=0.9,
                    ),
                )

            smaller_arrow = get_query_side_arrow(selected_query, segment, want_greater_side=False)
            if smaller_arrow is not None:
                arrow_start, arrow_end = smaller_arrow
                ax.annotate(
                    "",
                    xy=(float(arrow_end[0]), float(arrow_end[1])),
                    xytext=(float(arrow_start[0]), float(arrow_start[1])),
                    arrowprops=dict(arrowstyle="-|>", color="#111111", linewidth=1.8),
                    zorder=4,
                )
                label_position = arrow_end + 0.02 * (arrow_end - arrow_start)
                ax.text(
                    float(label_position[0]),
                    float(label_position[1]),
                    "<",
                    fontsize=11,
                    fontweight="bold",
                    color="#111111",
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white",
                        edgecolor="#111111",
                        alpha=0.9,
                    ),
                )

    if target_weights is not None:
        target_facecolor = "#111111" if target_point_is_black else "#ffffff"
        target_x, target_y = barycentric_to_cartesian(*target_weights.tolist())
        ax.scatter(
            [target_x],
            [target_y],
            s=target_point_size,
            c=target_facecolor,
            marker="o",
            edgecolors="none",
            linewidths=0.0,
            zorder=6,
        )
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Zielgewichtsvektor",
                markerfacecolor=target_facecolor,
                markeredgecolor=target_facecolor,
                markeredgewidth=0.0,
                markersize=8,
                linewidth=0,
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, SQRT3 / 2.0 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.7, top=0.98, bottom=0.03)
    return fig


def draw_sampling_plot(
    probability_samples: np.ndarray,
    query: Query,
    sample_point_size: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.7, 4.0))

    triangle_vertices = np.array(
        [
            barycentric_to_cartesian(1.0, 0.0, 0.0),
            barycentric_to_cartesian(0.0, 1.0, 0.0),
            barycentric_to_cartesian(0.0, 0.0, 1.0),
        ]
    )
    triangle = Polygon(
        triangle_vertices,
        closed=True,
        facecolor="#f8f9fb",
        edgecolor="#1f1f1f",
        linewidth=1.8,
    )
    ax.add_patch(triangle)

    sample_points = np.array(
        [barycentric_to_cartesian(*weights.tolist()) for weights in probability_samples],
        dtype=float,
    )
    diff = probability_samples[:, query.ziel_index_a] - query.value * probability_samples[:, query.ziel_index_b]
    less_mask = diff < 0.0
    greater_mask = diff > 0.0
    equal_mask = ~(less_mask | greater_mask)

    if np.any(less_mask):
        ax.scatter(
            sample_points[less_mask, 0],
            sample_points[less_mask, 1],
            s=sample_point_size,
            c="#4C78A8",
            marker="s",
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    if np.any(greater_mask):
        ax.scatter(
            sample_points[greater_mask, 0],
            sample_points[greater_mask, 1],
            s=sample_point_size,
            c="#F58518",
            marker="s",
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    if np.any(equal_mask):
        ax.scatter(
            sample_points[equal_mask, 0],
            sample_points[equal_mask, 1],
            s=sample_point_size,
            c="#B8B8BF",
            marker="s",
            linewidths=0,
            zorder=2,
            rasterized=True,
        )

    goal_positions = [
        barycentric_to_cartesian(1.0, 0.0, 0.0),
        barycentric_to_cartesian(0.0, 1.0, 0.0),
        barycentric_to_cartesian(0.0, 0.0, 1.0),
    ]
    goal_offsets = [(-0.06, -0.04), (0.01, -0.04), (-0.04, 0.02)]
    for (x_value, y_value), (x_offset, y_offset), label in zip(goal_positions, goal_offsets, GOAL_LABELS):
        ax.text(x_value + x_offset, y_value + y_offset, label, fontsize=11, fontweight="bold")

    segment = get_query_line_segment(query)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="< Seite", markerfacecolor="#4C78A8", markersize=7),
        plt.Line2D([0], [0], marker="s", color="w", label="> Seite", markerfacecolor="#F58518", markersize=7),
    ]
    if np.any(equal_mask):
        legend_handles.append(
            plt.Line2D([0], [0], marker="s", color="w", label="= Rand", markerfacecolor="#B8B8BF", markersize=7)
        )

    if segment is not None:
        (x_1, y_1), (x_2, y_2) = segment
        ax.plot(
            [x_1, x_2],
            [y_1, y_2],
            color="#111111",
            linewidth=2.2,
            linestyle="--",
            zorder=3,
        )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, SQRT3 / 2.0 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.7, top=0.98, bottom=0.03)
    return fig


def build_kandidatenmenge(
    alternativen_matrix: AlternativenMatrix,
    W: Any,
) -> list[int]:
    kandidatenmenge: list[int] = []

    for alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
        optimal_region = build_optimal_region_in_W(
            alternativen_matrix=alternativen_matrix,
            W=W,
            alternative_index=alternative_index,
        )
        if optimal_region.is_feasible():
            kandidatenmenge.append(alternative_index)

    return kandidatenmenge


def compute_algorithm_details(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    probability_samples: np.ndarray,
) -> dict[str, Any]:
    W = build_W(alternativen_matrix.get_anzahl_spalten(), answered_queries)
    if not W.is_feasible():
        raise ValueError("W(T) ist infeasible.")

    kandidatenmenge = build_kandidatenmenge(alternativen_matrix, W)
    if not kandidatenmenge:
        raise ValueError("Es wurden keine Kandidaten in W(T) gefunden.")

    state: dict[str, Any] = {
        "W": W,
        "kandidatenmenge": kandidatenmenge,
        "termination": None,
        "zielpaar_intervalle": [],
        "query_infos": [],
        "best_query": None,
        "best_expected": None,
        "samples_count": 0,
    }

    if len(kandidatenmenge) == 1:
        state["termination"] = build_one_remaining_candidate_termination_result(kandidatenmenge[0])
        return state

    if all_candidates_have_same_utility_values_in_W(
        alternativen_matrix=alternativen_matrix,
        W=W,
        kandidatenmenge=kandidatenmenge,
    ):
        state["termination"] = build_same_utility_termination_result(kandidatenmenge)
        return state

    zielpaar_intervalle = compute_all_ratio_intervals(
        alternativen_matrix=alternativen_matrix,
        answered_queries=answered_queries,
        kandidatenmenge=kandidatenmenge,
    )
    query_kandidaten = compute_all_query_kandidaten(zielpaar_intervalle, epsilon=QUERY_EPSILON)
    query_kandidaten = filter_already_answered_queries(query_kandidaten, answered_queries)
    state["zielpaar_intervalle"] = zielpaar_intervalle

    zielpaar_intervalle_lookup = build_zielpaar_intervalle_lookup(zielpaar_intervalle)

    query_infos: list[QueryInfo] = []

    for query in query_kandidaten:
        query_info = compute_query_info(
            query=query,
            samples=probability_samples.tolist(),
            zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
        )
        if query_info is not None:
            query_infos.append(query_info)

    informative_query_infos = filter_informative_query_infos(
        query_infos=query_infos,
        kandidatenmenge=set(kandidatenmenge),
    )

    if not informative_query_infos:
        state["termination"] = build_no_informative_query_termination_result(
            kandidatenmenge=kandidatenmenge,
            optimality_shares=estimate_optimality_shares(
                alternativen_matrix=alternativen_matrix,
                samples=probability_samples.tolist(),
                remaining_candidates=kandidatenmenge,
            ),
        )
        state["query_infos"] = []
        state["best_query"] = None
        state["best_expected"] = None
        state["samples_count"] = len(probability_samples)
        return state

    informative_query_infos.sort(
        key=lambda query_info: (query_info.expected_kandidatenanzahl, query_info.query.value)
    )
    best_query_info = informative_query_infos[0]
    state["query_infos"] = informative_query_infos
    state["best_query"] = best_query_info.query
    state["best_expected"] = best_query_info.expected_kandidatenanzahl
    state["samples_count"] = len(probability_samples)
    return state


def build_ratio_interval_dataframe(
    zielpaar_intervalle: list[Any],
    alternative_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for zielpaar in zielpaar_intervalle:
        left_label = GOAL_LABELS[zielpaar.ziel_index_a]
        right_label = GOAL_LABELS[zielpaar.ziel_index_b]
        for candidate_index, ratio_interval in zielpaar.intervalle_pro_kandidat.items():
            rows.append(
                {
                    "Ziel links": left_label,
                    "Ziel rechts": right_label,
                    "Kandidat": alternative_names[candidate_index],
                    "alpha": linear_result_to_text(ratio_interval.lower),
                    "beta": linear_result_to_text(ratio_interval.upper),
                }
            )

    return pd.DataFrame(rows)


def build_query_dataframe(
    query_infos: list[QueryInfo],
    alternative_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for index, query_info in enumerate(query_infos, start=1):
        query = query_info.query
        rows.append(
            {
                "Rang": index,
                "Query": query_to_text(query),
                "p_<": query_info.p_kleiner,
                "p_>": query_info.p_groesser,
                "N_<": len(query_info.kandidatenmenge_nach_kleiner_antwort),
                "N_>": len(query_info.kandidatenmenge_nach_groesser_antwort),
                "K_<": ", ".join(
                    alternative_names[i]
                    for i in sorted(query_info.kandidatenmenge_nach_kleiner_antwort)
                ),
                "K_>": ", ".join(
                    alternative_names[i]
                    for i in sorted(query_info.kandidatenmenge_nach_groesser_antwort)
                ),
                "E[N|q]": query_info.expected_kandidatenanzahl,
            }
        )

    return pd.DataFrame(rows)


def build_termination_shares_dataframe(
    termination: TerminationResult,
    alternative_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for candidate_index in termination.remaining_candidates:
        rows.append(
            {
                "Kandidat": alternative_names[candidate_index],
                "Index": candidate_index,
                "Optimality share": termination.optimality_shares[candidate_index],
            }
        )

    return pd.DataFrame(rows)


def get_query_info_for_query(
    query_infos: list[QueryInfo],
    query: Query,
) -> QueryInfo:
    for query_info in query_infos:
        if query_info.query == query:
            return query_info
    raise ValueError("query not found in query_infos")


def render_selectable_query_table(
    query_infos: list[QueryInfo],
    alternative_names: list[str],
    selected_query_index: int,
) -> int:
    query_df = build_query_dataframe(
        query_infos=query_infos,
        alternative_names=alternative_names,
    )
    query_df.insert(
        0,
        "Auswahl",
        [index == selected_query_index for index in range(len(query_df))],
    )
    edited_df = st.data_editor(
        query_df,
        use_container_width=True,
        hide_index=True,
        key="three_goals_query_table",
        disabled=["Rang", "Query", "p_<", "p_>", "N_<", "N_>", "K_<", "K_>", "E[N|q]"],
        column_config={
            "Auswahl": st.column_config.CheckboxColumn("Auswahl"),
            "p_<": st.column_config.NumberColumn("p_<", format="%.4f"),
            "p_>": st.column_config.NumberColumn("p_>", format="%.4f"),
            "E[N|q]": st.column_config.NumberColumn("E[N|q]", format="%.6f"),
        },
    )

    selected_rows = edited_df.index[edited_df["Auswahl"]].tolist()
    if not selected_rows:
        return selected_query_index
    if len(selected_rows) == 1:
        return int(selected_rows[0])

    if selected_query_index in selected_rows:
        for row_index in selected_rows:
            if row_index != selected_query_index:
                return int(row_index)

    return int(selected_rows[0])


def initialize_state(initial_n_alternatives: int) -> None:
    initialize_state_with_problem(
        initial_n_alternatives=initial_n_alternatives,
        loaded_problem=None,
    )


def apply_problem_state(
    alternative_names: list[str],
    utilities: np.ndarray,
    target_weights: np.ndarray,
    answered_queries: list[AnsweredQuery],
) -> None:
    st.session_state.three_goals_alternative_names = alternative_names
    st.session_state.three_goals_utilities = utilities
    st.session_state.three_goals_answered_queries = answered_queries
    st.session_state.three_goals_target_weights = target_weights
    st.session_state.three_goals_algorithm_requested = False
    st.session_state.three_goals_selected_query_index = 0
    st.session_state.three_goals_cached_visualization_signature = None
    st.session_state.three_goals_cached_visualization_weights = None
    st.session_state.three_goals_cached_algorithm_signature = None
    st.session_state.three_goals_cached_algorithm_details = None
    st.session_state.three_goals_sampling_version = 0


def initialize_state_with_problem(
    initial_n_alternatives: int,
    loaded_problem: tuple[list[str], np.ndarray, np.ndarray, list[AnsweredQuery], str] | None,
) -> None:
    if "three_goals_alternative_names" in st.session_state:
        return

    if loaded_problem is not None:
        alternative_names, utilities, target_weights, answered_queries, loaded_path = loaded_problem
        apply_problem_state(
            alternative_names=alternative_names,
            utilities=utilities,
            target_weights=target_weights,
            answered_queries=answered_queries,
        )
        st.session_state.three_goals_loaded_data_file = loaded_path
        st.session_state.three_goals_loaded_tradeoffs = bool(answered_queries)
    else:
        alternative_names, utilities = generate_random_alternatives(initial_n_alternatives)
        apply_problem_state(
            alternative_names=alternative_names,
            utilities=utilities,
            target_weights=generate_random_target_weights(),
            answered_queries=[],
        )
        st.session_state.three_goals_loaded_data_file = None
        st.session_state.three_goals_loaded_tradeoffs = False

    if "three_goals_query_mode" not in st.session_state:
        st.session_state.three_goals_query_mode = "Simple"


def regenerate_problem(n_alternatives: int) -> None:
    alternative_names, utilities = generate_random_alternatives(n_alternatives)
    apply_problem_state(
        alternative_names=alternative_names,
        utilities=utilities,
        target_weights=generate_random_target_weights(),
        answered_queries=[],
    )
    st.session_state.three_goals_loaded_data_file = None
    st.session_state.three_goals_loaded_tradeoffs = False


def append_answer(operator: str, query: Query) -> None:
    answered_queries = list(st.session_state.three_goals_answered_queries)
    answered_queries.append(query.answer(operator))
    st.session_state.three_goals_answered_queries = answered_queries
    st.session_state.three_goals_algorithm_requested = False
    st.session_state.three_goals_selected_query_index = 0
    st.session_state.three_goals_cached_visualization_signature = None
    st.session_state.three_goals_cached_visualization_weights = None
    st.session_state.three_goals_cached_algorithm_signature = None
    st.session_state.three_goals_cached_algorithm_details = None
    st.session_state.three_goals_sampling_version = 0


def main() -> None:
    st.set_page_config(page_title="3 Goals", layout="wide")
    st.title("3 Goals")

    cli_args = parse_cli_args()
    loaded_problem = None
    loaded_problem_error: str | None = None
    if cli_args.data_file:
        try:
            loaded_problem = load_problem_from_json(
                data_file=cli_args.data_file,
                load_tradeoffs=cli_args.load_tradeoffs,
            )
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            loaded_problem_error = str(exc)

    initial_n_alternatives = (
        len(loaded_problem[0]) if loaded_problem is not None else 10
    )
    initialize_state_with_problem(
        initial_n_alternatives=initial_n_alternatives,
        loaded_problem=loaded_problem,
    )

    with st.sidebar:
        st.header("Settings")
        show_handlungsalternativen = bool(
            st.toggle(
                "Show handlungsalternativen",
                value=False,
                key="three_goals_show_handlungsalternativen",
            )
        )
        grid_step = float(
            st.slider(
                "Grid step",
                min_value=0.0005,
                max_value=0.05,
                value=DEFAULT_GRID_STEP,
                step=0.0005,
                format="%.4f",
                help="Kleinere Werte machen die Hauptgrafik feiner.",
            )
        )
        grid_point_size = float(
            st.slider(
                "Grid point size",
                min_value=0.01,
                max_value=40.0,
                value=float(DEFAULT_GRID_POINT_SIZE),
                step=0.01,
                help="Groesse der Kaestchen in der Grid-Darstellung.",
            )
        )
        target_point_size = float(
            st.slider(
                "Target vector size",
                min_value=1.0,
                max_value=300.0,
                value=float(DEFAULT_TARGET_POINT_SIZE),
                step=1.0,
                help="Groesse des Zielgewichtsvektors w* in der Hauptgrafik.",
            )
        )
        target_point_is_black = bool(
            st.toggle(
                "Black target vector",
                value=True,
                help="Wechselt den Zielgewichtsvektor zwischen schwarzem und weissem Punkt.",
            )
        )
        query_mode_is_detailed = bool(
            st.toggle(
                "Detailed mode",
                value=st.session_state.get("three_goals_query_mode", "Simple") == "Detailed",
                key="three_goals_query_mode_toggle",
            )
        )
        query_mode = "Detailed" if query_mode_is_detailed else "Simple"
        st.session_state.three_goals_query_mode = query_mode
        if st.session_state.get("three_goals_algorithm_requested", False):
            sample_count_default = int(
                st.session_state.get("three_goals_sample_count", DEFAULT_SAMPLE_COUNT)
            )
            sample_burn_in_default = int(
                st.session_state.get("three_goals_sample_burn_in", DEFAULT_SAMPLE_BURN_IN)
            )
            sample_thinning_default = int(
                st.session_state.get("three_goals_sample_thinning", DEFAULT_SAMPLE_THINNING)
            )
            sample_point_size_default = float(
                st.session_state.get("three_goals_sample_point_size", DEFAULT_SAMPLE_POINT_SIZE)
            )

            if query_mode == "Detailed":
                show_verhaeltnisintervalle = bool(
                    st.toggle(
                        "Show verhaeltnis intervalle",
                        value=False,
                        key="three_goals_show_verhaeltnisintervalle",
                    )
                )
            else:
                show_verhaeltnisintervalle = False
            if query_mode == "Detailed":
                show_sampling = bool(
                    st.toggle(
                        "Show sampling",
                        value=False,
                        key="three_goals_show_sampling",
                    )
                )

                if show_sampling:
                    sample_count = int(
                        st.number_input(
                            "Samples",
                            min_value=100,
                            max_value=50000,
                            value=sample_count_default,
                            step=100,
                            key="three_goals_sample_count",
                            help="Mehr Samples stabilisieren p_< und p_>, kosten aber Laufzeit.",
                        )
                    )
                    sample_burn_in = int(
                        st.number_input(
                            "Burn-in",
                            min_value=0,
                            max_value=10000,
                            value=sample_burn_in_default,
                            step=50,
                            key="three_goals_sample_burn_in",
                            help="Anzahl verworfener Hit-and-Run-Schritte vor dem eigentlichen Sampling.",
                        )
                    )
                    sample_thinning = int(
                        st.number_input(
                            "Thinning",
                            min_value=1,
                            max_value=100,
                            value=sample_thinning_default,
                            step=1,
                            key="three_goals_sample_thinning",
                            help="Nimmt nur jeden k-ten Sampling-Schritt.",
                        )
                    )
                    if st.button("Resample", use_container_width=True):
                        st.session_state.three_goals_sampling_version += 1
                        st.session_state.three_goals_cached_visualization_signature = None
                        st.session_state.three_goals_cached_visualization_weights = None
                        st.session_state.three_goals_cached_algorithm_signature = None
                        st.session_state.three_goals_cached_algorithm_details = None
                        st.rerun()
                    sample_point_size = float(
                        st.slider(
                            "Sample point size",
                            min_value=0.2,
                            max_value=40.0,
                            value=sample_point_size_default,
                            step=0.2,
                            key="three_goals_sample_point_size",
                            help="Groesse der Stichprobenpunkte in der Sampling-Grafik.",
                        )
                    )
                else:
                    sample_count = sample_count_default
                    sample_burn_in = sample_burn_in_default
                    sample_thinning = sample_thinning_default
                    sample_point_size = sample_point_size_default
            else:
                show_sampling = False
                sample_count = sample_count_default
                sample_burn_in = sample_burn_in_default
                sample_thinning = sample_thinning_default
                sample_point_size = sample_point_size_default
        else:
            show_verhaeltnisintervalle = False
            show_sampling = False
            sample_count = DEFAULT_SAMPLE_COUNT
            sample_burn_in = DEFAULT_SAMPLE_BURN_IN
            sample_thinning = DEFAULT_SAMPLE_THINNING
            sample_point_size = float(DEFAULT_SAMPLE_POINT_SIZE)

    control_col_1, control_col_2, control_col_3 = st.columns([0.45, 0.275, 0.275])
    with control_col_1:
        current_n_alternatives = len(st.session_state.three_goals_alternative_names)
        n_alternatives = int(
            st.number_input(
                "Anzahl Handlungsalternativen",
                min_value=1,
                max_value=100,
                value=current_n_alternatives,
                step=1,
            )
        )
    with control_col_2:
        st.write("")
        st.write("")
        if st.button("Generiere Daten", use_container_width=True):
            regenerate_problem(n_alternatives)
            st.rerun()
    with control_col_3:
        st.write("")
        st.write("")
        if st.button("Generiere Zielgewichtsvektor", use_container_width=True):
            st.session_state.three_goals_target_weights = generate_random_target_weights()
            st.rerun()

    alternative_names: list[str] = st.session_state.three_goals_alternative_names
    utilities: np.ndarray = st.session_state.three_goals_utilities
    answered_queries: list[AnsweredQuery] = st.session_state.three_goals_answered_queries
    target_weights: np.ndarray | None = st.session_state.three_goals_target_weights

    if loaded_problem_error is not None:
        st.error(loaded_problem_error)
        return

    loaded_data_file = st.session_state.get("three_goals_loaded_data_file")
    if loaded_data_file is not None:
        tradeoffs_text = "mit geladenen Trade-offs" if st.session_state.get("three_goals_loaded_tradeoffs", False) else "ohne geladene Trade-offs"
        st.caption(f"Geladen aus `{loaded_data_file}` ({tradeoffs_text}).")

    if target_weights is None:
        st.write("**Zielgewichtsvektor:** Noch keiner generiert.")
    else:
        st.write(
            f"**Zielgewichtsvektor:** {GOAL_LABELS[0]} = {target_weights[0]:.4f}, "
            f"{GOAL_LABELS[1]} = {target_weights[1]:.4f}, "
            f"{GOAL_LABELS[2]} = {target_weights[2]:.4f}"
        )

    utilities_df = pd.DataFrame(
        {
            "Alternative": alternative_names,
            GOAL_LABELS[0]: utilities[:, 0],
            GOAL_LABELS[1]: utilities[:, 1],
            GOAL_LABELS[2]: utilities[:, 2],
        }
    )
    if show_handlungsalternativen:
        st.markdown("**Handlungsalternativen**")
        st.dataframe(utilities_df, use_container_width=True, hide_index=True)

    alternativen_matrix = AlternativenMatrix(entries=utilities.tolist())
    simplex_grid = generate_simplex_grid(grid_step)
    sampling_signature = build_sampling_signature(
        utilities=utilities,
        answered_queries=answered_queries,
        sample_count=sample_count,
        sample_burn_in=sample_burn_in,
        sample_thinning=sample_thinning,
        sampling_version=st.session_state.three_goals_sampling_version,
    )
    try:
        if st.session_state.three_goals_cached_visualization_signature != sampling_signature:
            st.session_state.three_goals_cached_visualization_weights = sample_visualization_weights(
                answered_queries=answered_queries,
                sample_count=sample_count,
                sample_burn_in=sample_burn_in,
                sample_thinning=sample_thinning,
            )
            st.session_state.three_goals_cached_visualization_signature = sampling_signature
        visualization_weights = st.session_state.three_goals_cached_visualization_weights
    except ValueError as exc:
        st.error(str(exc))
        return

    if visualization_weights is None:
        st.error("Es konnten keine Visualisierungssamples erzeugt werden.")
        return

    algorithm_details: dict[str, Any] | None = None
    algorithm_error: str | None = None

    if st.session_state.three_goals_algorithm_requested:
        try:
            if st.session_state.three_goals_cached_algorithm_signature != sampling_signature:
                st.session_state.three_goals_cached_algorithm_details = compute_algorithm_details(
                    alternativen_matrix=alternativen_matrix,
                    answered_queries=answered_queries,
                    probability_samples=visualization_weights,
                )
                st.session_state.three_goals_cached_algorithm_signature = sampling_signature
            algorithm_details = st.session_state.three_goals_cached_algorithm_details
        except ValueError as exc:
            algorithm_error = str(exc)

    selected_query: Query | None = None
    if algorithm_details is not None and not algorithm_details["termination"] and algorithm_details["query_infos"]:
        if query_mode == "Simple":
            selected_query = algorithm_details["best_query"]
        else:
            max_query_index = len(algorithm_details["query_infos"]) - 1
            current_index = min(st.session_state.three_goals_selected_query_index, max_query_index)
            selected_query = algorithm_details["query_infos"][current_index].query

    visualization = draw_triangle(
        grid=simplex_grid,
        utilities=utilities,
        alternative_names=alternative_names,
        answered_queries=answered_queries,
        selected_query=selected_query,
        target_weights=target_weights,
        grid_point_size=grid_point_size,
        target_point_size=target_point_size,
        target_point_is_black=target_point_is_black,
    )

    st.pyplot(visualization, use_container_width=False)
    st.caption("Diese Grid Darstellung zeigt welche Alternative in welchem Bereich optimal ist.")

    if len(alternative_names) > 12:
        st.info("Die Legende im Dreieck ist bei mehr als 12 Alternativen aus Platzgruenden ausgeblendet.")

    if answered_queries:
        st.markdown("**Bisher beantwortete Trade-off-Queries**")
        for index, answered_query in enumerate(answered_queries, start=1):
            st.write(f"{index}. {answered_query_to_text(answered_query)}")
    else:
        st.markdown("**Bisher beantwortete Trade-off-Queries**")
        st.write("Noch keine.")

    if not st.session_state.three_goals_algorithm_requested:
        if st.button("Run Algorithmus", use_container_width=True):
            st.session_state.three_goals_algorithm_requested = True
            st.session_state.three_goals_selected_query_index = 0
            st.rerun()

    if not st.session_state.three_goals_algorithm_requested:
        return

    if algorithm_error is not None:
        st.error(algorithm_error)
        return

    if algorithm_details is None:
        st.error("Der Algorithmus konnte nicht ausgewertet werden.")
        return

    kandidatenmenge = algorithm_details["kandidatenmenge"]
    kandidaten_namen = [alternative_names[index] for index in kandidatenmenge]
    termination = algorithm_details["termination"]
    if isinstance(termination, TerminationResult):
        if termination.reason == "one_remaining_candidate":
            st.success(
                f"Terminierungsfall 1: Nur noch {alternative_names[termination.remaining_candidates[0]]} bleibt uebrig."
            )
            return

        if termination.reason == "same_utility_values":
            same_names = [alternative_names[index] for index in termination.remaining_candidates]
            st.success(
                "Terminierungsfall 2: Alle verbleibenden Kandidaten haben auf ganz W(T) denselben Gesamtnutzen: "
                + ", ".join(same_names)
            )
        elif termination.reason == "no_informative_query":
            st.warning("Keine weitere informative Query existiert.")

        st.markdown("**Termination Result**")
        st.write(f"`reason`: `{termination.reason}`")
        st.write(f"`share_precision`: `{termination.share_precision}`")
        st.write(f"`remaining_candidates`: {termination.remaining_candidates}")
        st.write("`optimality_shares`:")
        st.dataframe(
            build_termination_shares_dataframe(
                termination=termination,
                alternative_names=alternative_names,
            ),
            use_container_width=True,
            hide_index=True,
        )
        return

    best_query: Query = algorithm_details["best_query"]
    best_expected: float = algorithm_details["best_expected"]
    answer_query: Query = best_query
    answer_expected: float = best_expected
    answer_query_info = get_query_info_for_query(algorithm_details["query_infos"], answer_query)

    if query_mode == "Detailed":
        current_index = min(
            st.session_state.three_goals_selected_query_index,
            len(algorithm_details["query_infos"]) - 1,
        )
        selected_query_info = algorithm_details["query_infos"][current_index]
        answer_query = selected_query_info.query
        answer_expected = selected_query_info.expected_kandidatenanzahl
        answer_query_info = selected_query_info

    if query_mode == "Detailed":
        st.subheader("Zwischenergebnisse")
        st.markdown("**Kandidatenmenge**")
        st.write(", ".join(kandidaten_namen))

        st.markdown("**Terminierungspruefung**")
        st.write("Kein Terminierungsfall aktiv.")

        if show_verhaeltnisintervalle:
            st.markdown("**Verhaeltnisintervalle**")
            ratio_interval_df = build_ratio_interval_dataframe(
                zielpaar_intervalle=algorithm_details["zielpaar_intervalle"],
                alternative_names=alternative_names,
            )
            st.dataframe(ratio_interval_df, use_container_width=True, hide_index=True)

        if show_sampling:
            st.markdown("**Sampling zur Wahrscheinlichkeitsschaetzung**")
            sampling_plot = draw_sampling_plot(
                probability_samples=visualization_weights,
                query=answer_query,
                sample_point_size=sample_point_size,
            )
            st.pyplot(sampling_plot, use_container_width=False)
            st.caption(
                f"Aktuelle Schaetzung fuer {query_to_text(answer_query)}: "
                f"p_< = {answer_query_info.p_kleiner:.4f}, "
                f"p_> = {answer_query_info.p_groesser:.4f}. "
                "Mit `Resample` werden nur die Stichproben und die daraus berechneten Query-Werte neu gezogen."
            )

        st.markdown("**Query-Auswertung**")
        st.write(
            f"Ausgewertet mit {algorithm_details['samples_count']} Samples aus W(T). "
            "Die folgende Tabelle enthaelt p_<, p_>, N_<, N_> und E[N|q]. "
            "Die aktuell ausgewaehlte Query ist markiert; durch Klick auf eine andere Zeile "
            "wird sie im Dreieck angezeigt."
        )
        selected_query_index = render_selectable_query_table(
            query_infos=algorithm_details["query_infos"],
            alternative_names=alternative_names,
            selected_query_index=current_index,
        )
        if selected_query_index != st.session_state.three_goals_selected_query_index:
            st.session_state.three_goals_selected_query_index = selected_query_index
            st.rerun()

        selected_query_info = algorithm_details["query_infos"][selected_query_index]
        answer_query = selected_query_info.query
        answer_expected = selected_query_info.expected_kandidatenanzahl
        answer_query_info = selected_query_info
        st.markdown("**Aktuell ausgewaehlte Query**")
        st.write(query_to_text(selected_query_info.query))
        st.latex(rf"\mathbb{{E}}[N \mid q] = {selected_query_info.expected_kandidatenanzahl:.6g}")
        st.caption("Diese ausgewaehlte Query kann direkt beantwortet werden.")

    if query_mode == "Simple":
        st.markdown("**Beste Query**")
        st.write(query_to_text(best_query))
        st.latex(
            rf"\mathbb{{E}}[N \mid q^\star] = {best_expected:g}"
        )
    else:
        st.markdown("**Query beantworten**")

    answer_left, answer_middle, answer_right = st.columns(3)
    with answer_left:
        if st.button("< beantworten", use_container_width=True):
            append_answer("<", answer_query)
            st.rerun()
    with answer_middle:
        if st.button("= beantworten", use_container_width=True):
            append_answer("=", answer_query)
            st.rerun()
    with answer_right:
        if st.button("> beantworten", use_container_width=True):
            append_answer(">", answer_query)
            st.rerun()


if __name__ == "__main__":
    main()
