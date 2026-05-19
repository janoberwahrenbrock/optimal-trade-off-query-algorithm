from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import compute_candidate_set
from multistep.src.models import AlternativenMatrix as MultistepAlternativenMatrix
from multistep.src.models import Query as MultistepQuery
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.value_function import MultistepConfig, compute_value_function
from multistep.src.weight_space import build_weight_space
from onestep.src.build_ungleichungssysteme import build_W, build_optimal_region_in_W
from onestep.src.io_models import AlternativenMatrix as OnestepAlternativenMatrix
from onestep.src.io_models import Query as OnestepQuery
from onestep.src.query_bewertung import (
    build_zielpaar_intervalle_lookup,
    compute_query_info,
    filter_informative_query_infos,
)
from onestep.src.query_kandidaten import compute_all_query_kandidaten
from onestep.src.ratio_intervalle import compute_all_ratio_intervals
from onestep.src.sampling import sample_points_from_ungleichungssystem


SQRT3 = math.sqrt(3.0)
PROBLEM_SEED = 1
PROBLEM_INDEX = 0
MIN_GOALS = 3
MAX_GOALS = 7
MIN_ALTERNATIVES = 3
MAX_ALTERNATIVES = 10
DEFAULT_SAMPLE_COUNT = 10_000
DEFAULT_BURN_IN = 200
DEFAULT_THINNING = 5
DEFAULT_QUERY_EPSILON = 1e-3
DEFAULT_SIMPLEX_STEP = 0.02


def generate_problem(
    problem_index: int,
    seed: int,
) -> tuple[int, int, list[list[float]]]:
    rng = np.random.default_rng(seed)

    goal_count = 0
    alternative_count = 0
    entries: list[list[float]] = []
    for _ in range(problem_index + 1):
        goal_count = int(rng.integers(MIN_GOALS, MAX_GOALS + 1))
        alternative_count = int(rng.integers(MIN_ALTERNATIVES, MAX_ALTERNATIVES + 1))
        entries = rng.uniform(
            0.0,
            1.0,
            size=(alternative_count, goal_count),
        ).astype(float).tolist()

    return goal_count, alternative_count, entries


def format_query(query: OnestepQuery | MultistepQuery | None) -> str:
    if query is None:
        return "-"

    return (
        f"({int(query.ziel_index_a)}, {int(query.ziel_index_b)}, "
        f"{float(query.value):.8g})"
    )


def query_key(query: OnestepQuery | MultistepQuery) -> tuple[int, int, float]:
    return (
        int(query.ziel_index_a),
        int(query.ziel_index_b),
        round(float(query.value), 12),
    )


def get_alternative_colors(alternative_count: int) -> list[str]:
    cmap = plt.get_cmap("nipy_spectral")
    return [
        mcolors.to_hex(cmap(position))
        for position in np.linspace(0.03, 0.97, alternative_count)
    ]


def barycentric_to_cartesian(w_1: float, w_2: float, w_3: float) -> tuple[float, float]:
    return w_2 + 0.5 * w_3, (SQRT3 / 2.0) * w_3


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


def build_full_weights_from_face(
    face_weights: np.ndarray,
    selected_goal_indices: tuple[int, int, int],
    goal_count: int,
) -> np.ndarray:
    full_weights = np.zeros((len(face_weights), goal_count), dtype=float)
    for local_index, goal_index in enumerate(selected_goal_indices):
        full_weights[:, goal_index] = face_weights[:, local_index]

    return full_weights


def compute_best_alternatives_on_face(
    entries: list[list[float]],
    face_weights: np.ndarray,
    selected_goal_indices: tuple[int, int, int],
) -> np.ndarray:
    full_weights = build_full_weights_from_face(
        face_weights=face_weights,
        selected_goal_indices=selected_goal_indices,
        goal_count=len(entries[0]),
    )
    utilities = np.array(entries, dtype=float)
    total_utilities = utilities @ full_weights.T
    return np.argmax(total_utilities, axis=0).astype(int)


def get_query_line_segment_on_face(
    query: OnestepQuery | MultistepQuery,
    selected_goal_indices: tuple[int, int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if query.ziel_index_a not in selected_goal_indices:
        return None

    if query.ziel_index_b not in selected_goal_indices:
        return None

    simplex_vertices = np.eye(3)
    edge_indices = [(0, 1), (1, 2), (2, 0)]
    points: list[np.ndarray] = []
    goal_to_local_index = {
        goal_index: local_index
        for local_index, goal_index in enumerate(selected_goal_indices)
    }

    local_index_a = goal_to_local_index[int(query.ziel_index_a)]
    local_index_b = goal_to_local_index[int(query.ziel_index_b)]

    def boundary_value(local_weights: np.ndarray) -> float:
        return float(local_weights[local_index_a] - float(query.value) * local_weights[local_index_b])

    def add_point(candidate: np.ndarray) -> None:
        for existing in points:
            if np.allclose(existing, candidate, atol=1e-9):
                return
        points.append(candidate)

    for start_index, end_index in edge_indices:
        start = simplex_vertices[start_index]
        end = simplex_vertices[end_index]
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

    if len(points) != 2:
        return None

    return (
        barycentric_to_cartesian(*points[0].tolist()),
        barycentric_to_cartesian(*points[1].tolist()),
    )


def render_simplex_face(
    entries: list[list[float]],
    candidates: list[int],
    selected_goal_indices: tuple[int, int, int],
    onestep_query: OnestepQuery | None,
    depth_one_query: MultistepQuery | None,
    step: float,
) -> plt.Figure:
    face_grid = generate_simplex_grid(step)
    best_alternatives = compute_best_alternatives_on_face(
        entries=entries,
        face_weights=face_grid,
        selected_goal_indices=selected_goal_indices,
    )
    grid_points = np.array(
        [barycentric_to_cartesian(*weights.tolist()) for weights in face_grid],
        dtype=float,
    )
    colors = get_alternative_colors(len(entries))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    triangle_vertices = np.array(
        [
            barycentric_to_cartesian(1.0, 0.0, 0.0),
            barycentric_to_cartesian(0.0, 1.0, 0.0),
            barycentric_to_cartesian(0.0, 0.0, 1.0),
        ]
    )
    ax.add_patch(
        Polygon(
            triangle_vertices,
            closed=True,
            facecolor="#f8f9fb",
            edgecolor="#222222",
            linewidth=1.6,
        )
    )

    for alternative_index in range(len(entries)):
        mask = best_alternatives == alternative_index
        if not np.any(mask):
            continue

        alpha = 0.30 if alternative_index in candidates else 0.10
        ax.scatter(
            grid_points[mask, 0],
            grid_points[mask, 1],
            s=12,
            c=colors[alternative_index],
            marker="s",
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    line_specs = [
        (onestep_query, "#111111", "One-Step"),
        (depth_one_query, "#d62728", "V1"),
    ]
    for query, color, label in line_specs:
        if query is None:
            continue

        segment = get_query_line_segment_on_face(
            query=query,
            selected_goal_indices=selected_goal_indices,
        )
        if segment is None:
            continue

        (x_1, y_1), (x_2, y_2) = segment
        ax.plot(
            [x_1, x_2],
            [y_1, y_2],
            color=color,
            linewidth=2.2,
            linestyle="--" if label == "One-Step" else "-",
            label=f"{label}: {format_query(query)}",
        )

    goal_positions = [
        barycentric_to_cartesian(1.0, 0.0, 0.0),
        barycentric_to_cartesian(0.0, 1.0, 0.0),
        barycentric_to_cartesian(0.0, 0.0, 1.0),
    ]
    goal_offsets = [(-0.05, -0.04), (0.01, -0.04), (-0.04, 0.025)]
    for (x_value, y_value), (x_offset, y_offset), goal_index in zip(
        goal_positions,
        goal_offsets,
        selected_goal_indices,
    ):
        ax.text(
            x_value + x_offset,
            y_value + y_offset,
            f"Ziel {goal_index}",
            fontsize=11,
            fontweight="bold",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=f"A{candidate_index}",
            markerfacecolor=colors[candidate_index],
            markersize=8,
        )
        for candidate_index in candidates
    ]
    if ax.get_legend_handles_labels()[0]:
        query_handles, query_labels = ax.get_legend_handles_labels()
        legend_handles.extend(query_handles)
        ax.legend(
            handles=legend_handles,
            labels=[handle.get_label() for handle in legend_handles[:-len(query_handles)]] + query_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=True,
        )
    else:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=True,
        )

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, SQRT3 / 2.0 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.98, bottom=0.03)
    return fig


@st.cache_data(show_spinner=True)
def compute_mismatch_data(
    sample_count: int,
    burn_in: int,
    thinning: int,
    query_epsilon: float,
) -> dict[str, Any]:
    goal_count, alternative_count, entries = generate_problem(
        problem_index=PROBLEM_INDEX,
        seed=PROBLEM_SEED,
    )
    onestep_alternatives = OnestepAlternativenMatrix(entries=entries)
    multistep_alternatives = MultistepAlternativenMatrix(entries=entries)

    onestep_weight_space = build_W(goal_count, [])
    onestep_candidates = [
        alternative_index
        for alternative_index in range(alternative_count)
        if build_optimal_region_in_W(
            onestep_alternatives,
            onestep_weight_space,
            alternative_index,
        ).is_feasible()
    ]
    onestep_ratio_intervals = compute_all_ratio_intervals(
        alternativen_matrix=onestep_alternatives,
        answered_queries=[],
        kandidatenmenge=onestep_candidates,
    )
    onestep_query_candidates = compute_all_query_kandidaten(
        zielpaar_intervalle_liste=onestep_ratio_intervals,
        epsilon=query_epsilon,
    )
    onestep_samples = sample_points_from_ungleichungssystem(
        onestep_weight_space,
        num_samples=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        seed=PROBLEM_SEED,
    )
    interval_lookup = build_zielpaar_intervalle_lookup(onestep_ratio_intervals)
    onestep_query_infos = [
        query_info
        for query in onestep_query_candidates
        if (
            query_info := compute_query_info(
                query=query,
                samples=onestep_samples,
                zielpaar_intervalle_lookup=interval_lookup,
            )
        )
        is not None
    ]
    informative_onestep_query_infos = filter_informative_query_infos(
        query_infos=onestep_query_infos,
        kandidatenmenge=set(onestep_candidates),
    )
    onestep_best_info = min(
        informative_onestep_query_infos,
        key=lambda query_info: query_info.expected_kandidatenanzahl,
    )

    depth_one_result = compute_value_function(
        alternatives=multistep_alternatives,
        answered_queries=[],
        remaining_depth=1,
        config=MultistepConfig(
            sample_count=sample_count,
            burn_in=burn_in,
            thinning=thinning,
            random_seed=PROBLEM_SEED,
            query_epsilon=query_epsilon,
        ),
    )
    multistep_weight_space = build_weight_space(goal_count=goal_count, answered_queries=[])
    multistep_candidates = compute_candidate_set(
        alternatives=multistep_alternatives,
        weight_space=multistep_weight_space,
    )

    return {
        "goal_count": goal_count,
        "alternative_count": alternative_count,
        "entries": entries,
        "onestep_candidates": onestep_candidates,
        "multistep_candidates": multistep_candidates,
        "onestep_query_infos": onestep_query_infos,
        "informative_onestep_query_infos": informative_onestep_query_infos,
        "onestep_best_query": onestep_best_info.query,
        "onestep_best_expected_value": onestep_best_info.expected_kandidatenanzahl,
        "depth_one_result": depth_one_result,
        "onestep_ratio_intervals": onestep_ratio_intervals,
    }


def build_query_comparison_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    depth_evaluations_by_key = {
        query_key(evaluation.query): evaluation
        for evaluation in data["depth_one_result"].query_evaluations
    }
    rows = []

    for query_info in data["onestep_query_infos"]:
        query = query_info.query
        depth_evaluation = depth_evaluations_by_key.get(query_key(query))
        depth_branches = {
            branch.answer: branch
            for branch in depth_evaluation.branches
        } if depth_evaluation is not None else {}

        rows.append(
            {
                "query": format_query(query),
                "one_step_E": query_info.expected_kandidatenanzahl,
                "depth1_E": None if depth_evaluation is None else depth_evaluation.expected_value,
                "diff": None if depth_evaluation is None else depth_evaluation.expected_value - query_info.expected_kandidatenanzahl,
                "old_p_<": query_info.p_kleiner,
                "old_p_>": query_info.p_groesser,
                "old_K_<": len(query_info.kandidatenmenge_nach_kleiner_antwort),
                "old_K_>": len(query_info.kandidatenmenge_nach_groesser_antwort),
                "new_p_<": None if "<" not in depth_branches else depth_branches["<"].probability,
                "new_p_=": None if "=" not in depth_branches else depth_branches["="].probability,
                "new_p_>": None if ">" not in depth_branches else depth_branches[">"].probability,
                "new_K_<": None if "<" not in depth_branches else depth_branches["<"].child_candidate_count,
                "new_K_=": None if "=" not in depth_branches else depth_branches["="].child_candidate_count,
                "new_K_>": None if ">" not in depth_branches else depth_branches[">"].child_candidate_count,
            }
        )

    return pd.DataFrame(rows).sort_values(["one_step_E", "depth1_E"], na_position="last")


def find_onestep_info_for_query(data: dict[str, Any], query: OnestepQuery | MultistepQuery):
    for query_info in data["onestep_query_infos"]:
        if query_key(query_info.query) == query_key(query):
            return query_info

    return None


def find_depth_evaluation_for_query(data: dict[str, Any], query: OnestepQuery | MultistepQuery):
    for evaluation in data["depth_one_result"].query_evaluations:
        if query_key(evaluation.query) == query_key(query):
            return evaluation

    return None


def compute_multistep_child_candidates(
    entries: list[list[float]],
    query: OnestepQuery | MultistepQuery,
) -> dict[str, list[int]]:
    alternatives = MultistepAlternativenMatrix(entries=entries)
    result: dict[str, list[int]] = {}
    multistep_query = MultistepQuery(
        ziel_index_a=int(query.ziel_index_a),
        ziel_index_b=int(query.ziel_index_b),
        value=float(query.value),
    )

    for answer in ANSWER_OPTIONS:
        child_weight_space = build_weight_space(
            goal_count=alternatives.get_anzahl_spalten(),
            answered_queries=[multistep_query.answer(answer)],
        )
        if not child_weight_space.is_feasible():
            result[answer] = []
            continue

        result[answer] = compute_candidate_set(
            alternatives=alternatives,
            weight_space=child_weight_space,
        )

    return result


def build_interval_dataframe(data: dict[str, Any], goal_pair: tuple[int, int]) -> pd.DataFrame:
    for goal_pair_intervals in data["onestep_ratio_intervals"]:
        if (
            int(goal_pair_intervals.ziel_index_a),
            int(goal_pair_intervals.ziel_index_b),
        ) != goal_pair:
            continue

        rows = []
        for candidate_index, interval in goal_pair_intervals.intervalle_pro_kandidat.items():
            rows.append(
                {
                    "candidate": int(candidate_index),
                    "lower_status": interval.lower.status,
                    "alpha": interval.lower.optimal_value,
                    "upper_status": interval.upper.status,
                    "beta": interval.upper.optimal_value,
                }
            )
        return pd.DataFrame(rows).sort_values("candidate")

    return pd.DataFrame()


def main() -> None:
    st.set_page_config(page_title="Depth-1 Mismatch", layout="wide")
    st.title("Depth-1 Mismatch Analyse")

    with st.sidebar:
        sample_count = int(
            st.number_input(
                "Samples",
                min_value=100,
                max_value=50_000,
                value=DEFAULT_SAMPLE_COUNT,
                step=500,
            )
        )
        burn_in = int(
            st.number_input(
                "Burn-in",
                min_value=0,
                max_value=5_000,
                value=DEFAULT_BURN_IN,
                step=50,
            )
        )
        thinning = int(
            st.number_input(
                "Thinning",
                min_value=1,
                max_value=50,
                value=DEFAULT_THINNING,
                step=1,
            )
        )
        query_epsilon = float(
            st.number_input(
                "Query Epsilon",
                min_value=1e-6,
                max_value=0.1,
                value=DEFAULT_QUERY_EPSILON,
                step=1e-3,
                format="%.6f",
            )
        )
        simplex_step = float(
            st.number_input(
                "Simplex Grid",
                min_value=0.005,
                max_value=0.05,
                value=DEFAULT_SIMPLEX_STEP,
                step=0.005,
                format="%.3f",
            )
        )

    data = compute_mismatch_data(
        sample_count=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        query_epsilon=query_epsilon,
    )
    entries = data["entries"]
    goal_count = data["goal_count"]
    onestep_best_query = data["onestep_best_query"]
    depth_one_best_query = data["depth_one_result"].best_query

    st.caption(
        "Reproduziert Vergleichsproblem idx=0 mit seed=1 aus "
        "`compare_depth1_with_onestep.py`. Das Problem hat 5 Ziele; der Simplex "
        "zeigt deshalb eine auswählbare 3-Ziel-Face mit allen anderen Gewichten 0."
    )

    summary_columns = st.columns(4)
    summary_columns[0].metric("Ziele", data["goal_count"])
    summary_columns[1].metric("Alternativen", data["alternative_count"])
    summary_columns[2].metric("One-Step E", f"{data['onestep_best_expected_value']:.6g}")
    summary_columns[3].metric("Depth-1 E", f"{data['depth_one_result'].value:.6g}")

    query_columns = st.columns(2)
    with query_columns[0]:
        st.subheader("One-Step")
        st.write("Beste Query:", format_query(onestep_best_query))
        st.write("Kandidaten:", data["onestep_candidates"])
    with query_columns[1]:
        st.subheader("Rekursive V1")
        st.write("Beste Query:", format_query(depth_one_best_query))
        st.write("Kandidaten:", data["multistep_candidates"])

    st.subheader("Handlungsalternativenmatrix")
    st.dataframe(
        pd.DataFrame(
            entries,
            index=[f"A{i}" for i in range(len(entries))],
            columns=[f"Ziel {i}" for i in range(goal_count)],
        ),
        width="stretch",
    )

    st.subheader("Simplex-Face")
    default_goals = [0, 3, 1]
    selected_goals = st.multiselect(
        "Angezeigte Ziele",
        options=list(range(goal_count)),
        default=default_goals,
        max_selections=3,
    )
    if len(selected_goals) != 3:
        st.warning("Bitte genau drei Ziele auswählen.")
    else:
        fig = render_simplex_face(
            entries=entries,
            candidates=data["multistep_candidates"],
            selected_goal_indices=tuple(selected_goals),
            onestep_query=onestep_best_query,
            depth_one_query=depth_one_best_query,
            step=simplex_step,
        )
        st.pyplot(fig, clear_figure=True)

    st.subheader("Query-Bewertungen")
    comparison_dataframe = build_query_comparison_dataframe(data)
    st.dataframe(
        comparison_dataframe,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Branch-Details")
    query_options = {
        f"One-Step best {format_query(onestep_best_query)}": onestep_best_query,
        f"Depth-1 best {format_query(depth_one_best_query)}": depth_one_best_query,
    }
    selected_query_label = st.selectbox(
        "Query",
        options=list(query_options),
    )
    selected_query = query_options[selected_query_label]
    if selected_query is not None:
        onestep_info = find_onestep_info_for_query(data, selected_query)
        depth_evaluation = find_depth_evaluation_for_query(data, selected_query)
        child_candidates = compute_multistep_child_candidates(
            entries=entries,
            query=selected_query,
        )

        branch_rows = []
        if onestep_info is not None:
            branch_rows.extend(
                [
                    {
                        "method": "one_step",
                        "answer": "<",
                        "probability": onestep_info.p_kleiner,
                        "candidate_count": len(onestep_info.kandidatenmenge_nach_kleiner_antwort),
                        "candidates": sorted(onestep_info.kandidatenmenge_nach_kleiner_antwort),
                    },
                    {
                        "method": "one_step",
                        "answer": ">",
                        "probability": onestep_info.p_groesser,
                        "candidate_count": len(onestep_info.kandidatenmenge_nach_groesser_antwort),
                        "candidates": sorted(onestep_info.kandidatenmenge_nach_groesser_antwort),
                    },
                ]
            )
        if depth_evaluation is not None:
            for branch in depth_evaluation.branches:
                branch_rows.append(
                    {
                        "method": "depth_1",
                        "answer": branch.answer,
                        "probability": branch.probability,
                        "candidate_count": branch.child_candidate_count,
                        "candidates": child_candidates[branch.answer],
                    }
                )

        st.dataframe(pd.DataFrame(branch_rows), width="stretch", hide_index=True)

        goal_pair = (int(selected_query.ziel_index_a), int(selected_query.ziel_index_b))
        st.subheader(f"Ratio-Intervalle für Zielpaar {goal_pair}")
        st.dataframe(
            build_interval_dataframe(data=data, goal_pair=goal_pair),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
