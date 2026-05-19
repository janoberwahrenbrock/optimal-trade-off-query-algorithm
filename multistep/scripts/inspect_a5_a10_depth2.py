from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
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
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query
from multistep.src.query_probability import ANSWER_OPTIONS
from multistep.src.value_function import MultistepConfig, compute_value_function
from multistep.src.weight_space import build_weight_space


DEFAULT_CASE_PATH = REPO_ROOT / "onestep" / "data" / "a5_a10_case.json"
DEFAULT_SAMPLE_COUNT = 400
DEFAULT_BURN_IN = 200
DEFAULT_THINNING = 5
DEFAULT_GRID_SIZE = 21
DEFAULT_QUERY_EPSILON = 1e-3
DEFAULT_MIN_QUERY_VALUE = 1e-3
DEFAULT_MAX_QUERY_VALUE = 100.0
DEFAULT_SIMPLEX_STEP = 0.0125
SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class CaseData:
    case_id: str
    goal_labels: list[str]
    alternative_names: list[str]
    entries: list[list[float]]
    answered_queries: list[AnsweredQuery]
    target_weights: list[float] | None


def load_case(path: Path) -> CaseData:
    with path.open("r", encoding="utf-8") as file:
        raw_case: dict[str, Any] = json.load(file)

    answered_queries = [
        AnsweredQuery(
            ziel_index_a=int(raw_query["ziel_index_a"]),
            ziel_index_b=int(raw_query["ziel_index_b"]),
            value=float(raw_query["value"]),
            operator=raw_query["operator"],
        )
        for raw_query in raw_case.get("tradeoffs", [])
    ]
    target_weights = raw_case.get("target_weights", {}).get("vector")
    if target_weights is not None:
        target_weights = [float(value) for value in target_weights]

    return CaseData(
        case_id=str(raw_case.get("id", path.stem)),
        goal_labels=[str(label) for label in raw_case["goal_labels"]],
        alternative_names=[str(name) for name in raw_case["alternative_names"]],
        entries=[
            [float(value) for value in alternative]
            for alternative in raw_case["handlungsalternativenmatrix"]
        ],
        answered_queries=answered_queries,
        target_weights=target_weights,
    )


def format_query(query: Query | None, goal_labels: list[str]) -> str:
    if query is None:
        return "-"

    goal_a = goal_labels[int(query.ziel_index_a)]
    goal_b = goal_labels[int(query.ziel_index_b)]
    return f"{goal_a} ? {float(query.value):.8g} * {goal_b}"


def format_answered_query(answered_query: AnsweredQuery, goal_labels: list[str]) -> str:
    goal_a = goal_labels[int(answered_query.ziel_index_a)]
    goal_b = goal_labels[int(answered_query.ziel_index_b)]
    return (
        f"{goal_a} {answered_query.operator} "
        f"{float(answered_query.value):.8g} * {goal_b}"
    )


def query_to_key(query: Query) -> tuple[int, int, float]:
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


def compute_best_alternatives(
    entries: list[list[float]],
    weights: np.ndarray,
) -> np.ndarray:
    utilities = np.array(entries, dtype=float)
    total_utilities = utilities @ weights.T
    return np.argmax(total_utilities, axis=0).astype(int)


def point_satisfies_system(
    weights: np.ndarray,
    system: LinearConstraintSystem,
    tolerance: float = 1e-8,
) -> bool:
    for left_side, right_side in zip(
        system.inequalities_left_side,
        system.inequalities_right_side,
    ):
        if float(np.dot(left_side, weights)) > float(right_side) + tolerance:
            return False

    for left_side, right_side in zip(
        system.equalities_left_side,
        system.equalities_right_side,
    ):
        if abs(float(np.dot(left_side, weights)) - float(right_side)) > tolerance:
            return False

    return True


def get_query_line_segment(query: Query | AnsweredQuery) -> tuple[tuple[float, float], tuple[float, float]] | None:
    simplex_vertices = np.eye(3)
    edge_indices = [(0, 1), (1, 2), (2, 0)]
    points: list[np.ndarray] = []

    def boundary_value(local_weights: np.ndarray) -> float:
        return float(
            local_weights[int(query.ziel_index_a)]
            - float(query.value) * local_weights[int(query.ziel_index_b)]
        )

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

        if start_value * end_value < 0.0:
            interpolation = start_value / (start_value - end_value)
            add_point(start + interpolation * (end - start))

    if len(points) != 2:
        return None

    return (
        barycentric_to_cartesian(*points[0].tolist()),
        barycentric_to_cartesian(*points[1].tolist()),
    )


def render_simplex(
    case_data: CaseData,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    depth_one_query: Query | None,
    depth_two_query: Query | None,
    simplex_step: float,
) -> plt.Figure:
    grid = generate_simplex_grid(simplex_step)
    grid_points = np.array(
        [barycentric_to_cartesian(*weights.tolist()) for weights in grid],
        dtype=float,
    )
    best_alternatives = compute_best_alternatives(case_data.entries, grid)
    in_weight_space = np.array(
        [
            point_satisfies_system(
                weights=weights,
                system=weight_space,
            )
            for weights in grid
        ],
        dtype=bool,
    )
    colors = get_alternative_colors(len(case_data.entries))

    fig, ax = plt.subplots(figsize=(7.0, 5.7))
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
            facecolor="#f7f8fb",
            edgecolor="#222222",
            linewidth=1.6,
        )
    )
    ax.scatter(
        grid_points[~in_weight_space, 0],
        grid_points[~in_weight_space, 1],
        s=9,
        c="#d8dbe2",
        marker="s",
        alpha=0.25,
        linewidths=0,
        rasterized=True,
    )

    for alternative_index in range(len(case_data.entries)):
        mask = in_weight_space & (best_alternatives == alternative_index)
        if not np.any(mask):
            continue

        alpha = 0.45 if alternative_index in candidates else 0.16
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

    for answered_query in case_data.answered_queries:
        segment = get_query_line_segment(answered_query)
        if segment is None:
            continue
        (x_1, y_1), (x_2, y_2) = segment
        ax.plot(
            [x_1, x_2],
            [y_1, y_2],
            color="#6f7785",
            linewidth=1.0,
            linestyle=":",
            alpha=0.85,
        )

    query_specs = [
        (depth_one_query, "#111111", "--", "Tiefe 1"),
        (depth_two_query, "#d62728", "-", "Tiefe 2"),
    ]
    for query, color, linestyle, label in query_specs:
        if query is None:
            continue

        segment = get_query_line_segment(query)
        if segment is None:
            continue

        (x_1, y_1), (x_2, y_2) = segment
        ax.plot(
            [x_1, x_2],
            [y_1, y_2],
            color=color,
            linewidth=2.4,
            linestyle=linestyle,
            label=f"{label}: {format_query(query, case_data.goal_labels)}",
        )

    if case_data.target_weights is not None:
        target_x, target_y = barycentric_to_cartesian(*case_data.target_weights)
        ax.scatter(
            [target_x],
            [target_y],
            s=90,
            c="#000000",
            marker="o",
            edgecolors="#ffffff",
            linewidths=1.4,
            zorder=10,
            label="Zielgewicht",
        )

    goal_positions = [
        barycentric_to_cartesian(1.0, 0.0, 0.0),
        barycentric_to_cartesian(0.0, 1.0, 0.0),
        barycentric_to_cartesian(0.0, 0.0, 1.0),
    ]
    goal_offsets = [(-0.07, -0.04), (0.01, -0.04), (-0.06, 0.03)]
    for (x_value, y_value), (x_offset, y_offset), goal_label in zip(
        goal_positions,
        goal_offsets,
        case_data.goal_labels,
    ):
        ax.text(
            x_value + x_offset,
            y_value + y_offset,
            goal_label,
            fontsize=11,
            fontweight="bold",
        )

    candidate_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=case_data.alternative_names[candidate_index],
            markerfacecolor=colors[candidate_index],
            markersize=8,
        )
        for candidate_index in candidates
    ]
    query_handles, query_labels = ax.get_legend_handles_labels()
    if candidate_handles or query_handles:
        ax.legend(
            handles=candidate_handles + query_handles,
            labels=[handle.get_label() for handle in candidate_handles] + query_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=True,
        )

    ax.set_xlim(-0.09, 1.09)
    ax.set_ylim(-0.08, SQRT3 / 2.0 + 0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.98, bottom=0.03)
    return fig


@st.cache_data(show_spinner=True)
def compute_case_analysis(
    sample_count: int,
    burn_in: int,
    thinning: int,
    random_seed: int,
    grid_size: int,
    query_epsilon: float,
    min_query_value: float,
    max_query_value: float,
) -> dict[str, Any]:
    case_data = load_case(DEFAULT_CASE_PATH)
    alternatives = AlternativenMatrix(entries=case_data.entries)
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=case_data.answered_queries,
    )
    candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    config = MultistepConfig(
        sample_count=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        random_seed=random_seed,
        grid_size=grid_size,
        min_query_value=min_query_value,
        max_query_value=max_query_value,
        grid_spacing="log",
        query_epsilon=query_epsilon,
    )
    depth_one_result = compute_value_function(
        alternatives=alternatives,
        answered_queries=case_data.answered_queries,
        remaining_depth=1,
        config=config,
    )
    depth_two_result = compute_value_function(
        alternatives=alternatives,
        answered_queries=case_data.answered_queries,
        remaining_depth=2,
        config=config,
    )
    child_depth_one_results = {}
    if depth_two_result.best_query is not None:
        for answer in ANSWER_OPTIONS:
            child_answered_queries = case_data.answered_queries + [
                depth_two_result.best_query.answer(answer)
            ]
            child_weight_space = build_weight_space(
                goal_count=alternatives.get_anzahl_spalten(),
                answered_queries=child_answered_queries,
            )
            if child_weight_space.is_feasible():
                child_depth_one_results[answer] = compute_value_function(
                    alternatives=alternatives,
                    answered_queries=child_answered_queries,
                    remaining_depth=1,
                    config=config,
                )
            else:
                child_depth_one_results[answer] = None

    return {
        "case_data": case_data,
        "weight_space": weight_space,
        "candidates": candidates,
        "depth_one_result": depth_one_result,
        "depth_two_result": depth_two_result,
        "child_depth_one_results": child_depth_one_results,
    }


def build_matrix_dataframe(case_data: CaseData) -> pd.DataFrame:
    return pd.DataFrame(
        case_data.entries,
        index=case_data.alternative_names,
        columns=case_data.goal_labels,
    )


def build_query_evaluations_dataframe(
    case_data: CaseData,
    result: Any,
    max_rows: int,
) -> pd.DataFrame:
    rows = []
    for evaluation in result.query_evaluations:
        branch_by_answer = {
            branch.answer: branch
            for branch in evaluation.branches
        }
        rows.append(
            {
                "query": format_query(evaluation.query, case_data.goal_labels),
                "E": evaluation.expected_value,
                "p_<": branch_by_answer["<"].probability,
                "V_<": branch_by_answer["<"].child_value,
                "p_=": branch_by_answer["="].probability,
                "V_=": branch_by_answer["="].child_value,
                "p_>": branch_by_answer[">"].probability,
                "V_>": branch_by_answer[">"].child_value,
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("E", kind="stable")
        .head(max_rows)
        .reset_index(drop=True)
    )


def build_best_branch_dataframe(
    case_data: CaseData,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    depth_two_result = analysis["depth_two_result"]
    child_results = analysis["child_depth_one_results"]
    if depth_two_result.best_query is None:
        return pd.DataFrame()

    best_evaluation = None
    best_query_key = query_to_key(depth_two_result.best_query)
    for evaluation in depth_two_result.query_evaluations:
        if query_to_key(evaluation.query) == best_query_key:
            best_evaluation = evaluation
            break

    if best_evaluation is None:
        return pd.DataFrame()

    rows = []
    for branch in best_evaluation.branches:
        child_result = child_results.get(branch.answer)
        rows.append(
            {
                "answer": branch.answer,
                "probability": branch.probability,
                "child_value": branch.child_value,
                "child_candidates": branch.child_candidate_count,
                "child_feasible": branch.is_child_feasible,
                "best_second_query": None
                if child_result is None
                else format_query(child_result.best_query, case_data.goal_labels),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="A5/A10 Tiefe 2", layout="wide")
    st.title("A5/A10 Case mit Tiefe 2")

    with st.sidebar:
        sample_count = int(
            st.number_input(
                "Samples",
                min_value=100,
                max_value=20_000,
                value=DEFAULT_SAMPLE_COUNT,
                step=100,
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
        random_seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                max_value=1_000_000,
                value=1,
                step=1,
            )
        )
        grid_size = int(
            st.slider(
                "Grid-Groesse fuer Tiefe 2",
                min_value=5,
                max_value=61,
                value=DEFAULT_GRID_SIZE,
                step=2,
            )
        )
        max_query_value = float(
            st.number_input(
                "Max s",
                min_value=1.0,
                max_value=500.0,
                value=DEFAULT_MAX_QUERY_VALUE,
                step=10.0,
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
                step=0.0025,
                format="%.4f",
            )
        )
        max_query_rows = int(
            st.slider(
                "Anzahl Query-Zeilen",
                min_value=5,
                max_value=100,
                value=25,
                step=5,
            )
        )

    analysis = compute_case_analysis(
        sample_count=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        random_seed=random_seed,
        grid_size=grid_size,
        query_epsilon=query_epsilon,
        min_query_value=DEFAULT_MIN_QUERY_VALUE,
        max_query_value=max_query_value,
    )
    case_data: CaseData = analysis["case_data"]
    depth_one_result = analysis["depth_one_result"]
    depth_two_result = analysis["depth_two_result"]

    st.caption(
        f"Geladen aus `{DEFAULT_CASE_PATH}`. Die vorhandenen Trade-offs werden "
        "als Zustand T verwendet; die rote Linie ist die beste erste Query fuer Tiefe 2."
    )

    metrics = st.columns(5)
    metrics[0].metric("Ziele", len(case_data.goal_labels))
    metrics[1].metric("Alternativen", len(case_data.entries))
    metrics[2].metric("Kandidaten", len(analysis["candidates"]))
    metrics[3].metric("Tiefe 1", f"{depth_one_result.value:.6g}")
    metrics[4].metric("Tiefe 2", f"{depth_two_result.value:.6g}")

    query_columns = st.columns(2)
    with query_columns[0]:
        st.subheader("Tiefe 1")
        st.write("Beste Query:", format_query(depth_one_result.best_query, case_data.goal_labels))
        st.write("Query-Kandidaten:", len(depth_one_result.query_evaluations))
    with query_columns[1]:
        st.subheader("Tiefe 2")
        st.write("Beste erste Query:", format_query(depth_two_result.best_query, case_data.goal_labels))
        st.write("Erste Query-Kandidaten:", len(depth_two_result.query_evaluations))

    st.subheader("Simplex")
    fig = render_simplex(
        case_data=case_data,
        weight_space=analysis["weight_space"],
        candidates=analysis["candidates"],
        depth_one_query=depth_one_result.best_query,
        depth_two_query=depth_two_result.best_query,
        simplex_step=simplex_step,
    )
    st.pyplot(fig, clear_figure=True)

    lower_columns = st.columns(2)
    with lower_columns[0]:
        st.subheader("Handlungsalternativen")
        st.dataframe(build_matrix_dataframe(case_data), width="stretch")

        st.subheader("Bisher beantwortete Trade-offs")
        st.dataframe(
            pd.DataFrame(
                {
                    "tradeoff": [
                        format_answered_query(query, case_data.goal_labels)
                        for query in case_data.answered_queries
                    ]
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with lower_columns[1]:
        st.subheader("Beste Tiefe-2-Query: Branches")
        st.dataframe(
            build_best_branch_dataframe(case_data=case_data, analysis=analysis),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Top Tiefe-2 Query-Bewertungen")
        st.dataframe(
            build_query_evaluations_dataframe(
                case_data=case_data,
                result=depth_two_result,
                max_rows=max_query_rows,
            ),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
