from __future__ import annotations

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

from multistep.optimized import (
    OptimizedMultistepConfig,
    compute_ratio_relevant_candidate_set,
    compute_value_function_optimized,
)
from multistep.optimized.value_function import is_query_already_answered
from multistep.src.candidates import compute_candidate_set
from multistep.src.linear_constraints import LinearConstraintSystem
from multistep.src.models import AlternativenMatrix, AnsweredQuery, Query
from multistep.src.query_probability import classify_query_answer
from multistep.src.value_function import ValueFunctionResult
from multistep.src.weight_space import build_weight_space


SQRT3 = math.sqrt(3.0)
GOAL_LABELS = ["Spaß", "Finanzen", "Sicherheit"]
DEFAULT_ALTERNATIVE_COUNT = 10
DEFAULT_SAMPLE_COUNT = 400
DEFAULT_BURN_IN = 200
DEFAULT_THINNING = 5
DEFAULT_GRID_SIZE = 21
DEFAULT_MAX_QUERY_VALUE = 100.0
DEFAULT_SIMPLEX_STEP = 0.01
DEFAULT_MIN_QUERY_VALUE = 1e-3


@dataclass(frozen=True)
class VisualizationData:
    goal_labels: list[str]
    alternative_names: list[str]
    entries: list[list[float]]
    target_weights: list[float] | None


def initialize_state() -> None:
    if "multistep_three_goals_alternative_names" in st.session_state:
        return

    alternative_names, utilities = generate_random_alternatives(
        DEFAULT_ALTERNATIVE_COUNT
    )
    st.session_state.multistep_three_goals_alternative_names = alternative_names
    st.session_state.multistep_three_goals_utilities = utilities
    st.session_state.multistep_three_goals_target_weights = (
        generate_random_target_weights()
    )
    st.session_state.multistep_three_goals_answered_queries = []
    st.session_state.multistep_three_goals_algorithm_requested = False
    st.session_state.multistep_three_goals_selected_query_index = 0


def reset_algorithm_state() -> None:
    st.session_state.multistep_three_goals_algorithm_requested = False
    st.session_state.multistep_three_goals_selected_query_index = 0


def regenerate_problem(alternative_count: int) -> None:
    alternative_names, utilities = generate_random_alternatives(alternative_count)
    st.session_state.multistep_three_goals_alternative_names = alternative_names
    st.session_state.multistep_three_goals_utilities = utilities
    st.session_state.multistep_three_goals_target_weights = (
        generate_random_target_weights()
    )
    st.session_state.multistep_three_goals_answered_queries = []
    reset_algorithm_state()


def generate_random_alternatives(alternative_count: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng()
    alternative_names = [f"A{index}" for index in range(1, alternative_count + 1)]
    utilities = np.round(
        rng.uniform(0.0, 1.0, size=(alternative_count, 3)),
        2,
    )
    return alternative_names, utilities


def generate_random_target_weights() -> np.ndarray:
    rng = np.random.default_rng()
    return rng.dirichlet(alpha=[1.0, 1.0, 1.0]).astype(float)


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


def get_alternative_colors(alternative_count: int) -> list[str]:
    cmap = plt.get_cmap("nipy_spectral")
    return [
        mcolors.to_hex(cmap(position))
        for position in np.linspace(0.03, 0.97, alternative_count)
    ]


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


def get_query_line_segment(
    query: Query | AnsweredQuery,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
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
    visualization_data: VisualizationData,
    weight_space: LinearConstraintSystem,
    candidates: list[int],
    selected_query: Query | None,
    simplex_step: float,
) -> plt.Figure:
    grid = generate_simplex_grid(simplex_step)
    grid_points = np.array(
        [barycentric_to_cartesian(*weights.tolist()) for weights in grid],
        dtype=float,
    )
    best_alternatives = compute_best_alternatives(
        entries=visualization_data.entries,
        weights=grid,
    )
    in_weight_space = np.array(
        [
            point_satisfies_system(weights=weights, system=weight_space)
            for weights in grid
        ],
        dtype=bool,
    )
    colors = get_alternative_colors(len(visualization_data.entries))

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
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

    for alternative_index in range(len(visualization_data.entries)):
        mask = in_weight_space & (best_alternatives == alternative_index)
        if not np.any(mask):
            continue

        alpha = 0.50 if alternative_index in candidates else 0.16
        ax.scatter(
            grid_points[mask, 0],
            grid_points[mask, 1],
            s=11,
            c=colors[alternative_index],
            marker="s",
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    for answered_query in st.session_state.multistep_three_goals_answered_queries:
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

    if selected_query is not None:
        segment = get_query_line_segment(selected_query)
        if segment is not None:
            (x_1, y_1), (x_2, y_2) = segment
            ax.plot(
                [x_1, x_2],
                [y_1, y_2],
                color="#d62728",
                linewidth=2.6,
                label=f"Tiefe 2: {query_to_text(selected_query)}",
            )

    if visualization_data.target_weights is not None:
        target_x, target_y = barycentric_to_cartesian(
            *visualization_data.target_weights
        )
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
        visualization_data.goal_labels,
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
            label=visualization_data.alternative_names[candidate_index],
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
    fig.subplots_adjust(left=0.03, right=0.68, top=0.98, bottom=0.03)
    return fig


def query_to_text(query: Query) -> str:
    return (
        f"{GOAL_LABELS[int(query.ziel_index_a)]} ? "
        f"{float(query.value):.8g} * {GOAL_LABELS[int(query.ziel_index_b)]}"
    )


def answered_query_to_text(answered_query: AnsweredQuery) -> str:
    return (
        f"{GOAL_LABELS[int(answered_query.ziel_index_a)]} "
        f"{answered_query.operator} {float(answered_query.value):.8g} * "
        f"{GOAL_LABELS[int(answered_query.ziel_index_b)]}"
    )


def build_algorithm_signature(
    entries: list[list[float]],
    answered_queries: list[AnsweredQuery],
    sample_count: int,
    burn_in: int,
    thinning: int,
    random_seed: int,
    grid_size: int,
    max_query_value: float,
    use_conditioned_samples: bool,
    canonical_grid: bool,
) -> tuple[Any, ...]:
    return (
        tuple(tuple(float(value) for value in row) for row in entries),
        tuple(
            (
                int(query.ziel_index_a),
                int(query.ziel_index_b),
                float(query.value),
                query.operator,
            )
            for query in answered_queries
        ),
        sample_count,
        burn_in,
        thinning,
        random_seed,
        grid_size,
        max_query_value,
        use_conditioned_samples,
        canonical_grid,
    )


@st.cache_data(show_spinner=True)
def compute_depth_two_result_cached(
    signature: tuple[Any, ...],
) -> dict[str, Any]:
    (
        entries_signature,
        answered_queries_signature,
        sample_count,
        burn_in,
        thinning,
        random_seed,
        grid_size,
        max_query_value,
        use_conditioned_samples,
        canonical_grid,
    ) = signature
    entries = [list(row) for row in entries_signature]
    answered_queries = [
        AnsweredQuery(
            ziel_index_a=int(ziel_index_a),
            ziel_index_b=int(ziel_index_b),
            value=float(value),
            operator=operator,
        )
        for ziel_index_a, ziel_index_b, value, operator in answered_queries_signature
    ]
    alternatives = AlternativenMatrix(entries=entries)
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    closed_candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    candidates = compute_ratio_relevant_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    result = compute_value_function_optimized(
        alternatives=alternatives,
        answered_queries=answered_queries,
        remaining_depth=2,
        config=OptimizedMultistepConfig(
            sample_count=int(sample_count),
            burn_in=int(burn_in),
            thinning=int(thinning),
            random_seed=int(random_seed),
            grid_size=int(grid_size),
            min_query_value=DEFAULT_MIN_QUERY_VALUE,
            max_query_value=float(max_query_value),
            skip_zero_probability_branches=True,
            pass_candidate_subset=True,
            reuse_conditioned_samples=bool(use_conditioned_samples),
            use_ratio_terminal_counts=True,
            canonical_grid_goal_pairs_only=bool(canonical_grid),
            parallelize_root=True,
            max_workers=4,
            candidate_count_mode="ratio_relevant",
            include_ratio_queries_on_grid_depths=True,
        ),
    )
    return {
        "weight_space": weight_space,
        "candidates": candidates,
        "closed_candidates": closed_candidates,
        "result": result,
    }


def build_query_evaluations_dataframe(result: ValueFunctionResult) -> pd.DataFrame:
    rows = []
    for evaluation in result.query_evaluations:
        branch_by_answer = {
            branch.answer: branch
            for branch in evaluation.branches
        }
        rows.append(
            {
                "query": query_to_text(evaluation.query),
                "source": evaluation.query_source,
                "E": evaluation.expected_value,
                "p_<": branch_by_answer["<"].probability,
                "V_<": branch_by_answer["<"].child_value,
                "N_<": branch_by_answer["<"].child_candidate_count,
                "p_=": branch_by_answer["="].probability,
                "V_=": branch_by_answer["="].child_value,
                "N_=": branch_by_answer["="].child_candidate_count,
                "p_>": branch_by_answer[">"].probability,
                "V_>": branch_by_answer[">"].child_value,
                "N_>": branch_by_answer[">"].child_candidate_count,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("E", kind="stable").reset_index(drop=True)


def build_best_query_source_dataframe(result: ValueFunctionResult) -> pd.DataFrame:
    rows = []
    source_specs = [
        ("Beste Query gesamt", {"grid", "ratio", "grid+ratio", "unknown"}),
        ("Beste Grid-Query", {"grid", "grid+ratio"}),
        ("Beste Ratio-Query", {"ratio", "grid+ratio"}),
    ]
    for label, allowed_sources in source_specs:
        matching_evaluations = [
            evaluation
            for evaluation in result.query_evaluations
            if evaluation.query_source in allowed_sources
        ]
        if not matching_evaluations:
            continue

        best_evaluation = min(
            matching_evaluations,
            key=lambda evaluation: evaluation.expected_value,
        )
        rows.append(
            {
                "Typ": label,
                "Query": query_to_text(best_evaluation.query),
                "Quelle": best_evaluation.query_source,
                "E": best_evaluation.expected_value,
            }
        )

    return pd.DataFrame(rows)


def append_answer(operator: str, query: Query) -> None:
    answered_queries = list(st.session_state.multistep_three_goals_answered_queries)
    if is_query_already_answered(
        query=query,
        answered_queries=answered_queries,
    ):
        reset_algorithm_state()
        return

    answered_queries.append(query.answer(operator))
    st.session_state.multistep_three_goals_answered_queries = answered_queries
    reset_algorithm_state()


def append_target_answer(query: Query) -> None:
    target_weights = st.session_state.multistep_three_goals_target_weights
    if target_weights is None:
        return

    answer = classify_query_answer(
        weights=target_weights.tolist(),
        query=query,
        equality_tol=0.0,
    )
    append_answer(answer, query)


def main() -> None:
    st.set_page_config(page_title="Multistep 3 Goals", layout="wide")
    st.title("Multistep 3 Goals")
    initialize_state()

    with st.sidebar:
        st.header("Settings")
        show_handlungsalternativen = bool(
            st.toggle("Show handlungsalternativen", value=False)
        )
        simplex_step = float(
            st.slider(
                "Simplex grid",
                min_value=0.0025,
                max_value=0.05,
                value=DEFAULT_SIMPLEX_STEP,
                step=0.0025,
                format="%.4f",
            )
        )
        sample_count = int(
            st.number_input(
                "Samples",
                min_value=50,
                max_value=20_000,
                value=DEFAULT_SAMPLE_COUNT,
                step=50,
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
        canonical_grid = bool(
            st.toggle(
                "Canonical grid",
                value=True,
                help="Betrachtet fuer Grid-Ebenen nur eine Orientierung je Zielpaar.",
            )
        )
        use_conditioned_samples = bool(
            st.toggle(
                "Conditioned samples",
                value=False,
                help="Schneller, aber approximativer.",
            )
        )
        show_query_table = bool(st.toggle("Show query table", value=False))

    control_col_1, control_col_2, control_col_3 = st.columns([0.45, 0.275, 0.275])
    with control_col_1:
        current_alternative_count = len(
            st.session_state.multistep_three_goals_alternative_names
        )
        alternative_count = int(
            st.number_input(
                "Anzahl Handlungsalternativen",
                min_value=1,
                max_value=100,
                value=current_alternative_count,
                step=1,
            )
        )
    with control_col_2:
        st.write("")
        st.write("")
        if st.button("Generiere Daten", use_container_width=True):
            regenerate_problem(alternative_count)
            st.rerun()
    with control_col_3:
        st.write("")
        st.write("")
        if st.button("Generiere Zielgewichtsvektor", use_container_width=True):
            st.session_state.multistep_three_goals_target_weights = (
                generate_random_target_weights()
            )
            st.rerun()

    alternative_names: list[str] = (
        st.session_state.multistep_three_goals_alternative_names
    )
    utilities: np.ndarray = st.session_state.multistep_three_goals_utilities
    answered_queries: list[AnsweredQuery] = (
        st.session_state.multistep_three_goals_answered_queries
    )
    target_weights: np.ndarray | None = (
        st.session_state.multistep_three_goals_target_weights
    )
    entries = utilities.astype(float).tolist()
    alternatives = AlternativenMatrix(entries=entries)
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    if not weight_space.is_feasible():
        st.error("Der aktuelle Weight-Space ist infeasible.")
        return

    candidates = compute_ratio_relevant_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    closed_candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )
    visualization_data = VisualizationData(
        goal_labels=GOAL_LABELS,
        alternative_names=alternative_names,
        entries=entries,
        target_weights=None if target_weights is None else target_weights.tolist(),
    )

    algorithm_details: dict[str, Any] | None = None
    if st.session_state.multistep_three_goals_algorithm_requested:
        signature = build_algorithm_signature(
            entries=entries,
            answered_queries=answered_queries,
            sample_count=sample_count,
            burn_in=burn_in,
            thinning=thinning,
            random_seed=random_seed,
            grid_size=grid_size,
            max_query_value=max_query_value,
            use_conditioned_samples=use_conditioned_samples,
            canonical_grid=canonical_grid,
        )
        algorithm_details = compute_depth_two_result_cached(signature)

    selected_query = None
    if algorithm_details is not None:
        result: ValueFunctionResult = algorithm_details["result"]
        selected_query = result.best_query
        candidates = algorithm_details["candidates"]
        closed_candidates = algorithm_details["closed_candidates"]
        weight_space = algorithm_details["weight_space"]

    if target_weights is None:
        st.write("**Zielgewichtsvektor:** Noch keiner generiert.")
    else:
        st.write(
            f"**Zielgewichtsvektor:** {GOAL_LABELS[0]} = {target_weights[0]:.4f}, "
            f"{GOAL_LABELS[1]} = {target_weights[1]:.4f}, "
            f"{GOAL_LABELS[2]} = {target_weights[2]:.4f}"
        )

    if show_handlungsalternativen:
        st.markdown("**Handlungsalternativen**")
        st.dataframe(
            pd.DataFrame(
                utilities,
                index=alternative_names,
                columns=GOAL_LABELS,
            ),
            width="stretch",
        )

    figure = render_simplex(
        visualization_data=visualization_data,
        weight_space=weight_space,
        candidates=candidates,
        selected_query=selected_query,
        simplex_step=simplex_step,
    )
    simplex_column, _ = st.columns([0.62, 0.38])
    with simplex_column:
        st.pyplot(figure, clear_figure=True, use_container_width=False)
    st.caption(
        "Farben zeigen die aktuell optimalen Alternativen in W(T). "
        "Grau liegt ausserhalb des aktuellen Weight-Space."
    )

    st.markdown("**Bisher beantwortete Trade-off-Queries**")
    if answered_queries:
        for index, answered_query in enumerate(answered_queries, start=1):
            st.write(f"{index}. {answered_query_to_text(answered_query)}")
    else:
        st.write("Noch keine.")

    if not st.session_state.multistep_three_goals_algorithm_requested:
        if st.button("Run Tiefe-2 Algorithmus", use_container_width=True):
            st.session_state.multistep_three_goals_algorithm_requested = True
            st.rerun()
        return

    if algorithm_details is None:
        st.error("Der Algorithmus konnte nicht ausgewertet werden.")
        return

    result = algorithm_details["result"]
    metrics = st.columns(4)
    metrics[0].metric("Kandidaten", len(candidates))
    metrics[1].metric("Tiefe", result.remaining_depth)
    metrics[2].metric("Wert", f"{result.value:.6g}")
    metrics[3].metric("Query-Kandidaten", len(result.query_evaluations))
    if len(closed_candidates) != len(candidates):
        st.caption(
            f"LP-closed Kandidaten: {len(closed_candidates)}. "
            "Die optimierte Variante verwendet ratio-relevante Kandidaten."
        )

    if result.best_query is None:
        st.success("Keine weitere Query erforderlich.")
        return

    st.markdown("**Beste Tiefe-2-Query**")
    st.write(query_to_text(result.best_query))
    st.latex(rf"V_2(T) = {result.value:.6g}")
    st.dataframe(
        build_best_query_source_dataframe(result),
        width="stretch",
        hide_index=True,
    )

    best_evaluation = min(
        result.query_evaluations,
        key=lambda evaluation: evaluation.expected_value,
    )
    branch_rows = [
        {
            "ψ": branch.answer,
            "p_ψ": branch.probability,
            "V_ψ": branch.child_value,
            "N_ψ": branch.child_candidate_count,
            "feasible_ψ": branch.is_child_feasible,
        }
        for branch in best_evaluation.branches
    ]
    st.dataframe(pd.DataFrame(branch_rows), width="stretch", hide_index=True)
    st.caption(
        "ψ = Antwort auf die Query. "
        "p_ψ = geschätzte Wahrscheinlichkeit dieser Antwort. "
        "V_ψ = Wert des Folgezustands nach dieser Antwort. "
        "N_ψ = direkte Kandidatenanzahl nach dieser Antwort. "
        "feasible_ψ = ob der Folgezustand zulässig ist."
    )

    answer_left, answer_equal, answer_right, answer_target = st.columns(4)
    with answer_left:
        if st.button("< beantworten", use_container_width=True):
            append_answer("<", result.best_query)
            st.rerun()
    with answer_equal:
        if st.button("= beantworten", use_container_width=True):
            append_answer("=", result.best_query)
            st.rerun()
    with answer_right:
        if st.button("> beantworten", use_container_width=True):
            append_answer(">", result.best_query)
            st.rerun()
    with answer_target:
        if st.button("Nach Zielgewicht", use_container_width=True):
            append_target_answer(result.best_query)
            st.rerun()

    if show_query_table:
        st.markdown("**Query-Auswertung**")
        st.dataframe(
            build_query_evaluations_dataframe(result),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
