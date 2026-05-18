from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


N_GOALS = 4
GOAL_LABELS = ["Spass", "Finanzen", "Sicherheit", "Zeit"]
QUERY_EPSILON = 1e-3
VISUALIZATION_EQUALITY_TOL = 0.0025
DEFAULT_GRID_STEP = 0.05
DEFAULT_GRID_POINT_SIZE = 3.0
DEFAULT_TARGET_POINT_SIZE = 8.0
DEFAULT_SAMPLE_COUNT = 2500
DEFAULT_SAMPLE_BURN_IN = 300
DEFAULT_SAMPLE_THINNING = 4
VERTICES_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0, 0.0],
        [0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)],
    ],
    dtype=float,
)
TETRAHEDRON_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TETRAHEDRON_FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]


def generate_random_alternatives(n_alternatives: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng()
    alternative_names = [f"A{i}" for i in range(1, n_alternatives + 1)]
    utilities = np.round(rng.uniform(0.0, 1.0, size=(n_alternatives, N_GOALS)), 2)
    return alternative_names, utilities


def generate_random_target_weights() -> np.ndarray:
    rng = np.random.default_rng()
    return rng.dirichlet(alpha=np.ones(N_GOALS, dtype=float)).astype(float)


def weights_to_cartesian(weights: np.ndarray) -> np.ndarray:
    return np.asarray(weights, dtype=float) @ VERTICES_3D


@st.cache_data(show_spinner=False)
def generate_simplex_grid(step: float) -> np.ndarray:
    subdivisions = max(1, int(round(1.0 / step)))
    grid: list[list[float]] = []

    for w_1_units in range(subdivisions + 1):
        for w_2_units in range(subdivisions - w_1_units + 1):
            for w_3_units in range(subdivisions - w_1_units - w_2_units + 1):
                w_4_units = subdivisions - w_1_units - w_2_units - w_3_units
                grid.append(
                    [
                        w_1_units / subdivisions,
                        w_2_units / subdivisions,
                        w_3_units / subdivisions,
                        w_4_units / subdivisions,
                    ]
                )

    return np.array(grid, dtype=float)


def query_to_text(query: Query) -> str:
    left_label = GOAL_LABELS[query.ziel_index_a]
    right_label = GOAL_LABELS[query.ziel_index_b]
    return f"{left_label} ? {query.value:g} * {right_label}"


def answered_query_to_text(answered_query: AnsweredQuery) -> str:
    left_label = GOAL_LABELS[answered_query.ziel_index_a]
    right_label = GOAL_LABELS[answered_query.ziel_index_b]
    return (
        f"{left_label} {answered_query.operator} "
        f"{answered_query.value:g} * {right_label}"
    )


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


def build_answered_query_signature(
    answered_queries: list[AnsweredQuery],
) -> tuple[tuple[int, int, float, str], ...]:
    return tuple(
        (
            int(answered_query.ziel_index_a),
            int(answered_query.ziel_index_b),
            float(answered_query.value),
            answered_query.operator,
        )
        for answered_query in answered_queries
    )


def build_answered_queries_from_signature(
    signature: tuple[tuple[int, int, float, str], ...],
) -> list[AnsweredQuery]:
    return [
        AnsweredQuery(
            ziel_index_a=ziel_index_a,
            ziel_index_b=ziel_index_b,
            value=value,
            operator=operator,
        )
        for ziel_index_a, ziel_index_b, value, operator in signature
    ]


@st.cache_data(show_spinner=False)
def sample_probability_weights(
    answered_query_signature: tuple[tuple[int, int, float, str], ...],
    sample_count: int,
    sample_burn_in: int,
    sample_thinning: int,
    sampling_version: int,
) -> np.ndarray:
    del sampling_version
    answered_queries = build_answered_queries_from_signature(answered_query_signature)
    W = build_W(N_GOALS, answered_queries)
    if not W.is_feasible():
        raise ValueError("W(T) ist infeasible.")

    samples = sample_points_from_ungleichungssystem(
        W,
        num_samples=sample_count,
        burn_in=sample_burn_in,
        thinning=sample_thinning,
    )
    return np.array(samples, dtype=float)


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
        "kandidatenmenge": kandidatenmenge,
        "termination": None,
        "query_infos": [],
        "best_query": None,
        "best_expected": None,
        "samples_count": len(probability_samples),
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
    query_kandidaten = compute_all_query_kandidaten(
        zielpaar_intervalle,
        epsilon=QUERY_EPSILON,
    )
    query_kandidaten = filter_already_answered_queries(query_kandidaten, answered_queries)
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
        return state

    informative_query_infos.sort(
        key=lambda query_info: (
            query_info.expected_kandidatenanzahl,
            query_info.query.value,
        )
    )
    best_query_info = informative_query_infos[0]
    state["query_infos"] = informative_query_infos
    state["best_query"] = best_query_info.query
    state["best_expected"] = best_query_info.expected_kandidatenanzahl
    return state


def compute_best_alternatives(samples: np.ndarray, utilities: np.ndarray) -> np.ndarray:
    total_utilities = utilities @ samples.T
    return np.argmax(total_utilities, axis=0).astype(int)


def get_alternative_colors(n_alternatives: int) -> list[str]:
    if n_alternatives == 1:
        return ["#0057FF"]

    colorscale = [
        "#0057FF",
        "#FFB000",
        "#9B00FF",
        "#00B8FF",
        "#FF7A00",
        "#001A8D",
        "#FF00A8",
        "#00D5C8",
        "#6A3D00",
        "#BFC500",
        "#2B0057",
        "#7A8CFF",
    ]
    return [colorscale[index % len(colorscale)] for index in range(n_alternatives)]


def get_query_plane_points(query: Query) -> np.ndarray:
    barycentric_points: list[np.ndarray] = []
    simplex_vertices = np.eye(N_GOALS)

    def boundary_value(weights: np.ndarray) -> float:
        return float(weights[query.ziel_index_a] - query.value * weights[query.ziel_index_b])

    def add_point(point: np.ndarray) -> None:
        if np.any(point < -1e-9):
            return
        normalized = point / float(np.sum(point))
        for existing in barycentric_points:
            if np.allclose(existing, normalized, atol=1e-9):
                return
        barycentric_points.append(normalized)

    for start_index, end_index in TETRAHEDRON_EDGES:
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

    if len(barycentric_points) < 3:
        return np.empty((0, 3), dtype=float)

    points_3d = weights_to_cartesian(np.array(barycentric_points, dtype=float))
    return order_coplanar_points(points_3d)


def order_coplanar_points(points: np.ndarray) -> np.ndarray:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis_x = vh[0]
    basis_y = vh[1]
    projected = np.column_stack((centered @ basis_x, centered @ basis_y))
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    return points[np.argsort(angles)]


def add_query_plane(fig: go.Figure, query: Query) -> None:
    plane_points = get_query_plane_points(query)
    if len(plane_points) < 3:
        return

    triangle_i = []
    triangle_j = []
    triangle_k = []
    for index in range(1, len(plane_points) - 1):
        triangle_i.append(0)
        triangle_j.append(index)
        triangle_k.append(index + 1)

    fig.add_trace(
        go.Mesh3d(
            x=plane_points[:, 0],
            y=plane_points[:, 1],
            z=plane_points[:, 2],
            i=triangle_i,
            j=triangle_j,
            k=triangle_k,
            color="#111111",
            opacity=0.22,
            name="Ausgewaehlte Query",
            showscale=False,
            hoverinfo="skip",
        )
    )

    closed_points = np.vstack([plane_points, plane_points[0]])
    fig.add_trace(
        go.Scatter3d(
            x=closed_points[:, 0],
            y=closed_points[:, 1],
            z=closed_points[:, 2],
            mode="lines",
            line=dict(color="#111111", width=6),
            name="Query-Schnitt",
            hoverinfo="skip",
        )
    )


def draw_tetrahedron(
    grid: np.ndarray,
    utilities: np.ndarray,
    alternative_names: list[str],
    answered_queries: list[AnsweredQuery],
    target_weights: np.ndarray,
    selected_query: Query | None,
    grid_point_size: float,
    target_point_size: float,
    target_point_is_black: bool,
) -> go.Figure:
    grid_points = weights_to_cartesian(grid)
    best_indices = compute_best_alternatives(grid, utilities)
    feasible_mask = np.array(
        [
            satisfies_all_answered_queries_for_visualization(weights, answered_queries)
            for weights in grid
        ],
        dtype=bool,
    )
    colors = get_alternative_colors(len(alternative_names))

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=VERTICES_3D[:, 0],
            y=VERTICES_3D[:, 1],
            z=VERTICES_3D[:, 2],
            i=[face[0] for face in TETRAHEDRON_FACES],
            j=[face[1] for face in TETRAHEDRON_FACES],
            k=[face[2] for face in TETRAHEDRON_FACES],
            color="#E8EAEE",
            opacity=0.18,
            name="Simplex",
            hoverinfo="skip",
            showscale=False,
        )
    )

    for start, end in TETRAHEDRON_EDGES:
        edge_points = VERTICES_3D[[start, end]]
        fig.add_trace(
            go.Scatter3d(
                x=edge_points[:, 0],
                y=edge_points[:, 1],
                z=edge_points[:, 2],
                mode="lines",
                line=dict(color="#202124", width=4),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    excluded_points = grid_points[~feasible_mask]
    if excluded_points.size > 0:
        fig.add_trace(
            go.Scatter3d(
                x=excluded_points[:, 0],
                y=excluded_points[:, 1],
                z=excluded_points[:, 2],
                mode="markers",
                marker=dict(
                    size=grid_point_size,
                    color="#C9CDD4",
                    opacity=0.45,
                ),
                name="Ausgeschlossen",
                hovertemplate="Ausgeschlossen<extra></extra>",
            )
        )

    for alternative_index, alternative_name in enumerate(alternative_names):
        mask = feasible_mask & (best_indices == alternative_index)
        if not np.any(mask):
            continue
        points = grid_points[mask]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=grid_point_size,
                    color=colors[alternative_index],
                    opacity=0.72,
                ),
                name=alternative_name,
                hovertemplate=f"{alternative_name}<extra></extra>",
            )
        )

    if selected_query is not None:
        add_query_plane(fig, selected_query)

    target_point = weights_to_cartesian(target_weights.reshape(1, -1))[0]
    target_color = "#111111" if target_point_is_black else "#FFFFFF"
    fig.add_trace(
        go.Scatter3d(
            x=[target_point[0]],
            y=[target_point[1]],
            z=[target_point[2]],
            mode="markers",
            marker=dict(
                size=target_point_size,
                color=target_color,
                line=dict(color="#111111", width=2),
            ),
            name="Zielgewichtsvektor",
            hovertemplate="Zielgewichtsvektor<extra></extra>",
        )
    )

    for index, label in enumerate(GOAL_LABELS):
        vertex = VERTICES_3D[index]
        fig.add_trace(
            go.Scatter3d(
                x=[vertex[0]],
                y=[vertex[1]],
                z=[vertex[2]],
                mode="markers+text",
                marker=dict(size=4, color="#111111"),
                text=[label],
                textposition="top center",
                showlegend=False,
                hovertemplate=label + "<extra></extra>",
            )
        )

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.78, y=0.98),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
    )
    return fig


def build_query_dataframe(query_infos: list[QueryInfo], alternative_names: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, query_info in enumerate(query_infos, start=1):
        rows.append(
            {
                "Rang": index,
                "Query": query_to_text(query_info.query),
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
    return pd.DataFrame(
        [
            {
                "Kandidat": alternative_names[candidate_index],
                "Index": candidate_index,
                "Optimality share": termination.optimality_shares[candidate_index],
            }
            for candidate_index in termination.remaining_candidates
        ]
    )


def initialize_state(initial_n_alternatives: int) -> None:
    if "four_goals_alternative_names" in st.session_state:
        return

    alternative_names, utilities = generate_random_alternatives(initial_n_alternatives)
    apply_problem_state(
        alternative_names=alternative_names,
        utilities=utilities,
        target_weights=generate_random_target_weights(),
        answered_queries=[],
    )


def apply_problem_state(
    alternative_names: list[str],
    utilities: np.ndarray,
    target_weights: np.ndarray,
    answered_queries: list[AnsweredQuery],
) -> None:
    st.session_state.four_goals_alternative_names = alternative_names
    st.session_state.four_goals_utilities = utilities
    st.session_state.four_goals_target_weights = target_weights
    st.session_state.four_goals_answered_queries = answered_queries
    st.session_state.four_goals_algorithm_requested = False
    st.session_state.four_goals_sampling_version = 0


def reset_algorithm_state() -> None:
    st.session_state.four_goals_algorithm_requested = False
    st.session_state.four_goals_sampling_version += 1


def regenerate_problem(n_alternatives: int) -> None:
    alternative_names, utilities = generate_random_alternatives(n_alternatives)
    apply_problem_state(
        alternative_names=alternative_names,
        utilities=utilities,
        target_weights=generate_random_target_weights(),
        answered_queries=[],
    )


def append_answer(operator: str, query: Query) -> None:
    answered_queries = list(st.session_state.four_goals_answered_queries)
    answered_queries.append(query.answer(operator))
    st.session_state.four_goals_answered_queries = answered_queries
    reset_algorithm_state()


def main() -> None:
    st.set_page_config(page_title="4 Goals", layout="wide")
    st.title("4 Goals")

    initialize_state(initial_n_alternatives=10)

    with st.sidebar:
        show_handlungsalternativen = bool(
            st.toggle("Show handlungsalternativen", value=False)
        )
        st.markdown("**Visualisierung**")
        grid_step = float(
            st.slider(
                "Grid step",
                min_value=0.02,
                max_value=0.2,
                value=DEFAULT_GRID_STEP,
                step=0.01,
                format="%.2f",
                help="Kleinere Werte machen das Tetraeder-Gitter feiner.",
            )
        )
        grid_point_size = float(
            st.slider(
                "Grid point size",
                min_value=1.0,
                max_value=12.0,
                value=DEFAULT_GRID_POINT_SIZE,
                step=0.5,
            )
        )
        target_point_size = float(
            st.slider(
                "Target vector size",
                min_value=4.0,
                max_value=24.0,
                value=DEFAULT_TARGET_POINT_SIZE,
                step=1.0,
            )
        )
        target_point_is_black = bool(
            st.toggle(
                "Black target vector",
                value=True,
            )
        )
        st.markdown("**Sampling fuer Query-Bewertung**")
        sample_count = int(
            st.number_input(
                "Samples",
                min_value=500,
                max_value=30000,
                value=DEFAULT_SAMPLE_COUNT,
                step=500,
            )
        )
        sample_burn_in = int(
            st.number_input(
                "Burn-in",
                min_value=0,
                max_value=10000,
                value=DEFAULT_SAMPLE_BURN_IN,
                step=50,
            )
        )
        sample_thinning = int(
            st.number_input(
                "Thinning",
                min_value=1,
                max_value=100,
                value=DEFAULT_SAMPLE_THINNING,
                step=1,
            )
        )
        if st.button("Resample", use_container_width=True):
            st.session_state.four_goals_sampling_version += 1
            st.rerun()

    control_col_1, control_col_2, control_col_3 = st.columns([0.45, 0.275, 0.275])
    with control_col_1:
        current_n_alternatives = len(st.session_state.four_goals_alternative_names)
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
            st.session_state.four_goals_target_weights = generate_random_target_weights()
            reset_algorithm_state()
            st.rerun()

    alternative_names: list[str] = st.session_state.four_goals_alternative_names
    utilities: np.ndarray = st.session_state.four_goals_utilities
    target_weights: np.ndarray = st.session_state.four_goals_target_weights
    answered_queries: list[AnsweredQuery] = st.session_state.four_goals_answered_queries

    st.write(
        "**Zielgewichtsvektor:** "
        + ", ".join(
            f"{label} = {target_weights[index]:.4f}"
            for index, label in enumerate(GOAL_LABELS)
        )
    )

    utilities_df = pd.DataFrame(
        {
            "Alternative": alternative_names,
            **{
                GOAL_LABELS[index]: utilities[:, index]
                for index in range(N_GOALS)
            },
        }
    )
    if show_handlungsalternativen:
        st.markdown("**Handlungsalternativen**")
        st.dataframe(utilities_df, use_container_width=True, hide_index=True)

    answered_query_signature = build_answered_query_signature(answered_queries)
    simplex_grid = generate_simplex_grid(grid_step)
    try:
        probability_samples = sample_probability_weights(
            answered_query_signature=answered_query_signature,
            sample_count=sample_count,
            sample_burn_in=sample_burn_in,
            sample_thinning=sample_thinning,
            sampling_version=st.session_state.four_goals_sampling_version,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    alternativen_matrix = AlternativenMatrix(entries=utilities.tolist())
    algorithm_details: dict[str, Any] | None = None
    algorithm_error: str | None = None
    if st.session_state.four_goals_algorithm_requested:
        try:
            algorithm_details = compute_algorithm_details(
                alternativen_matrix=alternativen_matrix,
                answered_queries=answered_queries,
                probability_samples=probability_samples,
            )
        except ValueError as exc:
            algorithm_error = str(exc)

    selected_query: Query | None = None
    if algorithm_details is not None and algorithm_details["termination"] is None:
        selected_query = algorithm_details["best_query"]

    fig = draw_tetrahedron(
        grid=simplex_grid,
        utilities=utilities,
        alternative_names=alternative_names,
        answered_queries=answered_queries,
        target_weights=target_weights,
        selected_query=selected_query,
        grid_point_size=grid_point_size,
        target_point_size=target_point_size,
        target_point_is_black=target_point_is_black,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Diese Gitter-Darstellung zeigt, welche Alternative in welchem Bereich "
        "des 4-Ziele-Simplex optimal ist."
    )

    if answered_queries:
        st.markdown("**Bisher beantwortete Trade-off-Queries**")
        for index, answered_query in enumerate(answered_queries, start=1):
            st.write(f"{index}. {answered_query_to_text(answered_query)}")
    else:
        st.markdown("**Bisher beantwortete Trade-off-Queries**")
        st.write("Noch keine.")

    if not st.session_state.four_goals_algorithm_requested:
        if st.button("Run Algorithmus", use_container_width=True):
            st.session_state.four_goals_algorithm_requested = True
            st.rerun()
        return

    if algorithm_error is not None:
        st.error(algorithm_error)
        return

    if algorithm_details is None:
        st.error("Der Algorithmus konnte nicht ausgewertet werden.")
        return

    termination = algorithm_details["termination"]
    if isinstance(termination, TerminationResult):
        st.markdown("**Termination Result**")
        st.write(f"`reason`: `{termination.reason}`")
        st.write(f"`share_precision`: `{termination.share_precision}`")
        st.write(f"`remaining_candidates`: {termination.remaining_candidates}")
        st.dataframe(
            build_termination_shares_dataframe(
                termination=termination,
                alternative_names=alternative_names,
            ),
            use_container_width=True,
            hide_index=True,
        )
        return

    query_infos: list[QueryInfo] = algorithm_details["query_infos"]
    best_query: Query = algorithm_details["best_query"]
    best_expected: float = algorithm_details["best_expected"]

    st.markdown("**Kandidatenmenge**")
    st.write(
        ", ".join(
            alternative_names[index]
            for index in algorithm_details["kandidatenmenge"]
        )
    )
    st.markdown("**Beste Query**")
    st.write(query_to_text(best_query))
    st.latex(rf"\mathbb{{E}}[N \mid q^\star] = {best_expected:g}")

    if query_infos:
        st.markdown("**Query-Auswertung**")
        st.dataframe(
            build_query_dataframe(query_infos, alternative_names),
            use_container_width=True,
            hide_index=True,
            column_config={
                "p_<": st.column_config.NumberColumn("p_<", format="%.4f"),
                "p_>": st.column_config.NumberColumn("p_>", format="%.4f"),
                "E[N|q]": st.column_config.NumberColumn("E[N|q]", format="%.6f"),
            },
        )

    answer_left, answer_middle, answer_right = st.columns(3)
    with answer_left:
        if st.button("< beantworten", use_container_width=True):
            append_answer("<", best_query)
            st.rerun()
    with answer_middle:
        if st.button("= beantworten", use_container_width=True):
            append_answer("=", best_query)
            st.rerun()
    with answer_right:
        if st.button("> beantworten", use_container_width=True):
            append_answer(">", best_query)
            st.rerun()


if __name__ == "__main__":
    main()
