from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors as mcolors
from matplotlib.patches import Polygon

MULTISTEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MULTISTEP_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multistep.src.candidates import compute_candidate_set
from multistep.src.query_value_function import (
    build_linear_query_values,
    compute_sample_ratio_range,
    estimate_depth_two_value_curve_for_goal_pair_from_samples,
    evaluate_query_value_curve_for_goal_pair,
)
from multistep.src.sampling import sample_points_from_constraint_system
from multistep.src.weight_space import build_weight_space
from multistep.src.models import AlternativenMatrix, Query


SQRT3 = np.sqrt(3.0)
GOAL_LABELS = ["Spaß", "Finanzen", "Sicherheit"]
DEFAULT_ALTERNATIVE_COUNT = 5
DEFAULT_SEED = 1
DEFAULT_SAMPLE_COUNT = 1500
DEFAULT_BURN_IN = 200
DEFAULT_THINNING = 4
DEFAULT_SWEEP_STEPS = 121
DEFAULT_INNER_SWEEP_STEPS = 21
DEFAULT_GRID_STEP = 0.01
DEFAULT_MAX_QUERY_VALUE = 10.0


def barycentric_to_cartesian(w_1: float, w_2: float, w_3: float) -> tuple[float, float]:
    x_value = w_2 + 0.5 * w_3
    y_value = (SQRT3 / 2.0) * w_3
    return x_value, y_value


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
    if alternative_count == 1:
        return ["#4C78A8"]

    cmap = plt.get_cmap("nipy_spectral")
    return [
        mcolors.to_hex(cmap(position))
        for position in np.linspace(0.03, 0.97, alternative_count)
    ]


def compute_best_alternatives_on_grid(
    grid: np.ndarray,
    alternatives: AlternativenMatrix,
) -> np.ndarray:
    utilities = np.array(alternatives.entries, dtype=float)
    total_utilities = utilities @ grid.T
    return np.argmax(total_utilities, axis=0).astype(int)


def get_query_line_segment(
    query: Query,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
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

    if len(barycentric_points) != 2:
        return None

    first = barycentric_to_cartesian(*barycentric_points[0].tolist())
    second = barycentric_to_cartesian(*barycentric_points[1].tolist())
    return first, second


def generate_random_problem(
    alternative_count: int,
    seed: int,
) -> tuple[list[str], AlternativenMatrix]:
    rng = np.random.default_rng(seed)
    alternative_names = [f"A{i}" for i in range(1, alternative_count + 1)]
    utilities = np.round(
        rng.uniform(0.0, 1.0, size=(alternative_count, len(GOAL_LABELS))),
        2,
    )
    return alternative_names, AlternativenMatrix(entries=utilities.tolist())


def build_alternatives_dataframe(
    alternative_names: list[str],
    alternatives: AlternativenMatrix,
) -> pd.DataFrame:
    return pd.DataFrame(
        alternatives.entries,
        index=alternative_names,
        columns=GOAL_LABELS,
    )


@st.cache_data(show_spinner=False)
def compute_plot_data(
    alternatives_entries: tuple[tuple[float, ...], ...],
    goal_pair: tuple[int, int],
    sample_count: int,
    burn_in: int,
    thinning: int,
    sweep_steps: int,
    inner_sweep_steps: int,
    max_query_value: float,
    include_depth_two: bool,
    seed: int,
) -> dict[str, Any]:
    alternatives = AlternativenMatrix(
        entries=[list(row) for row in alternatives_entries],
    )
    answered_queries = []
    weight_space = build_weight_space(
        goal_count=alternatives.get_anzahl_spalten(),
        answered_queries=answered_queries,
    )
    samples = sample_points_from_constraint_system(
        system=weight_space,
        num_samples=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed,
    )
    ratio_lower, ratio_upper = compute_sample_ratio_range(
        samples=samples,
        numerator_index=goal_pair[0],
        denominator_index=goal_pair[1],
    )
    plot_lower = 0.0
    plot_upper = max_query_value

    query_values = build_linear_query_values(
        lower=plot_lower,
        upper=plot_upper,
        steps=sweep_steps,
    )
    evaluations = evaluate_query_value_curve_for_goal_pair(
        alternatives=alternatives,
        answered_queries=answered_queries,
        samples=samples,
        goal_pair=goal_pair,
        query_values=query_values,
    )
    depth_two_evaluations = []
    if include_depth_two:
        depth_two_evaluations = estimate_depth_two_value_curve_for_goal_pair_from_samples(
            alternatives=alternatives,
            samples=samples,
            goal_pair=goal_pair,
            query_values=query_values,
            inner_query_value_steps=inner_sweep_steps,
            max_query_value=max_query_value,
        )
    candidates = compute_candidate_set(
        alternatives=alternatives,
        weight_space=weight_space,
    )

    return {
        "candidates": candidates,
        "ratio_lower": ratio_lower,
        "ratio_upper": ratio_upper,
        "plot_lower": plot_lower,
        "plot_upper": plot_upper,
        "query_values": [evaluation.query.value for evaluation in evaluations],
        "expected_values": [
            evaluation.expected_candidate_count
            for evaluation in evaluations
        ],
        "depth_two_expected_values": [
            evaluation.expected_candidate_count
            for evaluation in depth_two_evaluations
        ],
        "p_less": [
            evaluation.answer_evaluations[0].probability
            for evaluation in evaluations
        ],
        "p_equal": [
            evaluation.answer_evaluations[1].probability
            for evaluation in evaluations
        ],
        "p_greater": [
            evaluation.answer_evaluations[2].probability
            for evaluation in evaluations
        ],
    }


def render_value_plot(
    plot_data: dict[str, Any],
    goal_pair: tuple[int, int],
) -> plt.Figure:
    query_values = plot_data["query_values"]
    expected_values = plot_data["expected_values"]
    best_index = int(np.argmin(expected_values))
    depth_two_expected_values = plot_data["depth_two_expected_values"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        query_values,
        expected_values,
        color="#1f77b4",
        linewidth=2.2,
        label="Tiefe 1",
    )
    if depth_two_expected_values:
        depth_two_best_index = int(np.argmin(depth_two_expected_values))
        ax.plot(
            query_values,
            depth_two_expected_values,
            color="#d62728",
            linewidth=2.2,
            label="Tiefe 2",
        )
        ax.scatter(
            [query_values[depth_two_best_index]],
            [depth_two_expected_values[depth_two_best_index]],
            color="#d62728",
            s=52,
            zorder=5,
        )
    ax.scatter(
        [query_values[best_index]],
        [expected_values[best_index]],
        color="#111111",
        s=52,
        zorder=5,
    )
    ax.axvline(
        query_values[best_index],
        color="#111111",
        linestyle=":",
        linewidth=1.5,
    )
    ax.set_xlabel(f"s = {GOAL_LABELS[goal_pair[0]]} / {GOAL_LABELS[goal_pair[1]]}")
    ax.set_ylabel("Erwartete Kandidatenzahl")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def render_simplex_plot(
    alternative_names: list[str],
    alternatives: AlternativenMatrix,
    candidates: list[int],
    goal_pair: tuple[int, int],
    query_value: float,
    grid_step: float,
) -> plt.Figure:
    grid = generate_simplex_grid(grid_step)
    best_alternatives = compute_best_alternatives_on_grid(
        grid=grid,
        alternatives=alternatives,
    )
    grid_points = np.array(
        [barycentric_to_cartesian(*weights.tolist()) for weights in grid],
        dtype=float,
    )
    colors = get_alternative_colors(alternatives.get_anzahl_zeilen())

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
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

    for alternative_index in range(alternatives.get_anzahl_zeilen()):
        mask = best_alternatives == alternative_index
        if not np.any(mask):
            continue

        alpha = 0.28 if alternative_index in candidates else 0.10
        ax.scatter(
            grid_points[mask, 0],
            grid_points[mask, 1],
            s=9,
            c=colors[alternative_index],
            alpha=alpha,
            marker="s",
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    query = Query(
        ziel_index_a=goal_pair[0],
        ziel_index_b=goal_pair[1],
        value=query_value,
    )
    segment = get_query_line_segment(query)
    if segment is not None:
        (x_1, y_1), (x_2, y_2) = segment
        ax.plot(
            [x_1, x_2],
            [y_1, y_2],
            color="#111111",
            linewidth=2.1,
            linestyle="--",
            zorder=3,
        )

    goal_positions = [
        barycentric_to_cartesian(1.0, 0.0, 0.0),
        barycentric_to_cartesian(0.0, 1.0, 0.0),
        barycentric_to_cartesian(0.0, 0.0, 1.0),
    ]
    goal_offsets = [(-0.06, -0.04), (0.01, -0.04), (-0.04, 0.02)]
    for (x_value, y_value), (x_offset, y_offset), label in zip(
        goal_positions,
        goal_offsets,
        GOAL_LABELS,
    ):
        ax.text(
            x_value + x_offset,
            y_value + y_offset,
            label,
            fontsize=11,
            fontweight="bold",
        )

    legend_handles = []
    for candidate_index in candidates:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=alternative_names[candidate_index],
                markerfacecolor=colors[candidate_index],
                markersize=8,
            )
        )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#111111",
            linestyle="--",
            linewidth=2.1,
            label="bestes s",
        )
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
    fig.subplots_adjust(left=0.03, right=0.72, top=0.98, bottom=0.03)
    return fig


def main() -> None:
    st.set_page_config(page_title="Multistep 3 Goals", layout="wide")
    st.title("Multistep 3 Goals")

    with st.sidebar:
        alternative_count = int(
            st.number_input(
                "Alternativen",
                min_value=2,
                max_value=20,
                value=DEFAULT_ALTERNATIVE_COUNT,
                step=1,
            )
        )
        seed = int(
            st.number_input(
                "Seed",
                min_value=0,
                value=DEFAULT_SEED,
                step=1,
            )
        )
        sample_count = int(
            st.number_input(
                "Samples",
                min_value=100,
                max_value=20000,
                value=DEFAULT_SAMPLE_COUNT,
                step=100,
            )
        )
        burn_in = int(
            st.number_input(
                "Burn-in",
                min_value=0,
                max_value=5000,
                value=DEFAULT_BURN_IN,
                step=50,
            )
        )
        thinning = int(
            st.number_input(
                "Thinning",
                min_value=1,
                max_value=100,
                value=DEFAULT_THINNING,
                step=1,
            )
        )
        sweep_steps = int(
            st.number_input(
                "s-Werte",
                min_value=11,
                max_value=501,
                value=DEFAULT_SWEEP_STEPS,
                step=10,
            )
        )
        include_depth_two = st.checkbox("Tiefe 2 plotten", value=True)
        inner_sweep_steps = int(
            st.number_input(
                "s2-Werte",
                min_value=3,
                max_value=101,
                value=DEFAULT_INNER_SWEEP_STEPS,
                step=2,
                disabled=not include_depth_two,
            )
        )
        max_query_value = float(
            st.number_input(
                "Max s",
                min_value=0.01,
                max_value=1000.0,
                value=DEFAULT_MAX_QUERY_VALUE,
                step=1.0,
            )
        )
        grid_step = float(
            st.number_input(
                "Simplex Grid",
                min_value=0.002,
                max_value=0.05,
                value=DEFAULT_GRID_STEP,
                step=0.002,
                format="%.3f",
            )
        )
        goal_pair_label = st.selectbox(
            "Zielpaar",
            options=[
                f"{left} / {right}"
                for left in GOAL_LABELS
                for right in GOAL_LABELS
                if left != right
            ],
            index=1,
        )

    goal_pair_lookup = {
        f"{left} / {right}": (left_index, right_index)
        for left_index, left in enumerate(GOAL_LABELS)
        for right_index, right in enumerate(GOAL_LABELS)
        if left_index != right_index
    }
    goal_pair = goal_pair_lookup[goal_pair_label]

    alternative_names, alternatives = generate_random_problem(
        alternative_count=alternative_count,
        seed=seed,
    )
    alternatives_dataframe = build_alternatives_dataframe(
        alternative_names=alternative_names,
        alternatives=alternatives,
    )

    plot_data = compute_plot_data(
        alternatives_entries=tuple(tuple(row) for row in alternatives.entries),
        goal_pair=goal_pair,
        sample_count=sample_count,
        burn_in=burn_in,
        thinning=thinning,
        sweep_steps=sweep_steps,
        inner_sweep_steps=inner_sweep_steps,
        max_query_value=max_query_value,
        include_depth_two=include_depth_two,
        seed=seed + 10_000,
    )

    st.subheader("Problem")
    candidate_names = [
        alternative_names[candidate_index]
        for candidate_index in plot_data["candidates"]
    ]
    best_index = int(np.argmin(plot_data["expected_values"]))
    best_query_value = float(plot_data["query_values"][best_index])

    simplex_column, matrix_column = st.columns([1.05, 1.0])
    with simplex_column:
        simplex_fig = render_simplex_plot(
            alternative_names=alternative_names,
            alternatives=alternatives,
            candidates=plot_data["candidates"],
            goal_pair=goal_pair,
            query_value=best_query_value,
            grid_step=grid_step,
        )
        st.pyplot(simplex_fig, clear_figure=True)
    with matrix_column:
        st.dataframe(alternatives_dataframe, width="stretch")
        st.write("Kandidaten:", ", ".join(candidate_names))
    st.write(
        "Bestes getestetes s für Tiefe 1:",
        f"{best_query_value:.6g}",
    )
    if plot_data["depth_two_expected_values"]:
        best_depth_two_index = int(np.argmin(plot_data["depth_two_expected_values"]))
        st.write(
            "Bestes getestetes s für Tiefe 2:",
            f"{plot_data['query_values'][best_depth_two_index]:.6g}",
        )

    st.subheader("Query-Wertfunktion")
    st.caption(
        f"geplotteter s-Bereich: [{plot_data['plot_lower']:.6g}, {plot_data['plot_upper']:.6g}] "
        f"(Sample-Bereich: [{plot_data['ratio_lower']:.6g}, {plot_data['ratio_upper']:.6g}])"
    )
    if plot_data["depth_two_expected_values"]:
        st.caption(
            "Tiefe 2 ist sample-basiert: je erster Antwort wird die beste zweite Query "
            "über alle Zielpaare und das s2-Raster gesucht."
        )
    fig = render_value_plot(plot_data=plot_data, goal_pair=goal_pair)
    st.pyplot(fig, clear_figure=True)

    probability_dataframe = pd.DataFrame(
        {
            "s": plot_data["query_values"],
            "E[K]": plot_data["expected_values"],
            "p_<": plot_data["p_less"],
            "p_=": plot_data["p_equal"],
            "p_>": plot_data["p_greater"],
        }
    )
    if plot_data["depth_two_expected_values"]:
        probability_dataframe["E2[K]"] = plot_data["depth_two_expected_values"]
    st.dataframe(probability_dataframe, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
