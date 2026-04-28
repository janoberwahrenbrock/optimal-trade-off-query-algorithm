from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.build_ungleichungssysteme import build_W, build_optimal_region_in_W
from src.io_models import AlternativenMatrix, AnsweredQuery, Query
from src.query_bewertung import (
    build_zielpaar_intervalle_lookup,
    compute_query_info,
    filter_already_answered_queries,
    filter_informative_query_infos,
)
from src.query_kandidaten import compute_all_query_kandidaten
from src.ratio_intervalle import compute_all_ratio_intervals
from src.sampling import sample_points_from_ungleichungssystem
from src.ungleichungssysteme import Ungleichungssystem


DEFAULT_CASE_PATH = PROJECT_ROOT / "data" / "a5_a10_case.json"


@dataclass(frozen=True)
class CaseData:
    goal_labels: list[str]
    alternativen_matrix: AlternativenMatrix
    answered_queries: list[AnsweredQuery]


@dataclass(frozen=True)
class ChildEvaluation:
    feasible: bool
    candidate_count: int
    best_expected_candidates: float
    best_query: Query | None
    informative_query_count: int
    sample_count: int


@dataclass(frozen=True)
class SweepRow:
    s: float
    p_less: float
    p_greater: float
    less_evaluation: ChildEvaluation
    greater_evaluation: ChildEvaluation

    @property
    def two_step_expected_candidates(self) -> float:
        return (
            self.p_less * self.less_evaluation.best_expected_candidates
            + self.p_greater * self.greater_evaluation.best_expected_candidates
        )


def main() -> None:
    args = parse_args()
    case_data = load_case(args.case)

    goal_index_a = resolve_goal_index(case_data.goal_labels, args.goal_a)
    goal_index_b = resolve_goal_index(case_data.goal_labels, args.goal_b)

    lower, upper = compute_ratio_range_in_W(
        anzahl_ziele=case_data.alternativen_matrix.get_anzahl_spalten(),
        answered_queries=case_data.answered_queries,
        ziel_index_a=goal_index_a,
        ziel_index_b=goal_index_b,
    )

    s_values = build_s_values(
        lower=lower,
        upper=upper,
        steps=args.steps,
        include_endpoints=args.include_endpoints,
        scale=args.scale,
    )

    base_W = build_W(
        case_data.alternativen_matrix.get_anzahl_spalten(),
        case_data.answered_queries,
    )
    base_samples = sample_points_from_ungleichungssystem(
        system=base_W,
        num_samples=args.samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
    )

    rows: list[SweepRow] = []
    for index, s in enumerate(s_values, start=1):
        row = evaluate_s_value(
            s=s,
            case_data=case_data,
            goal_index_a=goal_index_a,
            goal_index_b=goal_index_b,
            base_samples=base_samples,
            child_samples=args.child_samples,
            min_child_samples=args.min_child_samples,
            burn_in=args.burn_in,
            thinning=args.thinning,
            seed=args.seed + index * 10_000,
        )
        rows.append(row)

    print()
    print(f"Case: {args.case}")
    print(
        "Erste Query: "
        f"{case_data.goal_labels[goal_index_a]} ? s * {case_data.goal_labels[goal_index_b]}"
    )
    print(f"Verhaeltnisintervall in W(T): [{lower:.8g}, {upper:.8g}]")
    print(f"s-Werte: {len(rows)}, samples={args.samples}, seed={args.seed}")
    print()
    print_table(rows, case_data.goal_labels)

    best_row = min(rows, key=lambda row: row.two_step_expected_candidates)
    print()
    print("Minimum:")
    print(
        f"  s = {best_row.s:.10g}, "
        f"F(s) = {best_row.two_step_expected_candidates:.6f}, "
        f"p_< = {best_row.p_less:.4f}, "
        f"p_> = {best_row.p_greater:.4f}"
    )
    print(
        "  nach <: "
        + format_child_evaluation(best_row.less_evaluation, case_data.goal_labels)
    )
    print(
        "  nach >: "
        + format_child_evaluation(best_row.greater_evaluation, case_data.goal_labels)
    )

    plot_path = resolve_plot_path(args.plot_path, args.case)
    if plot_path is not None:
        save_plot(
            rows=rows,
            best_row=best_row,
            goal_label_a=case_data.goal_labels[goal_index_a],
            goal_label_b=case_data.goal_labels[goal_index_b],
            output_path=plot_path,
            show_plot=args.show_plot,
        )
        print()
        print(f"Grafik gespeichert: {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ueber s fuer eine vorbereitende Query und berechne den "
            "minimalen erwarteten Kandidatenwert der besten zweiten Query."
        )
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=DEFAULT_CASE_PATH,
        help=f"Pfad zur JSON-Case-Datei. Default: {DEFAULT_CASE_PATH}",
    )
    parser.add_argument(
        "--goal-a",
        default="Finanzen",
        help="Zaehlerziel der ersten Query. Default: Finanzen",
    )
    parser.add_argument(
        "--goal-b",
        default="Sicherheit",
        help="Nennerziel der ersten Query. Default: Sicherheit",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=41,
        help="Anzahl der s-Werte im Intervall. Default: 41",
    )
    parser.add_argument(
        "--scale",
        choices=["linear", "log"],
        default="linear",
        help="Skalierung der s-Werte. Default: linear",
    )
    parser.add_argument(
        "--include-endpoints",
        action="store_true",
        help="Auch die Intervallenden als s-Werte testen.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Samples aus W(T) fuer p_< und p_>. Default: 5000",
    )
    parser.add_argument(
        "--child-samples",
        type=int,
        default=1000,
        help="Fallback-Samples fuer Kindzustaende mit zu wenigen Samples. Default: 1000",
    )
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=200,
        help="Mindestanzahl konditionierter Samples je Kindzustand. Default: 200",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=200,
        help="Burn-in fuer Hit-and-Run-Sampling. Default: 200",
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=5,
        help="Thinning fuer Hit-and-Run-Sampling. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed fuer reproduzierbare Sampling-Ergebnisse. Default: 1",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("auto"),
        help=(
            "Pfad fuer die PNG-Grafik. Default: data/<case>_two_step_s.png. "
            "Mit --plot-path none wird keine Grafik erzeugt."
        ),
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Grafik nach dem Speichern auch als Matplotlib-Fenster anzeigen.",
    )
    return parser.parse_args()


def load_case(path: Path) -> CaseData:
    with path.open("r", encoding="utf-8") as file:
        raw_case: dict[str, Any] = json.load(file)

    answered_queries = [
        AnsweredQuery(
            ziel_index_a=int(tradeoff["ziel_index_a"]),
            ziel_index_b=int(tradeoff["ziel_index_b"]),
            value=float(tradeoff["value"]),
            operator=tradeoff["operator"],
        )
        for tradeoff in raw_case["tradeoffs"]
    ]

    return CaseData(
        goal_labels=list(raw_case["goal_labels"]),
        alternativen_matrix=AlternativenMatrix(
            entries=raw_case["handlungsalternativenmatrix"],
        ),
        answered_queries=answered_queries,
    )


def resolve_goal_index(goal_labels: list[str], goal: str) -> int:
    if goal.isdigit():
        goal_index = int(goal)
        if 0 <= goal_index < len(goal_labels):
            return goal_index
        raise ValueError(f"goal index {goal_index} is out of range")

    normalized_goal = goal.casefold()
    for index, label in enumerate(goal_labels):
        if label.casefold() == normalized_goal:
            return index

    raise ValueError(f"unknown goal {goal!r}; available goals: {goal_labels}")


def compute_ratio_range_in_W(
    anzahl_ziele: int,
    answered_queries: list[AnsweredQuery],
    ziel_index_a: int,
    ziel_index_b: int,
) -> tuple[float, float]:
    normalized_W = build_normalized_W_for_ratio(
        anzahl_ziele=anzahl_ziele,
        answered_queries=answered_queries,
        normierungs_ziel_index=ziel_index_b,
    )

    zielfunktion = [0.0] * anzahl_ziele
    zielfunktion[ziel_index_a] = 1.0

    lower = normalized_W.minimize(zielfunktion)
    upper = normalized_W.maximize(zielfunktion)

    if lower.status != "optimal" or lower.optimal_value is None:
        raise RuntimeError(f"lower ratio bound is {lower.status}")

    if upper.status != "optimal" or upper.optimal_value is None:
        raise RuntimeError(f"upper ratio bound is {upper.status}")

    return float(lower.optimal_value), float(upper.optimal_value)


def build_normalized_W_for_ratio(
    anzahl_ziele: int,
    answered_queries: list[AnsweredQuery],
    normierungs_ziel_index: int,
) -> Ungleichungssystem:
    system = Ungleichungssystem()

    for ziel_index in range(anzahl_ziele):
        row = [0.0] * anzahl_ziele
        row[ziel_index] = -1.0
        system.add_ungleichung(row, 0.0)

    row = [0.0] * anzahl_ziele
    row[normierungs_ziel_index] = 1.0
    system.add_gleichung(row, 1.0)

    for answered_query in answered_queries:
        row, rhs, is_equality = build_query_constraint(answered_query, anzahl_ziele)
        if is_equality:
            system.add_gleichung(row, rhs)
        else:
            system.add_ungleichung(row, rhs)

    return system


def build_query_constraint(
    answered_query: AnsweredQuery,
    anzahl_ziele: int,
) -> tuple[list[float], float, bool]:
    row = [0.0] * anzahl_ziele
    a = int(answered_query.ziel_index_a)
    b = int(answered_query.ziel_index_b)
    value = float(answered_query.value)

    if answered_query.operator == ">":
        row[a] = -1.0
        row[b] = value
        return scale_constraint(row, 0.0, False)

    if answered_query.operator == "<":
        row[a] = 1.0
        row[b] = -value
        return scale_constraint(row, 0.0, False)

    if answered_query.operator == "=":
        row[a] = 1.0
        row[b] = -value
        return scale_constraint(row, 0.0, True)

    raise ValueError(f"unknown operator: {answered_query.operator}")


def scale_constraint(
    row: list[float],
    rhs: float,
    is_equality: bool,
) -> tuple[list[float], float, bool]:
    scale = max(abs(value) for value in row)
    if scale == 0.0:
        return row, rhs, is_equality
    return [value / scale for value in row], rhs / scale, is_equality


def build_s_values(
    lower: float,
    upper: float,
    steps: int,
    include_endpoints: bool,
    scale: str,
) -> list[float]:
    if steps <= 0:
        raise ValueError("steps must be positive")

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("ratio range must be finite")

    if lower > upper:
        raise ValueError("lower ratio bound must not be greater than upper bound")

    if math.isclose(lower, upper):
        return [lower]

    if scale == "log" and lower <= 0.0:
        raise ValueError("log scale requires a positive lower ratio bound")

    denominator = steps - 1 if include_endpoints and steps > 1 else steps + 1
    start = 0 if include_endpoints else 1

    values: list[float] = []
    for offset in range(steps):
        fraction = (start + offset) / denominator
        if scale == "linear":
            value = lower + fraction * (upper - lower)
        else:
            log_lower = math.log(lower)
            log_upper = math.log(upper)
            value = math.exp(log_lower + fraction * (log_upper - log_lower))
        values.append(float(value))

    return values


def evaluate_s_value(
    s: float,
    case_data: CaseData,
    goal_index_a: int,
    goal_index_b: int,
    base_samples: list[list[float]],
    child_samples: int,
    min_child_samples: int,
    burn_in: int,
    thinning: int,
    seed: int,
) -> SweepRow:
    first_query = Query(
        ziel_index_a=goal_index_a,
        ziel_index_b=goal_index_b,
        value=s,
    )
    less_samples = [
        sample
        for sample in base_samples
        if sample[goal_index_a] < s * sample[goal_index_b]
    ]
    greater_samples = [
        sample
        for sample in base_samples
        if sample[goal_index_a] > s * sample[goal_index_b]
    ]
    considered_sample_count = len(less_samples) + len(greater_samples)
    if considered_sample_count == 0:
        raise RuntimeError("all base samples lie on the first query boundary")

    less_evaluation = evaluate_child_state(
        alternativen_matrix=case_data.alternativen_matrix,
        answered_queries=case_data.answered_queries + [first_query.answer("<")],
        samples=less_samples,
        fallback_sample_count=child_samples,
        min_sample_count=min_child_samples,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed + 1,
    )
    greater_evaluation = evaluate_child_state(
        alternativen_matrix=case_data.alternativen_matrix,
        answered_queries=case_data.answered_queries + [first_query.answer(">")],
        samples=greater_samples,
        fallback_sample_count=child_samples,
        min_sample_count=min_child_samples,
        burn_in=burn_in,
        thinning=thinning,
        seed=seed + 2,
    )

    return SweepRow(
        s=s,
        p_less=len(less_samples) / considered_sample_count,
        p_greater=len(greater_samples) / considered_sample_count,
        less_evaluation=less_evaluation,
        greater_evaluation=greater_evaluation,
    )


def evaluate_child_state(
    alternativen_matrix: AlternativenMatrix,
    answered_queries: list[AnsweredQuery],
    samples: list[list[float]],
    fallback_sample_count: int,
    min_sample_count: int,
    burn_in: int,
    thinning: int,
    seed: int,
) -> ChildEvaluation:
    W = build_W(alternativen_matrix.get_anzahl_spalten(), answered_queries)
    if not W.is_feasible():
        return ChildEvaluation(
            feasible=False,
            candidate_count=0,
            best_expected_candidates=0.0,
            best_query=None,
            informative_query_count=0,
            sample_count=len(samples),
        )

    candidates = compute_candidates(alternativen_matrix, W)
    if not candidates:
        raise RuntimeError("child state is feasible but has no candidate")

    if len(samples) < min_sample_count:
        samples = sample_points_from_ungleichungssystem(
            system=W,
            num_samples=fallback_sample_count,
            burn_in=burn_in,
            thinning=thinning,
            seed=seed,
        )

    if len(candidates) == 1:
        return ChildEvaluation(
            feasible=True,
            candidate_count=len(candidates),
            best_expected_candidates=1.0,
            best_query=None,
            informative_query_count=0,
            sample_count=len(samples),
        )

    zielpaar_intervalle = compute_all_ratio_intervals(
        alternativen_matrix=alternativen_matrix,
        answered_queries=answered_queries,
        kandidatenmenge=candidates,
    )
    query_candidates = compute_all_query_kandidaten(zielpaar_intervalle)
    query_candidates = filter_already_answered_queries(query_candidates, answered_queries)
    zielpaar_intervalle_lookup = build_zielpaar_intervalle_lookup(zielpaar_intervalle)

    query_infos = []
    for query in query_candidates:
        query_info = compute_query_info(
            query=query,
            samples=samples,
            zielpaar_intervalle_lookup=zielpaar_intervalle_lookup,
        )
        if query_info is not None:
            query_infos.append(query_info)

    informative_query_infos = filter_informative_query_infos(
        query_infos=query_infos,
        kandidatenmenge=set(candidates),
    )

    if not informative_query_infos:
        return ChildEvaluation(
            feasible=True,
            candidate_count=len(candidates),
            best_expected_candidates=float(len(candidates)),
            best_query=None,
            informative_query_count=0,
            sample_count=len(samples),
        )

    best_query_info = min(
        informative_query_infos,
        key=lambda query_info: query_info.expected_kandidatenanzahl,
    )
    return ChildEvaluation(
        feasible=True,
        candidate_count=len(candidates),
        best_expected_candidates=float(best_query_info.expected_kandidatenanzahl),
        best_query=best_query_info.query,
        informative_query_count=len(informative_query_infos),
        sample_count=len(samples),
    )


def compute_candidates(
    alternativen_matrix: AlternativenMatrix,
    W: Ungleichungssystem,
) -> list[int]:
    candidates: list[int] = []
    for alternative_index in range(alternativen_matrix.get_anzahl_zeilen()):
        optimal_region = build_optimal_region_in_W(
            alternativen_matrix=alternativen_matrix,
            W=W,
            alternative_index=alternative_index,
        )
        if optimal_region.is_feasible():
            candidates.append(alternative_index)
    return candidates


def print_table(rows: list[SweepRow], goal_labels: list[str]) -> None:
    table_rows = [
        [
            f"{row.s:.8g}",
            f"{row.p_less:.4f}",
            f"{row.p_greater:.4f}",
            str(row.less_evaluation.candidate_count),
            f"{row.less_evaluation.best_expected_candidates:.4f}",
            format_query(row.less_evaluation.best_query, goal_labels),
            str(row.greater_evaluation.candidate_count),
            f"{row.greater_evaluation.best_expected_candidates:.4f}",
            format_query(row.greater_evaluation.best_query, goal_labels),
            f"{row.two_step_expected_candidates:.6f}",
        ]
        for row in rows
    ]
    headers = [
        "s",
        "p_<",
        "p_>",
        "K_<",
        "B_<",
        "beste q2_<",
        "K_>",
        "B_>",
        "beste q2_>",
        "F(s)",
    ]
    print_aligned_table(headers, table_rows)


def print_aligned_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(row[column_index]) for row in [headers, *rows])
        for column_index in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def format_child_evaluation(
    evaluation: ChildEvaluation,
    goal_labels: list[str],
) -> str:
    if not evaluation.feasible:
        return "infeasible"

    return (
        f"K={evaluation.candidate_count}, "
        f"B={evaluation.best_expected_candidates:.6f}, "
        f"q2={format_query(evaluation.best_query, goal_labels)}, "
        f"info_queries={evaluation.informative_query_count}, "
        f"samples={evaluation.sample_count}"
    )


def format_query(query: Query | None, goal_labels: list[str]) -> str:
    if query is None:
        return "-"

    return (
        f"{goal_labels[int(query.ziel_index_a)]}"
        f"?{float(query.value):.5g}*"
        f"{goal_labels[int(query.ziel_index_b)]}"
    )


def resolve_plot_path(plot_path_argument: Path, case_path: Path) -> Path | None:
    if str(plot_path_argument).casefold() == "none":
        return None

    if str(plot_path_argument).casefold() == "auto":
        return case_path.with_name(f"{case_path.stem}_two_step_s.png")

    return plot_path_argument


def save_plot(
    rows: list[SweepRow],
    best_row: SweepRow,
    goal_label_a: str,
    goal_label_b: str,
    output_path: Path,
    show_plot: bool,
) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    s_values = [row.s for row in rows]
    two_step_values = [row.two_step_expected_candidates for row in rows]
    less_values = [row.less_evaluation.best_expected_candidates for row in rows]
    greater_values = [row.greater_evaluation.best_expected_candidates for row in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        s_values,
        two_step_values,
        marker="o",
        linewidth=2.4,
        color="#0057FF",
        label="F(s)",
    )
    ax.plot(
        s_values,
        less_values,
        linestyle="--",
        linewidth=1.6,
        color="#FF7A00",
        label="B_< nach erster Antwort <",
    )
    ax.plot(
        s_values,
        greater_values,
        linestyle="--",
        linewidth=1.6,
        color="#9B00FF",
        label="B_> nach erster Antwort >",
    )
    ax.axvline(
        best_row.s,
        color="#111111",
        linestyle=":",
        linewidth=1.8,
        label=f"bestes s = {best_row.s:.5g}",
    )
    ax.scatter(
        [best_row.s],
        [best_row.two_step_expected_candidates],
        s=90,
        color="#111111",
        zorder=5,
    )

    ax.set_title(
        f"Two-Step-Sweep fuer erste Query: {goal_label_a} ? s * {goal_label_b}"
    )
    ax.set_xlabel(f"s = {goal_label_a} / {goal_label_b}")
    ax.set_ylabel("Erwartete Kandidatenzahl")
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
