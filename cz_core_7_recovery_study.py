#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

"""
cz_core_7_recovery_study.py

Seven-parameter synthetic recovery study for the 3-qutrit CZ-core model.

This file intentionally leaves the existing six-parameter demo in
synthetic_parameter_fit.py unchanged. It imports the established simulator,
fitting routines, and SVD selector, then performs the next required stage:

    1. construct the seven-parameter fit specification
    2. construct the seven-parameter finite-difference Jacobian
    3. select a compact fitting circuit subset
    4. verify the selected subset has full numerical rank
    5. run noiseless multi-start recovery
    6. run finite-shot multi-seed Monte Carlo recovery
    7. evaluate fitted parameters on held-out circuits
    8. save trial-level and summary CSV files

Run:

    python cz_core_7_recovery_study.py

Dependencies:

    numpy
    scipy                       recommended
    lightweight_cptp_channel.py
    multi_qubit_state_space.py
    multi_qubit_circuit_simulator.py
    sensitivity_svd_selector.py
    synthetic_parameter_fit.py

Outputs:

    cz_core_7_trial_results.csv
    cz_core_7_parameter_summary.csv

Current parameter interpretation:

    dphi:
        coherent conditional-phase error

    swap_x, swap_y:
        Cartesian residual-swap quadratures

    t1:
        amplitude-damping probability used by the current per-cycle channel

    tphi:
        pure-dephasing probability used by the current per-cycle channel

    leakage:
        leakage probability used by the current qutrit channel

    two_qubit_depolarizing:
        two-qutrit depolarizing probability

The names t1 and tphi are retained for compatibility with the existing code.
They are rates/probabilities in the current model, not physical times.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from multi_qubit_circuit_simulator import CircuitSpec

from sensitivity_svd_selector import (
    CircuitSelectionResult,
    FiniteDifferenceConfig,
    JacobianResult,
    ParameterName,
    ParameterPoint,
    build_pc_friendly_candidates,
    compute_finite_difference_jacobian,
    compute_svd_report,
    select_circuits_greedy_svd,
)

from synthetic_parameter_fit import (
    FitParameterSpec,
    FitResult,
    SyntheticData,
    fit_synthetic_data,
    generate_synthetic_data,
    simulate_selected_vector,
    fit_parameter_spec_from_registry,
)


CZ_CORE_7_PARAMETERS: Tuple[ParameterName, ...] = (
    "dphi",
    "swap_x",
    "swap_y",
    "t1",
    "tphi",
    "leakage",
    "two_qubit_depolarizing",
)


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for the seven-parameter recovery study."""

    num_qubits: int = 3
    local_dim: int = 3
    target_pair: Tuple[int, int] = (0, 1)

    max_depth: int = 12
    max_candidates: int = 80
    top_k: int = 12

    noiseless_initializations: int = 10
    noiseless_seed: int = 20260720
    noiseless_initial_perturbation: float = 0.50

    shots_values: Tuple[int, ...] = (1000, 4000, 16000)
    monte_carlo_seeds: int = 30
    monte_carlo_base_seed: int = 731000

    minimum_standard_error_factor: float = 0.5

    rank_tolerance: float = 1.0e-10
    noiseless_residual_tolerance: float = 5.0e-8
    noiseless_normalized_error_tolerance: float = 1.0e-4

    bound_tolerance_fraction: float = 1.0e-6

    trial_csv_path: str = "cz_core_7_trial_results.csv"
    summary_csv_path: str = "cz_core_7_parameter_summary.csv"


@dataclass(frozen=True)
class CircuitDesign:
    """Candidate, selected, and held-out circuit collections."""

    candidates: Tuple[CircuitSpec, ...]
    selected_circuits: Tuple[CircuitSpec, ...]
    held_out_circuits: Tuple[CircuitSpec, ...]
    jacobian_result: JacobianResult
    selection_result: CircuitSelectionResult


@dataclass(frozen=True)
class TrialResult:
    """One finite-shot recovery trial."""

    shots: int
    seed: int
    optimizer_success: bool
    normalized_residual_rms: float
    held_out_rms: float
    held_out_max_abs_error: float
    bound_hit_count: int
    fitted_values: Dict[str, float]
    errors: Dict[str, float]
    normalized_errors: Dict[str, float]


def seven_parameter_fit_spec() -> FitParameterSpec:
    """
    Return the fitting specification for the complete CZ-core-7 model.
    """
    return fit_parameter_spec_from_registry(CZ_CORE_7_PARAMETERS)


def make_design_point() -> ParameterPoint:
    """Return the nominal point used for finite-difference circuit design."""
    theta = 0.004
    chi = 0.3

    return ParameterPoint(
        dphi=-0.010,
        theta=theta,
        chi=chi,
        swap_x=theta * np.cos(chi),
        swap_y=theta * np.sin(chi),
        t1=2.0e-4,
        tphi=3.0e-4,
        local_depolarizing=1.0e-4,
        leakage=5.0e-5,
        seepage=1.0e-4,
        two_qubit_depolarizing=5.0e-4,
        readout_0_to_1=0.010,
        readout_1_to_0=0.015,
    )


def make_true_point() -> ParameterPoint:
    """
    Return the synthetic truth.

    Every fitted parameter differs from the design/initial point. In
    particular, tphi is deliberately changed so that seven-parameter recovery
    cannot pass by leaving tphi fixed at its initial value.
    """
    theta = 0.0052
    chi = 0.3

    return ParameterPoint(
        dphi=-0.013,
        theta=theta,
        chi=chi,
        swap_x=theta * np.cos(chi),
        swap_y=theta * np.sin(chi),
        t1=2.6e-4,
        tphi=4.2e-4,
        local_depolarizing=1.0e-4,
        leakage=7.0e-5,
        seepage=1.0e-4,
        two_qubit_depolarizing=7.5e-4,
        readout_0_to_1=0.010,
        readout_1_to_0=0.015,
    )


def make_initial_point() -> ParameterPoint:
    """Return the standard initial point used by finite-shot trials."""
    return make_design_point()


def build_seven_parameter_design(config: StudyConfig) -> CircuitDesign:
    """
    Build the candidate pool, seven-column Jacobian, selected fitting subset,
    and held-out subset.
    """
    candidates = tuple(
        build_pc_friendly_candidates(
            num_qubits=config.num_qubits,
            local_dim=config.local_dim,
            target_pair=config.target_pair,
            max_depth=config.max_depth,
            max_candidates=config.max_candidates,
        )
    )

    if len(candidates) <= config.top_k:
        raise ValueError(
            "The candidate pool must contain more circuits than top_k so that "
            "a non-empty held-out set remains."
        )

    fd_config = FiniteDifferenceConfig(
        target_parameters=CZ_CORE_7_PARAMETERS,
        use_central_difference=True,
    )

    jacobian_result = compute_finite_difference_jacobian(
        circuits=candidates,
        base_point=make_design_point(),
        fd_config=fd_config,
    )

    if jacobian_result.jacobian.shape[1] != len(CZ_CORE_7_PARAMETERS):
        raise RuntimeError(
            "The computed Jacobian does not contain exactly seven columns."
        )

    selection_result = select_circuits_greedy_svd(
        circuits=candidates,
        jac_res=jacobian_result,
        top_k=config.top_k,
        use_norm=True,
    )

    selected_indices = set(selection_result.selected_indices)

    held_out_circuits = tuple(
        circuit
        for index, circuit in enumerate(candidates)
        if index not in selected_indices
    )

    if not held_out_circuits:
        raise RuntimeError("Circuit selection left no held-out circuits.")

    return CircuitDesign(
        candidates=candidates,
        selected_circuits=tuple(selection_result.selected_circuits),
        held_out_circuits=held_out_circuits,
        jacobian_result=jacobian_result,
        selection_result=selection_result,
    )


def summarize_design(
    design: CircuitDesign,
    config: StudyConfig,
) -> None:
    """Print seven-parameter Jacobian and selected-subset diagnostics."""
    full_report = compute_svd_report(
        design.jacobian_result.normalized_jacobian,
        rank_tol=config.rank_tolerance,
    )

    selected_report = compute_svd_report(
        design.selection_result.selected_jacobian,
        rank_tol=config.rank_tolerance,
    )

    print("\nCZ-core-7 circuit design")
    print("=" * 100)
    print(f"candidate circuits       = {len(design.candidates)}")
    print(f"selected circuits        = {len(design.selected_circuits)}")
    print(f"held-out circuits        = {len(design.held_out_circuits)}")
    print(f"full observables         = {design.jacobian_result.jacobian.shape[0]}")
    print(f"number of parameters     = {design.jacobian_result.jacobian.shape[1]}")
    print(f"full numerical rank      = {full_report.numerical_rank}")
    print(f"full condition number    = {full_report.condition_number:.6e}")
    print(f"selected numerical rank  = {selected_report.numerical_rank}")
    print(f"selected condition       = {selected_report.condition_number:.6e}")
    print(
        "selected singular vals  = "
        f"{np.array2string(selected_report.singular_values, precision=6)}"
    )

    print("\nJacobian column norms")
    print("-" * 100)

    raw_column_norms = np.linalg.norm(
        design.jacobian_result.jacobian,
        axis=0,
    )

    for name, norm in zip(
        design.jacobian_result.parameter_names,
        raw_column_norms,
    ):
        print(f"{name:28s} {norm:.10e}")

    print("\nSelected circuits")
    print("-" * 100)

    for index, circuit in enumerate(design.selected_circuits, start=1):
        print(
            f"{index:02d}. {circuit.name:48s} "
            f"n={circuit.repetitions:<3d} "
            f"measurements={len(circuit.measurements)}"
        )

    if selected_report.numerical_rank != len(CZ_CORE_7_PARAMETERS):
        raise AssertionError(
            "The selected circuit subset is not full rank for CZ-core-7."
        )


def exact_synthetic_data(
    circuits: Sequence[CircuitSpec],
    true_point: ParameterPoint,
) -> SyntheticData:
    """
    Construct exact noiseless data without performing unnecessary binomial
    sampling.
    """
    labels, values = simulate_selected_vector(
        circuits=circuits,
        point=true_point,
    )

    values = np.asarray(values, dtype=float)

    return SyntheticData(
        labels=list(labels),
        noiseless_values=values.copy(),
        measured_values=values.copy(),
        standard_errors=np.ones_like(values),
        shots=0,
    )


def randomly_perturbed_initial_point(
    true_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    rng: np.random.Generator,
    perturbation_fraction: float,
    local_dim: int,
) -> ParameterPoint:
    """
    Generate a random initial point around the truth.

    Each fitted parameter is displaced by a uniform random amount between
    minus and plus perturbation_fraction times its fitting scale.
    """
    point = true_point

    for name in fit_spec.names:
        perturbation = rng.uniform(
            -perturbation_fraction,
            perturbation_fraction,
        ) * fit_spec.scales[name]

        value = true_point.get(name) + perturbation
        value = float(
            np.clip(
                value,
                fit_spec.lower_bounds[name],
                fit_spec.upper_bounds[name],
            )
        )

        point = point.with_update(
            name=name,
            value=value,
            local_dim=local_dim,
        )

    return point


def parameter_error_dictionaries(
    result: FitResult,
    true_point: ParameterPoint,
    fit_spec: FitParameterSpec,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return fitted values, physical errors, and scale-normalized errors."""
    fitted_values: Dict[str, float] = {}
    errors: Dict[str, float] = {}
    normalized_errors: Dict[str, float] = {}

    for name in fit_spec.names:
        fitted = result.fitted_point.get(name)
        error = fitted - true_point.get(name)

        fitted_values[name] = fitted
        errors[name] = error
        normalized_errors[name] = error / fit_spec.scales[name]

    return fitted_values, errors, normalized_errors


def count_bound_hits(
    result: FitResult,
    fit_spec: FitParameterSpec,
    tolerance_fraction: float,
) -> int:
    """
    Count fitted parameters that lie numerically close to their bounds.

    The tolerance is computed relative to each parameter's physical range.
    An additional machine-precision floor prevents a zero numerical
    tolerance without introducing a parameter-scale-independent threshold.
    """
    if tolerance_fraction < 0.0:
        raise ValueError("tolerance_fraction must be non-negative")

    hits = 0

    for name in fit_spec.names:
        value = result.fitted_point.get(name)
        lower = fit_spec.lower_bounds[name]
        upper = fit_spec.upper_bounds[name]

        parameter_range = upper - lower
        tolerance = max(
            tolerance_fraction * parameter_range,
            10.0
            * np.finfo(float).eps
            * max(abs(lower), abs(upper), parameter_range),
        )

        near_lower_bound = value <= lower + tolerance
        near_upper_bound = value >= upper - tolerance

        if near_lower_bound or near_upper_bound:
            hits += 1

    return hits


def prediction_errors(
    circuits: Sequence[CircuitSpec],
    true_point: ParameterPoint,
    fitted_point: ParameterPoint,
) -> Tuple[float, float]:
    """
    Return unweighted RMS and maximum absolute prediction errors on a circuit
    collection.
    """
    true_labels, true_values = simulate_selected_vector(
        circuits=circuits,
        point=true_point,
    )
    fitted_labels, fitted_values = simulate_selected_vector(
        circuits=circuits,
        point=fitted_point,
    )

    if true_labels != fitted_labels:
        raise RuntimeError("Held-out observable labels are inconsistent.")

    difference = np.asarray(fitted_values) - np.asarray(true_values)

    rms = float(np.sqrt(np.mean(difference * difference)))
    max_abs_error = float(np.max(np.abs(difference)))

    return rms, max_abs_error


def run_noiseless_multistart(
    design: CircuitDesign,
    config: StudyConfig,
    fit_spec: FitParameterSpec,
    true_point: ParameterPoint,
) -> None:
    """Run and enforce the seven-parameter noiseless recovery test."""
    data = exact_synthetic_data(
        circuits=design.selected_circuits,
        true_point=true_point,
    )

    rng = np.random.default_rng(config.noiseless_seed)
    passed = 0

    print("\nNoiseless seven-parameter multi-start recovery")
    print("=" * 100)
    print(
        f"{'trial':>7s} "
        f"{'success':>9s} "
        f"{'residual_rms':>15s} "
        f"{'max_norm_error':>16s} "
        f"{'heldout_rms':>14s}"
    )
    print("-" * 100)

    for trial_index in range(config.noiseless_initializations):
        initial_point = randomly_perturbed_initial_point(
            true_point=true_point,
            fit_spec=fit_spec,
            rng=rng,
            perturbation_fraction=config.noiseless_initial_perturbation,
            local_dim=config.local_dim,
        )

        result = fit_synthetic_data(
            circuits=design.selected_circuits,
            data=data,
            initial_point=initial_point,
            true_point=true_point,
            fit_spec=fit_spec,
        )

        _, _, normalized_errors = parameter_error_dictionaries(
            result=result,
            true_point=true_point,
            fit_spec=fit_spec,
        )

        max_normalized_error = max(
            abs(value) for value in normalized_errors.values()
        )

        held_out_rms, _ = prediction_errors(
            circuits=design.held_out_circuits,
            true_point=true_point,
            fitted_point=result.fitted_point,
        )

        trial_passed = (
            result.success
            and result.normalized_residual_rms
            < config.noiseless_residual_tolerance
            and max_normalized_error
            < config.noiseless_normalized_error_tolerance
        )

        passed += int(trial_passed)

        print(
            f"{trial_index:7d} "
            f"{str(result.success):>9s} "
            f"{result.normalized_residual_rms:15.6e} "
            f"{max_normalized_error:16.6e} "
            f"{held_out_rms:14.6e}"
        )

    print("-" * 100)
    print(
        f"passed {passed}/{config.noiseless_initializations} "
        "noiseless initializations"
    )

    if passed != config.noiseless_initializations:
        raise AssertionError(
            "CZ-core-7 noiseless multi-start recovery did not pass all trials. "
            "Do not proceed to model expansion."
        )


def run_one_monte_carlo_trial(
    design: CircuitDesign,
    config: StudyConfig,
    fit_spec: FitParameterSpec,
    true_point: ParameterPoint,
    initial_point: ParameterPoint,
    shots: int,
    seed: int,
) -> TrialResult:
    """Generate, fit, and validate one finite-shot synthetic data set."""
    rng = np.random.default_rng(seed)

    minimum_standard_error = (
        config.minimum_standard_error_factor / max(float(shots), 1.0)
    )

    data = generate_synthetic_data(
        circuits=design.selected_circuits,
        true_point=true_point,
        shots=shots,
        rng=rng,
        min_standard_error=minimum_standard_error,
    )

    result = fit_synthetic_data(
        circuits=design.selected_circuits,
        data=data,
        initial_point=initial_point,
        true_point=true_point,
        fit_spec=fit_spec,
    )

    fitted_values, errors, normalized_errors = (
        parameter_error_dictionaries(
            result=result,
            true_point=true_point,
            fit_spec=fit_spec,
        )
    )

    held_out_rms, held_out_max_abs_error = prediction_errors(
        circuits=design.held_out_circuits,
        true_point=true_point,
        fitted_point=result.fitted_point,
    )

    bound_hit_count = count_bound_hits(
        result=result,
        fit_spec=fit_spec,
        tolerance_fraction=config.bound_tolerance_fraction,
    )

    return TrialResult(
        shots=shots,
        seed=seed,
        optimizer_success=result.success,
        normalized_residual_rms=result.normalized_residual_rms,
        held_out_rms=held_out_rms,
        held_out_max_abs_error=held_out_max_abs_error,
        bound_hit_count=bound_hit_count,
        fitted_values=fitted_values,
        errors=errors,
        normalized_errors=normalized_errors,
    )


def run_monte_carlo(
    design: CircuitDesign,
    config: StudyConfig,
    fit_spec: FitParameterSpec,
    true_point: ParameterPoint,
    initial_point: ParameterPoint,
) -> List[TrialResult]:
    """Run all finite-shot Monte Carlo recovery trials."""
    trials: List[TrialResult] = []

    print("\nFinite-shot Monte Carlo recovery")
    print("=" * 100)

    for shots_index, shots in enumerate(config.shots_values):
        shot_trials: List[TrialResult] = []

        for trial_index in range(config.monte_carlo_seeds):
            seed = (
                config.monte_carlo_base_seed
                + 100000 * shots_index
                + trial_index
            )

            trial = run_one_monte_carlo_trial(
                design=design,
                config=config,
                fit_spec=fit_spec,
                true_point=true_point,
                initial_point=initial_point,
                shots=shots,
                seed=seed,
            )

            trials.append(trial)
            shot_trials.append(trial)

            print(
                f"shots={shots:6d} "
                f"trial={trial_index + 1:02d}/{config.monte_carlo_seeds:02d} "
                f"success={trial.optimizer_success!s:5s} "
                f"fit_rms={trial.normalized_residual_rms:8.4f} "
                f"heldout={trial.held_out_rms:10.3e} "
                f"bounds={trial.bound_hit_count}"
            )

        success_count = sum(
            int(trial.optimizer_success) for trial in shot_trials
        )
        mean_fit_rms = float(
            np.mean(
                [trial.normalized_residual_rms for trial in shot_trials]
            )
        )
        mean_held_out_rms = float(
            np.mean([trial.held_out_rms for trial in shot_trials])
        )

        print("-" * 100)
        print(
            f"shots={shots}: success={success_count}/"
            f"{config.monte_carlo_seeds}, "
            f"mean fit RMS={mean_fit_rms:.6f}, "
            f"mean held-out RMS={mean_held_out_rms:.6e}"
        )
        print("-" * 100)

    return trials


def compute_parameter_summary_rows(
    trials: Sequence[TrialResult],
    fit_spec: FitParameterSpec,
    true_point: ParameterPoint,
    shots_values: Sequence[int],
) -> List[Dict[str, float]]:
    """Compute parameter-level empirical statistics for each shot count."""
    rows: List[Dict[str, float]] = []

    for shots in shots_values:
        shot_trials = [trial for trial in trials if trial.shots == shots]

        if not shot_trials:
            continue

        for name in fit_spec.names:
            estimates = np.asarray(
                [trial.fitted_values[name] for trial in shot_trials],
                dtype=float,
            )
            errors = np.asarray(
                [trial.errors[name] for trial in shot_trials],
                dtype=float,
            )
            normalized_errors = np.asarray(
                [trial.normalized_errors[name] for trial in shot_trials],
                dtype=float,
            )

            row = {
                "shots": int(shots),
                "parameter": str(name),
                "truth": float(true_point.get(name)),
                "mean_estimate": float(np.mean(estimates)),
                "bias": float(np.mean(errors)),
                "standard_deviation": float(
                    np.std(estimates, ddof=1)
                    if estimates.size > 1
                    else 0.0
                ),
                "rmse": float(np.sqrt(np.mean(errors * errors))),
                "median_absolute_error": float(
                    np.median(np.abs(errors))
                ),
                "p95_absolute_error": float(
                    np.quantile(np.abs(errors), 0.95)
                ),
                "normalized_rmse": float(
                    np.sqrt(np.mean(normalized_errors * normalized_errors))
                ),
                "optimizer_success_rate": float(
                    np.mean(
                        [
                            float(trial.optimizer_success)
                            for trial in shot_trials
                        ]
                    )
                ),
                "bound_hit_trial_rate": float(
                    np.mean(
                        [
                            float(trial.bound_hit_count > 0)
                            for trial in shot_trials
                        ]
                    )
                ),
                "mean_fit_residual_rms": float(
                    np.mean(
                        [
                            trial.normalized_residual_rms
                            for trial in shot_trials
                        ]
                    )
                ),
                "mean_held_out_rms": float(
                    np.mean(
                        [trial.held_out_rms for trial in shot_trials]
                    )
                ),
            }

            rows.append(row)

    return rows


def print_parameter_summary(
    rows: Sequence[Dict[str, float]],
    fit_spec: FitParameterSpec,
    shots_values: Sequence[int],
) -> None:
    """Print compact Monte Carlo parameter statistics."""
    print("\nMonte Carlo parameter summary")
    print("=" * 125)

    for shots in shots_values:
        print(f"\nshots = {shots}")
        print("-" * 125)
        print(
            f"{'parameter':28s} "
            f"{'bias':>13s} "
            f"{'std':>13s} "
            f"{'rmse':>13s} "
            f"{'norm_rmse':>13s} "
            f"{'p95_abs':>13s} "
            f"{'success':>10s} "
            f"{'bound_rate':>11s}"
        )

        shot_rows = [row for row in rows if row["shots"] == shots]

        for name in fit_spec.names:
            row = next(
                row for row in shot_rows if row["parameter"] == name
            )

            print(
                f"{name:28s} "
                f"{row['bias']:13.4e} "
                f"{row['standard_deviation']:13.4e} "
                f"{row['rmse']:13.4e} "
                f"{row['normalized_rmse']:13.4e} "
                f"{row['p95_absolute_error']:13.4e} "
                f"{row['optimizer_success_rate']:10.3f} "
                f"{row['bound_hit_trial_rate']:11.3f}"
            )


def print_shot_scaling_slopes(
    rows: Sequence[Dict[str, float]],
    fit_spec: FitParameterSpec,
) -> None:
    """
    Fit log(RMSE) against log(shots).

    Ideal independent-shot scaling gives a slope near -0.5.
    """
    print("\nEmpirical shot-scaling slopes")
    print("=" * 100)
    print(f"{'parameter':28s} {'slope':>14s} {'interpretation':>30s}")
    print("-" * 100)

    for name in fit_spec.names:
        parameter_rows = sorted(
            (
                row
                for row in rows
                if row["parameter"] == name and row["rmse"] > 0.0
            ),
            key=lambda row: row["shots"],
        )

        if len(parameter_rows) < 2:
            print(f"{name:28s} {'n/a':>14s} {'insufficient data':>30s}")
            continue

        log_shots = np.log(
            np.asarray(
                [row["shots"] for row in parameter_rows],
                dtype=float,
            )
        )
        log_rmse = np.log(
            np.asarray(
                [row["rmse"] for row in parameter_rows],
                dtype=float,
            )
        )

        slope = float(np.polyfit(log_shots, log_rmse, deg=1)[0])

        if -0.75 <= slope <= -0.25:
            interpretation = "consistent with shot scaling"
        elif slope < -0.75:
            interpretation = "faster than expected/noisy estimate"
        else:
            interpretation = "weak or absent scaling"

        print(f"{name:28s} {slope:14.6f} {interpretation:>30s}")


def write_trial_csv(
    trials: Sequence[TrialResult],
    fit_spec: FitParameterSpec,
    path: str,
) -> None:
    """Save every Monte Carlo trial to CSV."""
    fieldnames = [
        "shots",
        "seed",
        "optimizer_success",
        "normalized_residual_rms",
        "held_out_rms",
        "held_out_max_abs_error",
        "bound_hit_count",
    ]

    for name in fit_spec.names:
        fieldnames.extend(
            [
                f"{name}_fitted",
                f"{name}_error",
                f"{name}_normalized_error",
            ]
        )

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for trial in trials:
            row: Dict[str, object] = {
                "shots": trial.shots,
                "seed": trial.seed,
                "optimizer_success": trial.optimizer_success,
                "normalized_residual_rms": trial.normalized_residual_rms,
                "held_out_rms": trial.held_out_rms,
                "held_out_max_abs_error": trial.held_out_max_abs_error,
                "bound_hit_count": trial.bound_hit_count,
            }

            for name in fit_spec.names:
                row[f"{name}_fitted"] = trial.fitted_values[name]
                row[f"{name}_error"] = trial.errors[name]
                row[f"{name}_normalized_error"] = (
                    trial.normalized_errors[name]
                )

            writer.writerow(row)


def write_summary_csv(
    rows: Sequence[Dict[str, float]],
    path: str,
) -> None:
    """Save parameter-level Monte Carlo summaries to CSV."""
    if not rows:
        raise ValueError("No summary rows are available.")

    fieldnames = list(rows[0].keys())

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def validate_monte_carlo_results(
    trials: Sequence[TrialResult],
    rows: Sequence[Dict[str, float]],
    config: StudyConfig,
    fit_spec: FitParameterSpec,
) -> None:
    """
    Apply minimal M0 statistical checks.

    These checks deliberately avoid requiring every individual parameter RMSE
    to decrease monotonically in a 30-seed pilot study. They enforce optimizer
    reliability and improvement of the aggregate normalized RMSE.
    """
    print("\nM0 finite-shot validation")
    print("=" * 100)

    failed = False
    aggregate_rmse_by_shots: List[Tuple[int, float]] = []

    for shots in config.shots_values:
        shot_trials = [trial for trial in trials if trial.shots == shots]
        success_rate = float(
            np.mean(
                [float(trial.optimizer_success) for trial in shot_trials]
            )
        )
        bound_rate = float(
            np.mean(
                [float(trial.bound_hit_count > 0) for trial in shot_trials]
            )
        )

        shot_rows = [row for row in rows if row["shots"] == shots]
        aggregate_normalized_rmse = float(
            np.median(
                [row["normalized_rmse"] for row in shot_rows]
            )
        )

        aggregate_rmse_by_shots.append(
            (shots, aggregate_normalized_rmse)
        )

        print(
            f"shots={shots:6d}: "
            f"success_rate={success_rate:.3f}, "
            f"bound_hit_rate={bound_rate:.3f}, "
            f"median_normalized_RMSE={aggregate_normalized_rmse:.6e}"
        )

        if success_rate < 0.95:
            print("  FAIL: optimizer success rate is below 0.95")
            failed = True

        if bound_rate > 0.25:
            print("  FAIL: more than 25% of trials hit a parameter bound")
            failed = True

    for previous, current in zip(
        aggregate_rmse_by_shots[:-1],
        aggregate_rmse_by_shots[1:],
    ):
        previous_shots, previous_rmse = previous
        current_shots, current_rmse = current

        if current_rmse >= previous_rmse:
            print(
                "  FAIL: aggregate normalized RMSE did not decrease from "
                f"{previous_shots} to {current_shots} shots"
            )
            failed = True

    if failed:
        raise AssertionError(
            "Finite-shot CZ-core-7 validation failed. Inspect the generated "
            "CSV files before modifying or expanding the physical model."
        )

    print("\nFinite-shot CZ-core-7 validation passed.")


def main() -> None:
    """Run the complete CZ-core-7 M0 recovery study."""
    config = StudyConfig()
    fit_spec = seven_parameter_fit_spec()
    true_point = make_true_point()
    initial_point = make_initial_point()

    design = build_seven_parameter_design(config)
    summarize_design(design, config)

    run_noiseless_multistart(
        design=design,
        config=config,
        fit_spec=fit_spec,
        true_point=true_point,
    )

    trials = run_monte_carlo(
        design=design,
        config=config,
        fit_spec=fit_spec,
        true_point=true_point,
        initial_point=initial_point,
    )

    summary_rows = compute_parameter_summary_rows(
        trials=trials,
        fit_spec=fit_spec,
        true_point=true_point,
        shots_values=config.shots_values,
    )

    print_parameter_summary(
        rows=summary_rows,
        fit_spec=fit_spec,
        shots_values=config.shots_values,
    )

    print_shot_scaling_slopes(
        rows=summary_rows,
        fit_spec=fit_spec,
    )

    write_trial_csv(
        trials=trials,
        fit_spec=fit_spec,
        path=config.trial_csv_path,
    )

    write_summary_csv(
        rows=summary_rows,
        path=config.summary_csv_path,
    )

    print("\nSaved result files")
    print("=" * 100)
    print(config.trial_csv_path)
    print(config.summary_csv_path)

    validate_monte_carlo_results(
        trials=trials,
        rows=summary_rows,
        config=config,
        fit_spec=fit_spec,
    )


if __name__ == "__main__":
    main()


# In[ ]:




