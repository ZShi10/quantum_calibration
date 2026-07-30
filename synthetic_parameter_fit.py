#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
synthetic_parameter_fit.py

Synthetic parameter fitting demo for lightweight MEADD-style calibration.

Run:

    python synthetic_parameter_fit.py

Dependencies:

    multi_qubit_state_space.py
    lightweight_cptp_channel.py
    multi_qubit_circuit_simulator.py
    sensitivity_svd_selector.py

Optional dependency:

    scipy

If scipy is available, this script uses scipy.optimize.least_squares.
If scipy is unavailable, it falls back to a small damped Gauss-Newton solver.

Purpose
-------
This file performs the next step after sensitivity_svd_selector.py:

    1. build candidate MEADD-style circuits
    2. compute finite-difference Jacobian
    3. select compact circuit subset by SVD
    4. generate synthetic experimental data from true parameters
    5. add finite-shot noise
    6. fit selected parameters
    7. compare true vs fitted parameters

Default fitted parameters:

    dphi, swap_x, swap_y, t1, leakage, two_qubit_depolarizing

The swap quadratures are fitted directly instead of theta/chi because chi is
ill-conditioned when theta is small. For small residual swap:

    swap_x = theta cos(chi)
    swap_y = theta sin(chi)

The XX and YX odd-subspace DD probes are intended to select these two
quadratures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares as scipy_least_squares
except Exception:
    scipy_least_squares = None

from parameter_registry import DEFAULT_PARAMETER_REGISTRY

from multi_qubit_circuit_simulator import (
    CircuitSpec,
    simulate_observable_vector,
)

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


@dataclass(frozen=True)
class FitParameterSpec:
    """
    Parameter vector configuration.

    names:
        Parameters to fit.

    scales:
        Internal normalized variable x represents:
            parameter = initial_parameter + scale * x

    lower_bounds / upper_bounds:
        Physical bounds in parameter units.
    """

    names: Tuple[ParameterName, ...]
    scales: Dict[ParameterName, float]
    lower_bounds: Dict[ParameterName, float]
    upper_bounds: Dict[ParameterName, float]

    def validate(self) -> None:
        """
        Validate parameter specification.
        """
        if not self.names:
            raise ValueError("at least one fitted parameter is required")

        for name in self.names:
            if name not in self.scales:
                raise ValueError(f"missing scale for {name}")
            if self.scales[name] <= 0.0:
                raise ValueError(f"scale for {name} must be positive")
            if name not in self.lower_bounds:
                raise ValueError(f"missing lower bound for {name}")
            if name not in self.upper_bounds:
                raise ValueError(f"missing upper bound for {name}")
            if self.lower_bounds[name] >= self.upper_bounds[name]:
                raise ValueError(f"invalid bounds for {name}")


@dataclass(frozen=True)
class SyntheticData:
    """
    Synthetic measured data.
    """

    labels: List[str]
    noiseless_values: np.ndarray
    measured_values: np.ndarray
    standard_errors: np.ndarray
    shots: int


@dataclass(frozen=True)
class FitResult:
    """
    Fitting result.
    """

    fitted_point: ParameterPoint
    initial_point: ParameterPoint
    true_point: ParameterPoint
    fitted_vector: np.ndarray
    residual_vector: np.ndarray
    normalized_residual_rms: float
    parameter_names: Tuple[ParameterName, ...]
    covariance: np.ndarray
    standard_errors: np.ndarray
    success: bool
    message: str
    num_iterations: int


CZ_CORE_7_PARAMETER_NAMES = (
    "dphi",
    "swap_x",
    "swap_y",
    "t1",
    "tphi",
    "leakage",
    "two_qubit_depolarizing",
)

def fit_parameter_spec_from_registry(
    names: Tuple[ParameterName, ...],
) -> FitParameterSpec:
    """Build a fitting specification from registered parameter metadata."""
    scales: Dict[ParameterName, float] = {}
    lower_bounds: Dict[ParameterName, float] = {}
    upper_bounds: Dict[ParameterName, float] = {}

    for name in names:
        parameter_spec = DEFAULT_PARAMETER_REGISTRY.get(name)

        if not parameter_spec.supports_default_fit:
            raise ValueError(
                f"parameter {name} has no default fitting metadata"
            )

        assert parameter_spec.fit_scale is not None
        assert parameter_spec.lower_bound is not None
        assert parameter_spec.upper_bound is not None

        scales[name] = parameter_spec.fit_scale
        lower_bounds[name] = parameter_spec.lower_bound
        upper_bounds[name] = parameter_spec.upper_bound

    fit_spec = FitParameterSpec(
        names=names,
        scales=scales,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    fit_spec.validate()
    return fit_spec

def default_fit_parameter_spec() -> FitParameterSpec:
    """
    Return the default seven-parameter fitting specification.

    The fitted CZ-core-7 parameters are:

        dphi
        swap_x
        swap_y
        t1
        tphi
        leakage
        two_qubit_depolarizing

    Notes
    -----
    In the current simulator, t1 and tphi are per-cycle channel
    probabilities/rates rather than physical relaxation times.
    """
    return fit_parameter_spec_from_registry(
        CZ_CORE_7_PARAMETER_NAMES
    )


def point_to_normalized_vector(
    point: ParameterPoint,
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
) -> np.ndarray:
    """
    Convert parameter point to normalized fit vector.
    """
    fit_spec.validate()

    values = []

    for name in fit_spec.names:
        value = point.get(name)
        reference = reference_point.get(name)
        scale = fit_spec.scales[name]
        values.append((value - reference) / scale)

    return np.asarray(values, dtype=float)


def normalized_vector_to_point(
    vector: np.ndarray,
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    local_dim: int,
) -> ParameterPoint:
    """
    Convert normalized fit vector to ParameterPoint.
    """
    fit_spec.validate()

    if vector.shape != (len(fit_spec.names),):
        raise ValueError("vector has wrong shape")

    point = reference_point

    for index, name in enumerate(fit_spec.names):
        raw_value = reference_point.get(name) + fit_spec.scales[name] * float(vector[index])
        clipped_value = float(
            np.clip(
                raw_value,
                fit_spec.lower_bounds[name],
                fit_spec.upper_bounds[name],
            )
        )
        point = point.with_update(
            name=name,
            value=clipped_value,
            local_dim=local_dim,
        )

    return point


def normalized_bounds(
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return lower/upper bounds in normalized coordinates.
    """
    lower = []
    upper = []

    for name in fit_spec.names:
        reference = reference_point.get(name)
        scale = fit_spec.scales[name]
        lower.append((fit_spec.lower_bounds[name] - reference) / scale)
        upper.append((fit_spec.upper_bounds[name] - reference) / scale)

    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def infer_local_dim(circuits: Sequence[CircuitSpec]) -> int:
    """
    Infer common local dimension.
    """
    if not circuits:
        raise ValueError("circuits must be non-empty")

    local_dim = circuits[0].local_dim

    for circuit in circuits:
        if circuit.local_dim != local_dim:
            raise ValueError("all circuits must have the same local_dim")

    return local_dim


def simulate_selected_vector(
    circuits: Sequence[CircuitSpec],
    point: ParameterPoint,
) -> Tuple[List[str], np.ndarray]:
    """
    Simulate selected circuit observable vector at one parameter point.
    """
    local_dim = infer_local_dim(circuits)

    return simulate_observable_vector(
        circuits=circuits,
        coherent_params=point.coherent_params(),
        config=point.cptp_config(local_dim=local_dim),
    )


def generate_synthetic_data(
    circuits: Sequence[CircuitSpec],
    true_point: ParameterPoint,
    shots: int,
    rng: np.random.Generator,
    min_standard_error: float = 1.0e-3,
) -> SyntheticData:
    """
    Generate synthetic measured expectation values.

    For each observable m in [-1,1], use binary measurement approximation:
        p = (1 + m) / 2
        k ~ Binomial(shots, p)
        measured m_hat = 2k/shots - 1

    Leakage observables are handled as probabilities:
        p = leakage
        measured leakage = k/shots

    The standard error is estimated from binomial variance and clipped below
    by min_standard_error to avoid over-weighting saturated observables.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    labels, noiseless_values = simulate_selected_vector(
        circuits=circuits,
        point=true_point,
    )

    measured_values = np.zeros_like(noiseless_values, dtype=float)
    standard_errors = np.zeros_like(noiseless_values, dtype=float)

    for index, (label, value) in enumerate(zip(labels, noiseless_values)):
        if label.endswith("/leakage"):
            probability = float(np.clip(value, 0.0, 1.0))
            counts = rng.binomial(shots, probability)
            measured = counts / shots
            variance = probability * (1.0 - probability) / shots
            standard_error = np.sqrt(max(variance, 0.0))
        else:
            clipped = float(np.clip(value, -1.0, 1.0))
            probability = 0.5 * (1.0 + clipped)
            counts = rng.binomial(shots, probability)
            measured = 2.0 * counts / shots - 1.0
            variance = max(1.0 - clipped * clipped, 0.0) / shots
            standard_error = np.sqrt(variance)

        measured_values[index] = measured
        standard_errors[index] = max(float(standard_error), min_standard_error)

    return SyntheticData(
        labels=labels,
        noiseless_values=noiseless_values,
        measured_values=measured_values,
        standard_errors=standard_errors,
        shots=shots,
    )


def residual_function(
    normalized_vector: np.ndarray,
    circuits: Sequence[CircuitSpec],
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    data: SyntheticData,
) -> np.ndarray:
    """
    Weighted residual vector:
        r_i = (model_i - data_i) / sigma_i
    """
    local_dim = infer_local_dim(circuits)

    point = normalized_vector_to_point(
        vector=normalized_vector,
        reference_point=reference_point,
        fit_spec=fit_spec,
        local_dim=local_dim,
    )

    labels, model_values = simulate_selected_vector(
        circuits=circuits,
        point=point,
    )

    if labels != data.labels:
        raise RuntimeError("observable labels changed during fitting")

    return (model_values - data.measured_values) / data.standard_errors


def finite_difference_residual_jacobian(
    x: np.ndarray,
    residual_at_x: np.ndarray,
    circuits: Sequence[CircuitSpec],
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    data: SyntheticData,
    step: float = 1.0e-4,
) -> np.ndarray:
    """
    Finite-difference Jacobian of weighted residuals with respect to
    normalized fit coordinates.
    """
    jacobian = np.zeros((residual_at_x.size, x.size), dtype=float)

    for col in range(x.size):
        dx = np.zeros_like(x)
        dx[col] = step

        plus = residual_function(
            normalized_vector=x + dx,
            circuits=circuits,
            reference_point=reference_point,
            fit_spec=fit_spec,
            data=data,
        )
        minus = residual_function(
            normalized_vector=x - dx,
            circuits=circuits,
            reference_point=reference_point,
            fit_spec=fit_spec,
            data=data,
        )

        jacobian[:, col] = (plus - minus) / (2.0 * step)

    return jacobian


def damped_gauss_newton_fit(
    circuits: Sequence[CircuitSpec],
    initial_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    data: SyntheticData,
    max_iterations: int = 30,
    damping: float = 1.0e-2,
    tolerance: float = 1.0e-7,
) -> Tuple[np.ndarray, bool, str, int]:
    """
    Small fallback optimizer used when scipy is unavailable.

    This is not intended to replace scipy.optimize.least_squares, but is
    sufficient for the small synthetic demos in this project.
    """
    local_dim = infer_local_dim(circuits)

    x = np.zeros(len(fit_spec.names), dtype=float)
    lower, upper = normalized_bounds(
        reference_point=initial_point,
        fit_spec=fit_spec,
    )

    x = np.clip(x, lower, upper)
    previous_cost = np.inf

    for iteration in range(1, max_iterations + 1):
        residual = residual_function(
            normalized_vector=x,
            circuits=circuits,
            reference_point=initial_point,
            fit_spec=fit_spec,
            data=data,
        )

        cost = 0.5 * float(np.dot(residual, residual))

        if abs(previous_cost - cost) < tolerance * max(1.0, previous_cost):
            return x, True, "fallback Gauss-Newton converged", iteration

        previous_cost = cost

        jacobian = finite_difference_residual_jacobian(
            x=x,
            residual_at_x=residual,
            circuits=circuits,
            reference_point=initial_point,
            fit_spec=fit_spec,
            data=data,
        )

        lhs = jacobian.T @ jacobian + damping * np.eye(x.size)
        rhs = -jacobian.T @ residual

        try:
            step = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        accepted = False

        for shrink in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate = np.clip(x + shrink * step, lower, upper)
            candidate_residual = residual_function(
                normalized_vector=candidate,
                circuits=circuits,
                reference_point=initial_point,
                fit_spec=fit_spec,
                data=data,
            )
            candidate_cost = 0.5 * float(np.dot(candidate_residual, candidate_residual))

            if candidate_cost < cost:
                x = candidate
                accepted = True
                break

        if not accepted:
            damping *= 10.0
        else:
            damping = max(damping * 0.5, 1.0e-8)

        if np.linalg.norm(step) < tolerance * max(1.0, np.linalg.norm(x)):
            return x, True, "fallback Gauss-Newton step tolerance reached", iteration

    return x, False, "fallback Gauss-Newton reached max_iterations", max_iterations


def estimate_covariance(
    x: np.ndarray,
    circuits: Sequence[CircuitSpec],
    reference_point: ParameterPoint,
    fit_spec: FitParameterSpec,
    data: SyntheticData,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate covariance in physical parameter units.

    Uses:
        cov_x ≈ inv(J^T J) * reduced_chi2

    where J is the Jacobian of weighted residuals with respect to normalized
    coordinates.
    """
    residual = residual_function(
        normalized_vector=x,
        circuits=circuits,
        reference_point=reference_point,
        fit_spec=fit_spec,
        data=data,
    )

    jacobian = finite_difference_residual_jacobian(
        x=x,
        residual_at_x=residual,
        circuits=circuits,
        reference_point=reference_point,
        fit_spec=fit_spec,
        data=data,
    )

    dof = max(1, residual.size - x.size)
    reduced_chi2 = float(np.dot(residual, residual) / dof)

    fisher = jacobian.T @ jacobian

    try:
        cov_x = np.linalg.inv(fisher) * reduced_chi2
    except np.linalg.LinAlgError:
        cov_x = np.linalg.pinv(fisher) * reduced_chi2

    scale_vector = np.asarray(
        [fit_spec.scales[name] for name in fit_spec.names],
        dtype=float,
    )

    covariance = cov_x * scale_vector[:, None] * scale_vector[None, :]
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return covariance, standard_errors


def fit_synthetic_data(
    circuits: Sequence[CircuitSpec],
    data: SyntheticData,
    initial_point: ParameterPoint,
    true_point: ParameterPoint,
    fit_spec: FitParameterSpec,
) -> FitResult:
    """
    Fit synthetic data.
    """
    local_dim = infer_local_dim(circuits)
    fit_spec.validate()

    x0 = np.zeros(len(fit_spec.names), dtype=float)
    lower, upper = normalized_bounds(
        reference_point=initial_point,
        fit_spec=fit_spec,
    )

    if scipy_least_squares is not None:
        result = scipy_least_squares(
            fun=lambda x: residual_function(
                normalized_vector=x,
                circuits=circuits,
                reference_point=initial_point,
                fit_spec=fit_spec,
                data=data,
            ),
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=80,
            xtol=1.0e-8,
            ftol=1.0e-8,
            gtol=1.0e-8,
        )

        fitted_x = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = str(result.message)
        num_iterations = int(result.nfev)
    else:
        fitted_x, success, message, num_iterations = damped_gauss_newton_fit(
            circuits=circuits,
            initial_point=initial_point,
            fit_spec=fit_spec,
            data=data,
        )

    fitted_point = normalized_vector_to_point(
        vector=fitted_x,
        reference_point=initial_point,
        fit_spec=fit_spec,
        local_dim=local_dim,
    )

    fitted_labels, fitted_vector = simulate_selected_vector(
        circuits=circuits,
        point=fitted_point,
    )

    if fitted_labels != data.labels:
        raise RuntimeError("observable labels changed after fitting")

    residual = (fitted_vector - data.measured_values) / data.standard_errors
    normalized_residual_rms = float(np.sqrt(np.mean(residual * residual)))

    covariance, standard_errors = estimate_covariance(
        x=fitted_x,
        circuits=circuits,
        reference_point=initial_point,
        fit_spec=fit_spec,
        data=data,
    )

    return FitResult(
        fitted_point=fitted_point,
        initial_point=initial_point,
        true_point=true_point,
        fitted_vector=fitted_vector,
        residual_vector=residual,
        normalized_residual_rms=normalized_residual_rms,
        parameter_names=fit_spec.names,
        covariance=covariance,
        standard_errors=standard_errors,
        success=success,
        message=message,
        num_iterations=num_iterations,
    )


def make_default_design_point() -> ParameterPoint:
    """
    Return the nominal parameter point used for Jacobian-based circuit
    selection.
    """
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


def make_default_true_point() -> ParameterPoint:
    """
    Return the default seven-parameter synthetic truth.

    Every fitted parameter differs from its design-point value. In
    particular, tphi deliberately differs from the initial value so that
    recovery tests cannot pass by keeping tphi fixed.
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


def build_selected_circuits_for_fit(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    max_depth: int = 12,
    max_candidates: int = 80,
    top_k: int = 12,
) -> Tuple[List[CircuitSpec], JacobianResult, CircuitSelectionResult]:
    """
    Build the SVD-selected circuit subset for the seven-parameter CZ-core
    model.
    """
    candidates = build_pc_friendly_candidates(
        num_qubits=num_qubits,
        local_dim=local_dim,
        target_pair=target_pair,
        max_depth=max_depth,
        max_candidates=max_candidates,
    )

    base_point = make_default_design_point()

    fd_config = FiniteDifferenceConfig(
        target_parameters=CZ_CORE_7_PARAMETER_NAMES,
        use_central_difference=True,
    )

    jacobian_result = compute_finite_difference_jacobian(
        circuits=candidates,
        base_point=base_point,
        fd_config=fd_config,
    )

    if jacobian_result.jacobian.shape[1] != len(
        CZ_CORE_7_PARAMETER_NAMES
    ):
        raise RuntimeError(
            "Default CZ-core Jacobian must contain exactly seven columns."
        )

    selection = select_circuits_greedy_svd(
        circuits=candidates,
        jac_res=jacobian_result,
        top_k=top_k,
        use_norm=True,
    )

    selected_report = compute_svd_report(
        selection.selected_jacobian,
        rank_tol=1.0e-10,
    )

    if selected_report.numerical_rank != len(
        CZ_CORE_7_PARAMETER_NAMES
    ):
        raise RuntimeError(
            "Selected circuit subset is not full rank for CZ-core-7: "
            f"rank={selected_report.numerical_rank}, "
            f"required={len(CZ_CORE_7_PARAMETER_NAMES)}"
        )

    return (
        selection.selected_circuits,
        jacobian_result,
        selection,
    )


def summarize_selected_circuits(selection: CircuitSelectionResult) -> None:
    """
    Print selected circuit list.
    """
    print("\nSelected circuits for fitting")
    print("=" * 88)

    for index, circuit in enumerate(selection.selected_circuits, start=1):
        print(
            f"{index:02d}. {circuit.name:44s} "
            f"n={circuit.repetitions:<3d} "
            f"measurements={len(circuit.measurements)}"
        )

    print("\nSelected-subset normalized SVD")
    print("-" * 88)
    print(f"singular values  = {np.array2string(selection.selected_singular_values, precision=4)}")
    print(f"condition number = {selection.selected_condition_number:.6e}")
    print(f"observables      = {len(selection.selected_observable_labels)}")


def summarize_fit_result(result: FitResult) -> None:
    """
    Print true vs fitted parameter table.
    """
    print("\nFit result")
    print("=" * 88)
    print(f"success                 = {result.success}")
    print(f"message                 = {result.message}")
    print(f"function evaluations    = {result.num_iterations}")
    print(f"normalized residual RMS = {result.normalized_residual_rms:.6f}")

    print("\nParameter comparison")
    print("-" * 88)
    print(
        f"{'parameter':26s} "
        f"{'initial':>14s} "
        f"{'true':>14s} "
        f"{'fitted':>14s} "
        f"{'fit-true':>14s} "
        f"{'1sigma':>14s}"
    )

    for index, name in enumerate(result.parameter_names):
        initial = result.initial_point.get(name)
        true = result.true_point.get(name)
        fitted = result.fitted_point.get(name)
        error = fitted - true
        sigma = result.standard_errors[index]

        print(
            f"{name:26s} "
            f"{initial:14.6e} "
            f"{true:14.6e} "
            f"{fitted:14.6e} "
            f"{error:14.6e} "
            f"{sigma:14.6e}"
        )


def summarize_data_quality(data: SyntheticData, max_lines: int = 12) -> None:
    """
    Print a compact view of synthetic data.
    """
    print("\nSynthetic data preview")
    print("=" * 88)
    print(f"shots per observable = {data.shots}")
    print(f"num observables      = {len(data.labels)}")

    print("\nFirst observables")
    print("-" * 88)

    for index in range(min(max_lines, len(data.labels))):
        print(
            f"{data.labels[index]:58s} "
            f"ideal={data.noiseless_values[index]: .7f} "
            f"meas={data.measured_values[index]: .7f} "
            f"sigma={data.standard_errors[index]:.3e}"
        )


def run_default_3q_fit_demo() -> None:
    """
    Run the default seven-parameter 3-qutrit synthetic fitting demo.
    """
    print("\n3-qutrit CZ-core-7 synthetic parameter fit")
    print("=" * 88)

    selected_circuits, _, selection = build_selected_circuits_for_fit(
        num_qubits=3,
        local_dim=3,
        target_pair=(0, 1),
        max_depth=12,
        max_candidates=80,
        top_k=12,
    )

    summarize_selected_circuits(selection)

    true_point = make_default_true_point()
    initial_point = make_default_design_point()

    rng = np.random.default_rng(1234)

    data = generate_synthetic_data(
        circuits=selected_circuits,
        true_point=true_point,
        shots=4000,
        rng=rng,
        min_standard_error=1.0e-3,
    )

    summarize_data_quality(data)

    fit_spec = default_fit_parameter_spec()

    result = fit_synthetic_data(
        circuits=selected_circuits,
        data=data,
        initial_point=initial_point,
        true_point=true_point,
        fit_spec=fit_spec,
    )

    summarize_fit_result(result)


def run_noiseless_recovery_test() -> None:
    """
    Verify exact seven-parameter recovery without shot noise.

    This is a lightweight regression test. The more comprehensive multi-start
    and finite-shot validation remains in cz_core_7_recovery_study.py.
    """
    print("\nCZ-core-7 noiseless recovery regression test")
    print("=" * 88)

    selected_circuits, _, _ = build_selected_circuits_for_fit(
        num_qubits=3,
        local_dim=3,
        target_pair=(0, 1),
        max_depth=12,
        max_candidates=80,
        top_k=12,
    )

    true_point = make_default_true_point()
    initial_point = make_default_design_point()

    labels, noiseless_values = simulate_selected_vector(
        circuits=selected_circuits,
        point=true_point,
    )

    noiseless_values = np.asarray(noiseless_values, dtype=float)

    noiseless_data = SyntheticData(
        labels=list(labels),
        noiseless_values=noiseless_values.copy(),
        measured_values=noiseless_values.copy(),
        standard_errors=np.ones_like(noiseless_values),
        shots=0,
    )

    fit_spec = default_fit_parameter_spec()

    result = fit_synthetic_data(
        circuits=selected_circuits,
        data=noiseless_data,
        initial_point=initial_point,
        true_point=true_point,
        fit_spec=fit_spec,
    )

    summarize_fit_result(result)

    normalized_errors = {
        name: (
            result.fitted_point.get(name)
            - true_point.get(name)
        ) / fit_spec.scales[name]
        for name in fit_spec.names
    }

    max_normalized_error = max(
        abs(error) for error in normalized_errors.values()
    )

    print(
        "\nmax scale-normalized fitted parameter error = "
        f"{max_normalized_error:.6e}"
    )

    if not result.success:
        raise AssertionError(
            f"Noiseless optimizer failed: {result.message}"
        )

    if result.normalized_residual_rms > 5.0e-8:
        raise AssertionError(
            "Noiseless normalized residual RMS is larger than expected: "
            f"{result.normalized_residual_rms:.6e}"
        )

    if max_normalized_error > 1.0e-4:
        raise AssertionError(
            "Noiseless normalized parameter error is larger than expected: "
            f"{max_normalized_error:.6e}"
        )

    print("\nCZ-core-7 noiseless regression test passed.")


def run_shot_scaling_demo() -> None:
    """
    Show how fitting improves with more shots.
    """
    print("\nShot scaling demo")
    print("=" * 88)

    selected_circuits, _, _ = build_selected_circuits_for_fit(
        num_qubits=3,
        local_dim=3,
        target_pair=(0, 1),
        max_depth=12,
        max_candidates=80,
        top_k=12,
    )

    true_point = ParameterPoint(
        dphi=-0.013,
        theta=0.0052,
        chi=0.3,
        swap_x=0.0052 * np.cos(0.3),
        swap_y=0.0052 * np.sin(0.3),
        t1=2.6e-4,
        tphi=3.0e-4,
        local_depolarizing=1.0e-4,
        leakage=7.0e-5,
        seepage=1.0e-4,
        two_qubit_depolarizing=7.5e-4,
        readout_0_to_1=0.01,
        readout_1_to_0=0.015,
    )

    initial_point = ParameterPoint(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
        swap_x=0.004 * np.cos(0.3),
        swap_y=0.004 * np.sin(0.3),
        t1=2.0e-4,
        tphi=3.0e-4,
        local_depolarizing=1.0e-4,
        leakage=5.0e-5,
        seepage=1.0e-4,
        two_qubit_depolarizing=5.0e-4,
        readout_0_to_1=0.01,
        readout_1_to_0=0.015,
    )

    fit_spec = default_fit_parameter_spec()

    print(
    f"{'shots':>8s} "
    f"{'rms':>10s} "
    f"{'dphi_err':>12s} "
    f"{'swap_x_err':>12s} "
    f"{'swap_y_err':>12s} "
    f"{'t1_err':>12s} "
    f"{'leak_err':>12s} "
    f"{'depol_err':>12s}"
    )
    print("-" * 88)

    for shots in (1000, 4000, 16000):
        rng = np.random.default_rng(1000 + shots)

        data = generate_synthetic_data(
            circuits=selected_circuits,
            true_point=true_point,
            shots=shots,
            rng=rng,
            min_standard_error=1.0e-3,
        )

        result = fit_synthetic_data(
            circuits=selected_circuits,
            data=data,
            initial_point=initial_point,
            true_point=true_point,
            fit_spec=fit_spec,
        )

        errors = {
            name: result.fitted_point.get(name) - true_point.get(name)
            for name in fit_spec.names
        }

        print(
            f"{shots:8d} "
            f"{result.normalized_residual_rms:10.4f} "
            f"{errors['dphi']:12.3e} "
            f"{errors['swap_x']:12.3e} "
            f"{errors['swap_y']:12.3e} "
            f"{errors['t1']:12.3e} "
            f"{errors['leakage']:12.3e} "
            f"{errors['two_qubit_depolarizing']:12.3e}"
       )


def main() -> None:
    """
    Lightweight CLI entry point.

    Statistical Monte Carlo validation is intentionally not run here.
    Run cz_core_7_recovery_study.py for the formal M0 study.
    """
    run_default_3q_fit_demo()
    run_noiseless_recovery_test()


if __name__ == "__main__":
    main()

# In[ ]:




