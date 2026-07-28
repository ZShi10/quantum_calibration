#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from parameter_registry import DEFAULT_PARAMETER_REGISTRY
from lightweight_cptp_channel import (
    CPTPModelConfig,
    LocalNoiseRates,
    ReadoutNoise,
    TwoQubitCoherentParams,
    TwoQubitNoiseRates,
)

from multi_qubit_circuit_simulator import (
    CircuitSpec,
    make_default_probe_circuits,
    simulate_observable_vector,
)

ParameterName = Literal[
    "dphi",
    "theta",
    "chi",
    "swap_x",
    "swap_y",
    "t1",
    "tphi",
    "local_depolarizing",
    "leakage",
    "seepage",
    "two_qubit_depolarizing",
    "readout_0_to_1",
    "readout_1_to_0",
]

@dataclass(frozen=True)
class ParameterPoint:
    """
    Combined coherent + CPTP parameter point.

    Coherent parameters:
        dphi
        swap_x = theta cos(chi)
        swap_y = theta sin(chi)

    Internal logic:
        The simulator and physics layers consume TwoQubitCoherentParams.
        Fitting uses swap_x/swap_y because chi is ill-conditioned at small theta.
    """

    dphi: float = -0.01
    theta: float = 0.004
    chi: float = 0.3
    swap_x: float = 3.821346e-3  # Derived from theta=0.004, chi=0.3
    swap_y: float = 1.182081e-3
    t1: float = 2.0e-4
    tphi: float = 3.0e-4
    local_depolarizing: float = 1.0e-4
    leakage: float = 5.0e-5
    seepage: float = 1.0e-4
    two_qubit_depolarizing: float = 5.0e-4
    readout_0_to_1: float = 0.01
    readout_1_to_0: float = 0.015

    def coherent_params(self) -> TwoQubitCoherentParams:
        """
        Convert to coherent CZ-like params.
        """
        return TwoQubitCoherentParams(
            dphi=float(self.dphi),
            theta=float(self.theta),
            chi=float(self.chi),
            swap_x=float(self.swap_x),
            swap_y=float(self.swap_y),
        )

    def cptp_config(self, local_dim: int) -> CPTPModelConfig:
        """
        Convert to CPTPModelConfig based on current parameter coordinates.
        Note: The unified qutrit model in lightweight_cptp_channel uses
        'leakage' and 'two_qubit_depolarizing' for the gate Kraus set.
        """
        local_noise = LocalNoiseRates(
            amplitude_damping=float(self.t1),
            pure_dephasing=float(self.tphi),
            depolarizing=float(self.local_depolarizing),
            leakage=float(self.leakage) if local_dim >= 3 else 0.0,
            seepage=float(self.seepage) if local_dim >= 3 else 0.0,
        )

        two_qubit_noise = TwoQubitNoiseRates(
            two_qubit_depolarizing=float(self.two_qubit_depolarizing),
            zz_phase_jitter_std=0.0,
        )

        readout_noise = ReadoutNoise(
            assignment_error_0_to_1=float(self.readout_0_to_1),
            assignment_error_1_to_0=float(self.readout_1_to_0),
            leakage_report_as_1=0.5,
        )

        config = CPTPModelConfig(
            local_noise=local_noise,
            two_qubit_noise=two_qubit_noise,
            readout_noise=readout_noise,
        )
        config.validate(local_dim=local_dim)
        return config

    def get(self, name: ParameterName) -> float:
        return float(getattr(self, name))

    def with_update(
        self,
        name: ParameterName,
        value: float,
        local_dim: int,
    ) -> "ParameterPoint":
        """
        Update a parameter and maintain consistency between swap_xy and theta/chi.
        """
        updates = self.__dict__.copy()
        value = float(value)

        # Clipping for physical rates
        if name in {
            "t1", "tphi", "local_depolarizing", "leakage", "seepage",
            "two_qubit_depolarizing", "readout_0_to_1", "readout_1_to_0",
        }:
            value = float(np.clip(value, 0.0, 1.0))

        if local_dim < 3 and name in {"leakage", "seepage"}:
            value = 0.0

        updates[name] = value

        # Sync coordinate systems
        if name in {"theta", "chi"}:
            t = float(updates["theta"])
            c = float(updates["chi"])
            updates["swap_x"] = t * float(np.cos(c))
            updates["swap_y"] = t * float(np.sin(c))
        elif name in {"swap_x", "swap_y"}:
            sx = float(updates["swap_x"])
            sy = float(updates["swap_y"])
            updates["theta"] = float(np.hypot(sx, sy))
            updates["chi"] = float(np.arctan2(sy, sx))

        return ParameterPoint(**updates)

@dataclass(frozen=True)
class FiniteDifferenceConfig:
    """
    Finite-difference settings for sensitivity analysis.
    """
    target_parameters: Tuple[ParameterName, ...] = (
        "dphi",
        "swap_x",
        "swap_y",
        "t1",
        "tphi",
        "leakage",
        "two_qubit_depolarizing",
    )
    step_sizes: Optional[Dict[ParameterName, float]] = None
    use_central_difference: bool = True

    def step_for(self, name: ParameterName) -> float:
        if self.step_sizes is not None and name in self.step_sizes:
            step = float(self.step_sizes[name])
        else:
            step = DEFAULT_PARAMETER_REGISTRY.finite_difference_step(name)

        if step <= 0.0:
            raise ValueError(
                f"finite-difference step for {name} must be positive"
            )

        return step

@dataclass(frozen=True)
class JacobianResult:
    """
    Results of the Jacobian calculation.
    """
    circuits: Tuple[CircuitSpec, ...]
    observable_labels: Tuple[str, ...]
    parameter_names: Tuple[ParameterName, ...]
    base_vector: np.ndarray
    jacobian: np.ndarray
    normalized_jacobian: np.ndarray
    parameter_scales: Dict[ParameterName, float]

    @property
    def labels(self) -> List[str]:
        return list(self.observable_labels)

    @property
    def column_norms(self) -> np.ndarray:
        return np.asarray([self.parameter_scales[n] for n in self.parameter_names], dtype=float)

@dataclass(frozen=True)
class SVDReport:
    singular_values: np.ndarray
    condition_number: float
    numerical_rank: int
    right_singular_vectors: np.ndarray

@dataclass(frozen=True)
class CircuitSelectionResult:
    selected_circuits: List[CircuitSpec]
    selected_indices: List[int]
    selected_observable_labels: List[str]
    selected_jacobian: np.ndarray
    selected_singular_values: np.ndarray
    selected_condition_number: float
    all_singular_values: np.ndarray
    all_condition_number: float

def infer_local_dim(circuits: Sequence[CircuitSpec]) -> int:
    if not circuits: raise ValueError("circuits must be non-empty")
    ld = circuits[0].local_dim
    if any(c.local_dim != ld for c in circuits):
        raise ValueError("all circuits must have same local_dim")
    return ld

def simulate_at_parameter_point(
    circuits: Sequence[CircuitSpec],
    parameter_point: ParameterPoint,
) -> Tuple[List[str], np.ndarray]:
    local_dim = infer_local_dim(circuits)
    return simulate_observable_vector(
        circuits=circuits,
        coherent_params=parameter_point.coherent_params(),
        config=parameter_point.cptp_config(local_dim=local_dim),
    )

def validate_observable_layout(circuits, labels, vector) -> None:
    if len(labels) != int(vector.shape[0]):
        raise ValueError(f"Label mismatch: {len(labels)} vs {vector.shape[0]}")
    if not circuits: raise ValueError("No circuits.")

def compute_parameter_scales(jacobian, parameter_names) -> Dict[ParameterName, float]:
    column_norms = np.linalg.norm(jacobian, axis=0)
    return {n: float(column_norms[i]) for i, n in enumerate(parameter_names)}

def normalize_jacobian_columns(jacobian, parameter_names, parameter_scales) -> np.ndarray:
    normalized = jacobian.copy()
    for col, name in enumerate(parameter_names):
        scale = float(parameter_scales[name])
        if scale > 0.0: normalized[:, col] /= scale
    return normalized

def compute_finite_difference_jacobian(
    circuits: Sequence[CircuitSpec],
    base_point: ParameterPoint,
    fd_config: FiniteDifferenceConfig,
) -> JacobianResult:
    if not circuits: raise ValueError("No circuits.")
    
    circ_tuple = tuple(circuits)
    local_dim = infer_local_dim(circ_tuple)

    labels, base_vector = simulate_at_parameter_point(circ_tuple, base_point)
    obs_labels = tuple(labels)
    validate_observable_layout(circ_tuple, obs_labels, base_vector)

    param_names = tuple(fd_config.target_parameters)
    num_vals, num_params = len(base_vector), len(param_names)
    jacobian = np.zeros((num_vals, num_params), dtype=float)

    for col, name in enumerate(param_names):
        step = fd_config.step_for(name)
        base_val = base_point.get(name)

        if fd_config.use_central_difference:
            p_point = base_point.with_update(name, base_val + step, local_dim)
            m_point = base_point.with_update(name, base_val - step, local_dim)
            _, p_vec = simulate_at_parameter_point(circ_tuple, p_point)
            _, m_vec = simulate_at_parameter_point(circ_tuple, m_point)
            denom = p_point.get(name) - m_point.get(name)
            jacobian[:, col] = (p_vec - m_vec) / denom
        else:
            p_point = base_point.with_update(name, base_val + step, local_dim)
            _, p_vec = simulate_at_parameter_point(circ_tuple, p_point)
            denom = p_point.get(name) - base_val
            jacobian[:, col] = (p_vec - base_vector) / denom

    scales = compute_parameter_scales(jacobian, param_names)
    norm_j = normalize_jacobian_columns(jacobian, param_names, scales)

    return JacobianResult(circ_tuple, obs_labels, param_names, base_vector, jacobian, norm_j, scales)

def compute_svd_report(matrix: np.ndarray, rank_tol: float = 1.0e-10) -> SVDReport:
    if matrix.size == 0:
        return SVDReport(np.array([]), np.inf, 0, np.zeros((0,0)))
    _, s, vt = np.linalg.svd(matrix, full_matrices=False)
    m_sv = float(s[0]) if s.size > 0 else 0.0
    thresh = rank_tol * max(1.0, m_sv)
    rank = int(np.sum(s > thresh))
    cond = float(s[0] / s[-1]) if (s.size > 0 and s[-1] > thresh) else np.inf
    return SVDReport(s, cond, rank, vt)

def circuit_observable_slices(circuits, labels) -> List[np.ndarray]:
    res: List[np.ndarray] = []
    for c in circuits:
        prefix = f"{c.name}/"
        res.append(np.asarray([i for i, l in enumerate(labels) if l.startswith(prefix)], dtype=int))
    return res

def get_jacobian_matrix(res: JacobianResult, use_normalized: bool = True) -> np.ndarray:
    return res.normalized_jacobian if use_normalized else res.jacobian

def score_candidate_addition(current_rows, candidate_rows, matrix, min_sv_w=1.0, logdet_w=0.1) -> float:
    rows = np.unique(np.concatenate([current_rows, candidate_rows]))
    if rows.size == 0: return -np.inf
    report = compute_svd_report(matrix[rows, :])
    if report.singular_values.size == 0: return -np.inf
    s = report.singular_values
    pos = s[s > 1.0e-12]
    if pos.size == 0: return -np.inf
    min_sv = float(np.min(pos))
    log_vol = float(np.sum(np.log(pos + 1.0e-15)))
    penalty = 0.05 * np.log1p(report.condition_number) if np.isfinite(report.condition_number) else 5.0
    return min_sv_w * min_sv + logdet_w * log_vol + 10.0 * report.numerical_rank - penalty

def select_circuits_greedy_svd(circuits, jac_res, top_k, use_norm=True) -> CircuitSelectionResult:
    circ_tuple = tuple(circuits)
    matrix = get_jacobian_matrix(jac_res, use_norm)
    slices = circuit_observable_slices(circ_tuple, jac_res.observable_labels)
    all_report = compute_svd_report(matrix)

    selected_idxs: List[int] = []
    selected_rows = np.array([], dtype=int)
    remaining = set(range(len(circ_tuple)))

    for _ in range(min(top_k, len(circ_tuple))):
        best_idx, best_score = None, -np.inf
        for cand_idx in sorted(remaining):
            rows = slices[cand_idx]
            if rows.size == 0: continue
            score = score_candidate_addition(selected_rows, rows, matrix)
            if score > best_score:
                best_score, best_idx = score, cand_idx
        if best_idx is None: break
        selected_idxs.append(best_idx)
        selected_rows = np.unique(np.concatenate([selected_rows, slices[best_idx]]))
        remaining.remove(best_idx)

    sel_circs = [circ_tuple[i] for i in selected_idxs]
    sel_jac = matrix[selected_rows, :]
    sel_report = compute_svd_report(sel_jac)
    sel_labels = [jac_res.observable_labels[i] for i in selected_rows.tolist()]

    return CircuitSelectionResult(sel_circs, selected_idxs, sel_labels, sel_jac, sel_report.singular_values, sel_report.condition_number, all_report.singular_values, all_report.condition_number)

def rank_observables_by_sensitivity(jac_res, use_norm=True, top_n=20) -> List[Tuple[str, float]]:
    matrix = get_jacobian_matrix(jac_res, use_norm)
    norms = np.linalg.norm(matrix, axis=1)
    order = np.argsort(norms)[::-1]
    return [(jac_res.observable_labels[i], float(norms[i])) for i in order[:top_n]]

def summarize_jacobian(res: JacobianResult) -> None:
    print("\nJacobian summary")
    print("=" * 88)
    print(f"num circuits    = {len(res.circuits)}")
    print(f"num observables = {res.jacobian.shape[0]}")
    print(f"num parameters  = {res.jacobian.shape[1]}")
    print("\nColumn norms")
    print("-" * 88)
    for n in res.parameter_names:
        print(f"{n:26s} {res.parameter_scales[n]:.6e}")
    
    r_rep = compute_svd_report(res.jacobian)
    n_rep = compute_svd_report(res.normalized_jacobian)
    print(f"\nRaw-J Condition: {r_rep.condition_number:.2e} | Rank: {r_rep.numerical_rank}")
    print(f"Normalized-J Condition: {n_rep.condition_number:.2e} | Rank: {n_rep.numerical_rank}")

def summarize_observable_ranking(jac_res, top_n=20) -> None:
    print("\nTop observable sensitivities")
    print("=" * 88)
    ranking = rank_observables_by_sensitivity(jac_res, True, top_n)
    for label, score in ranking:
        print(f"{label:60s} {score:.6e}")

def summarize_selection(sel: CircuitSelectionResult) -> None:
    print("\nSelected circuits")
    print("=" * 88)
    for r, c in enumerate(sel.selected_circuits, 1):
        print(f"{r:02d}. {c.name:44s} n={c.repetitions:<3d} target={c.target_pair}")
    print(f"\nSelected Condition Number: {sel.selected_condition_number:.4e}")
    print(f"Selected Observables: {len(sel.selected_observable_labels)}")

def build_pc_friendly_candidates(num_qubits=3, local_dim=3, target_pair=(0, 1), max_depth=12, max_candidates=80) -> List[CircuitSpec]:
    circs = make_default_probe_circuits(num_qubits, local_dim, target_pair, max_depth)
    return circs[:max_candidates]

def check_observable_ranges(labels, values, tolerance=1.05) -> List[Tuple[str, float]]:
    suspicious = []
    for l, v in zip(labels, values):
        if not l.endswith("/leakage") and abs(float(v)) > tolerance:
            suspicious.append((l, float(v)))
    return suspicious

def run_3q_svd_demo() -> None:
    """
    Standard demo for 3-qutrit Jacobian and SVD selection using unified noise model.
    """
    print("\n3-qutrit SVD selector demo (Unified Noise Model)")
    print("=" * 88)

    circuits = build_pc_friendly_candidates(num_qubits=3, local_dim=3, target_pair=(0, 1))

    # Initial point emphasizing the core qutrit gate leakage
    base_point = ParameterPoint(
        dphi=-0.015,
        swap_x=0.0035,
        swap_y=0.0012,
        leakage=1.2e-4,  # This now goes into gate-level Kraus
        two_qubit_depolarizing=8.0e-4
    )

    fd_config = FiniteDifferenceConfig(
        target_parameters=("dphi", "swap_x", "swap_y", "t1", "tphi", "leakage", "two_qubit_depolarizing")
    )

    jac_res = compute_finite_difference_jacobian(circuits, base_point, fd_config)
    
    suspicious = check_observable_ranges(jac_res.observable_labels, jac_res.base_vector)
    if suspicious:
        print("\nWARNING: Suspicious Pauli values (>1.05) detected.")

    summarize_jacobian(jac_res)
    summarize_observable_ranking(jac_res, 15)

    selection = select_circuits_greedy_svd(circuits, jac_res, top_k=10)
    summarize_selection(selection)

def main() -> None:
    run_3q_svd_demo()

if __name__ == "__main__":
    main()


# In[ ]:




