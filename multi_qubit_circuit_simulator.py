#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
multi_qubit_circuit_simulator.py

Lightweight 3-4 qubit circuit simulator for MEADD-style calibration.

Run:

    python multi_qubit_circuit_simulator.py

Dependencies:

    multi_qubit_state_space.py
    lightweight_cptp_channel.py

Purpose
-------
This file implements the circuit layer of the PPT plan:

    prepare initial state
    repeat n times:
        noisy CZ-like CPTP channel
        Pauli / single-qubit interleaving
    measure selected observables
    return f1...fk

This module intentionally avoids dense superoperators. It evolves density
matrices by sequentially applying:

    rho -> U rho U^dagger
    rho -> sum_k K_k rho K_k^dagger

This is safe for the target PC for:

    3-4 qubits, local_dim=2
    3-4 qubits, local_dim=3 with one leakage level

Design notes
------------
1. This simulator is not yet the final fitting pipeline.
   It produces synthetic observable functions f1...fk that the next file,
   sensitivity_svd_selector.py, will differentiate and analyze with SVD.

2. Decoherence is handled through CPTP channels from lightweight_cptp_channel.py.
   The old observable-level decoherence envelope is not used.

3. Default depths and candidate sets are small so that a normal PC can run
   the examples quickly.

4. The code supports generic circuits, but includes several MEADD-inspired
   helper constructors:
       - phi_det style matrix-element measurements
       - theta/chi odd-subspace Bloch measurements
       - Z-string readout measurements
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

from multi_qubit_state_space import (
    Array,
    MultiQubitStateSpace,
    apply_unitary_to_density,
    bell_odd_state,
    embed_local_operator,
    embed_two_local_operator,
    expectation_value,
    ket_plus_on_qubit,
    local_h,
    local_identity,
    local_phase_rotation,
    local_x,
    local_xy_rotation,
    local_y,
    local_z,
    pauli_string_operator,
)

from lightweight_cptp_channel import (
    CPTPModelConfig,
    ReadoutNoise,
    TwoQubitCoherentParams,
    apply_all_local_noise,
    apply_pauli_interleaving,
    apply_single_qubit_rotation,
    apply_two_qubit_cz_like_cycle,
    default_cptp_config,
    leakage_probability,
    observed_pauli_z_expectation,
    trace_distance_from_one,
    zero_noise_config,
)


InitialStateKind = Literal[
    "basis",
    "plus_on_qubit",
    "bell_odd",
    "custom_state",
    "custom_density",
]

MeasurementKind = Literal[
    "pauli",
    "readout_z",
    "leakage",
    "odd_bloch_x",
    "odd_bloch_y",
    "odd_bloch_z",
    "matrix_element_real",
    "matrix_element_imag",
]


@dataclass(frozen=True)
class InitialStateSpec:
    """
    Initial state specification.

    kind="basis":
        occupation must be provided.

    kind="plus_on_qubit":
        target must be provided.
        base_occupation optionally fixes other qubits.

    kind="bell_odd":
        targets=(a,b) must be provided.
        phase controls (|01> + phase |10>)/sqrt(2).

    kind="custom_state":
        custom_state must be a state vector.

    kind="custom_density":
        custom_density must be a density matrix.
    """

    kind: InitialStateKind
    occupation: Optional[Tuple[int, ...]] = None
    target: Optional[int] = None
    targets: Optional[Tuple[int, int]] = None
    base_occupation: Optional[Tuple[int, ...]] = None
    phase: complex = 1.0
    custom_state: Optional[Array] = None
    custom_density: Optional[Array] = None


@dataclass(frozen=True)
class InterleavingStep:
    """
    One interleaving operation after each noisy CZ cycle.

    kind:
        "pauli"      -> apply Pauli string such as "XXII"
        "local"      -> apply explicit local_unitary on target
        "xy"         -> apply local XY rotation
        "z_phase"    -> apply local phase rotation diag(1, exp(i angle))
        "none"       -> no operation
    """

    kind: str
    pauli_string: Optional[str] = None
    target: Optional[int] = None
    local_unitary: Optional[Array] = None
    angle: float = 0.0
    phase: float = 0.0


@dataclass(frozen=True)
class MeasurementSpec:
    """
    Measurement specification.

    kind="pauli":
        pauli_string gives an exact quantum expectation value.

    kind="readout_z":
        z_mask gives a Z-string expectation after classical readout error.

    kind="leakage":
        total leakage probability.

    kind="odd_bloch_x/y/z":
        targets=(a,b) defines odd subspace {|01>, |10>} and measures
        X_odd, Y_odd, or Z_odd.

    kind="matrix_element_real/imag":
        bra_occupation and ket_occupation define Re/Im rho[ket, bra]
        through the density matrix element <bra|rho|ket>.
    """

    name: str
    kind: MeasurementKind
    pauli_string: Optional[str] = None
    z_mask: Optional[Tuple[int, ...]] = None
    targets: Optional[Tuple[int, int]] = None
    bra_occupation: Optional[Tuple[int, ...]] = None
    ket_occupation: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class CircuitSpec:
    """
    Full circuit specification.

    target_pair:
        Pair acted on by the repeated noisy CZ-like channel.

    repetitions:
        Number of repeated noisy CZ cycles.

    interleaving:
        Steps applied after each noisy CZ cycle.

    final_operations:
        Optional operations applied once before measurement.
    """

    name: str
    num_qubits: int
    local_dim: int
    target_pair: Tuple[int, int]
    repetitions: int
    initial_state: InitialStateSpec
    measurements: Tuple[MeasurementSpec, ...]
    interleaving: Tuple[InterleavingStep, ...] = ()
    final_operations: Tuple[InterleavingStep, ...] = ()
    leakage_level: Optional[int] = None

    def validate(self) -> None:
        """
        Validate circuit specification.
        """
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        if self.local_dim < 2:
            raise ValueError("local_dim must be at least 2")

        if self.repetitions < 0:
            raise ValueError("repetitions must be non-negative")

        a, b = self.target_pair
        if a == b:
            raise ValueError("target_pair entries must be distinct")

        if not 0 <= a < self.num_qubits:
            raise ValueError("first target out of range")

        if not 0 <= b < self.num_qubits:
            raise ValueError("second target out of range")

        if not self.measurements:
            raise ValueError("at least one measurement is required")


@dataclass(frozen=True)
class SimulationResult:
    """
    Output of one circuit simulation.
    """

    circuit_name: str
    values: Dict[str, float]
    final_trace_error: float
    final_leakage_probability: float
    final_density: Optional[Array] = None


def make_state_space(circuit: CircuitSpec) -> MultiQubitStateSpace:
    """
    Build state space from circuit spec.
    """
    leakage_level = circuit.leakage_level
    if leakage_level is None and circuit.local_dim >= 3:
        leakage_level = 2

    return MultiQubitStateSpace(
        num_qubits=circuit.num_qubits,
        local_dim=circuit.local_dim,
        leakage_level=leakage_level,
    )


def prepare_initial_density(
    state_space: MultiQubitStateSpace,
    initial_state: InitialStateSpec,
) -> Array:
    """
    Prepare initial density matrix.
    """
    if initial_state.kind == "basis":
        if initial_state.occupation is None:
            raise ValueError("basis initial state requires occupation")
        return state_space.basis_density(initial_state.occupation)

    if initial_state.kind == "plus_on_qubit":
        if initial_state.target is None:
            raise ValueError("plus_on_qubit initial state requires target")
        state = ket_plus_on_qubit(
            state_space=state_space,
            target=initial_state.target,
            base_occupation=initial_state.base_occupation,
        )
        return state_space.density_from_state(state)

    if initial_state.kind == "bell_odd":
        if initial_state.targets is None:
            raise ValueError("bell_odd initial state requires targets")
        state = bell_odd_state(
            state_space=state_space,
            qubit_a=initial_state.targets[0],
            qubit_b=initial_state.targets[1],
            phase=initial_state.phase,
            base_occupation=initial_state.base_occupation,
        )
        return state_space.density_from_state(state)

    if initial_state.kind == "custom_state":
        if initial_state.custom_state is None:
            raise ValueError("custom_state requires custom_state")
        return state_space.density_from_state(initial_state.custom_state)

    if initial_state.kind == "custom_density":
        if initial_state.custom_density is None:
            raise ValueError("custom_density requires custom_density")
        rho = np.asarray(initial_state.custom_density, dtype=complex)
        if rho.shape != (state_space.dim, state_space.dim):
            raise ValueError("custom_density has wrong shape")
        return rho

    raise ValueError(f"unsupported initial state kind: {initial_state.kind}")


def apply_interleaving_step(
    state_space: MultiQubitStateSpace,
    rho: Array,
    step: InterleavingStep,
) -> Array:
    """
    Apply one interleaving/final operation.
    """
    if step.kind == "none":
        return np.asarray(rho, dtype=complex)

    if step.kind == "pauli":
        if step.pauli_string is None:
            raise ValueError("pauli interleaving requires pauli_string")
        return apply_pauli_interleaving(
            state_space=state_space,
            rho=rho,
            pauli_string=step.pauli_string,
        )

    if step.kind == "local":
        if step.target is None:
            raise ValueError("local interleaving requires target")
        if step.local_unitary is None:
            raise ValueError("local interleaving requires local_unitary")
        return apply_single_qubit_rotation(
            state_space=state_space,
            rho=rho,
            target=step.target,
            local_unitary=step.local_unitary,
        )

    if step.kind == "xy":
        if step.target is None:
            raise ValueError("xy interleaving requires target")
        local_unitary = local_xy_rotation(
            angle=step.angle,
            phase=step.phase,
            local_dim=state_space.local_dim,
        )
        return apply_single_qubit_rotation(
            state_space=state_space,
            rho=rho,
            target=step.target,
            local_unitary=local_unitary,
        )

    if step.kind == "z_phase":
        if step.target is None:
            raise ValueError("z_phase interleaving requires target")
        local_unitary = local_phase_rotation(
            angle=step.angle,
            local_dim=state_space.local_dim,
        )
        return apply_single_qubit_rotation(
            state_space=state_space,
            rho=rho,
            target=step.target,
            local_unitary=local_unitary,
        )

    raise ValueError(f"unsupported interleaving kind: {step.kind}")


def odd_subspace_operator(
    state_space: MultiQubitStateSpace,
    targets: Tuple[int, int],
    component: Literal["x", "y", "z"],
) -> Array:
    """
    Build odd-subspace Bloch operator for target pair.

    Basis:
        |01>, |10>

    X_odd = |01><10| + |10><01|
    Y_odd = -i|01><10| + i|10><01|
    Z_odd = |01><01| - |10><10|

    Spectator qubits are summed over once. Target qubits are not included
    in the spectator enumeration, avoiding duplicate counting.
    """
    a, b = targets

    if a == b:
        raise ValueError("targets must be distinct")

    if not 0 <= a < state_space.num_qubits:
        raise ValueError("first target out of range")

    if not 0 <= b < state_space.num_qubits:
        raise ValueError("second target out of range")

    spectator_qubits = [
        qubit
        for qubit in range(state_space.num_qubits)
        if qubit not in targets
    ]

    op = np.zeros((state_space.dim, state_space.dim), dtype=complex)

    for spectator_values in np.ndindex(*(2 for _ in spectator_qubits)):
        occ_01 = [0] * state_space.num_qubits
        occ_10 = [0] * state_space.num_qubits

        for qubit, value in zip(spectator_qubits, spectator_values):
            occ_01[qubit] = value
            occ_10[qubit] = value

        occ_01[a] = 0
        occ_01[b] = 1

        occ_10[a] = 1
        occ_10[b] = 0

        idx_01 = state_space.occupation_to_index(occ_01)
        idx_10 = state_space.occupation_to_index(occ_10)

        if component == "x":
            op[idx_01, idx_10] += 1.0
            op[idx_10, idx_01] += 1.0
        elif component == "y":
            op[idx_01, idx_10] += -1.0j
            op[idx_10, idx_01] += 1.0j
        elif component == "z":
            op[idx_01, idx_01] += 1.0
            op[idx_10, idx_10] += -1.0
        else:
            raise ValueError("component must be x, y, or z")

    return op

def density_matrix_element(
    state_space: MultiQubitStateSpace,
    rho: Array,
    bra_occupation: Tuple[int, ...],
    ket_occupation: Tuple[int, ...],
) -> complex:
    """
    Return <bra|rho|ket>.
    """
    bra_index = state_space.occupation_to_index(bra_occupation)
    ket_index = state_space.occupation_to_index(ket_occupation)
    return complex(np.asarray(rho, dtype=complex)[bra_index, ket_index])


def evaluate_measurement(
    state_space: MultiQubitStateSpace,
    rho: Array,
    measurement: MeasurementSpec,
    readout_noise: ReadoutNoise,
) -> float:
    """
    Evaluate one measurement.
    """
    if measurement.kind == "pauli":
        if measurement.pauli_string is None:
            raise ValueError("pauli measurement requires pauli_string")
        observable = pauli_string_operator(
            state_space=state_space,
            pauli_string=measurement.pauli_string,
        )
        return expectation_value(rho, observable)

    if measurement.kind == "readout_z":
        if measurement.z_mask is None:
            raise ValueError("readout_z measurement requires z_mask")
        return observed_pauli_z_expectation(
            state_space=state_space,
            rho=rho,
            z_mask=measurement.z_mask,
            readout_noise=readout_noise,
        )

    if measurement.kind == "leakage":
        return leakage_probability(state_space, rho)

    if measurement.kind in ("odd_bloch_x", "odd_bloch_y", "odd_bloch_z"):
        if measurement.targets is None:
            raise ValueError("odd_bloch measurement requires targets")
        component = {
            "odd_bloch_x": "x",
            "odd_bloch_y": "y",
            "odd_bloch_z": "z",
        }[measurement.kind]
        observable = odd_subspace_operator(
            state_space=state_space,
            targets=measurement.targets,
            component=component,
        )
        return expectation_value(rho, observable)

    if measurement.kind in ("matrix_element_real", "matrix_element_imag"):
        if measurement.bra_occupation is None:
            raise ValueError("matrix element measurement requires bra_occupation")
        if measurement.ket_occupation is None:
            raise ValueError("matrix element measurement requires ket_occupation")

        value = density_matrix_element(
            state_space=state_space,
            rho=rho,
            bra_occupation=measurement.bra_occupation,
            ket_occupation=measurement.ket_occupation,
        )

        if measurement.kind == "matrix_element_real":
            return float(np.real(value))
        return float(np.imag(value))

    raise ValueError(f"unsupported measurement kind: {measurement.kind}")


def simulate_circuit(
    circuit: CircuitSpec,
    coherent_params: TwoQubitCoherentParams,
    config: CPTPModelConfig,
    keep_final_density: bool = False,
    apply_local_noise_before_cz: bool = False,
    apply_local_noise_after_cz: bool = True,
) -> SimulationResult:
    """
    Simulate one circuit and return measurement values.
    """
    circuit.validate()
    state_space = make_state_space(circuit)
    config.validate(local_dim=state_space.local_dim)

    rho = prepare_initial_density(
        state_space=state_space,
        initial_state=circuit.initial_state,
    )

    for _ in range(circuit.repetitions):
        rho = apply_two_qubit_cz_like_cycle(
            state_space=state_space,
            rho=rho,
            targets=circuit.target_pair,
            coherent_params=coherent_params,
            config=config,
            apply_local_noise_before=apply_local_noise_before_cz,
            apply_local_noise_after=apply_local_noise_after_cz,
        )

        for step in circuit.interleaving:
            rho = apply_interleaving_step(
                state_space=state_space,
                rho=rho,
                step=step,
            )

    for step in circuit.final_operations:
        rho = apply_interleaving_step(
            state_space=state_space,
            rho=rho,
            step=step,
        )

    values = {
        measurement.name: evaluate_measurement(
            state_space=state_space,
            rho=rho,
            measurement=measurement,
            readout_noise=config.readout_noise,
        )
        for measurement in circuit.measurements
    }

    return SimulationResult(
        circuit_name=circuit.name,
        values=values,
        final_trace_error=trace_distance_from_one(rho),
        final_leakage_probability=leakage_probability(state_space, rho),
        final_density=rho if keep_final_density else None,
    )


def simulate_circuit_batch(
    circuits: Sequence[CircuitSpec],
    coherent_params: TwoQubitCoherentParams,
    config: CPTPModelConfig,
    keep_final_density: bool = False,
) -> List[SimulationResult]:
    """
    Simulate a small batch of circuits.

    This function intentionally does not parallelize. The next SVD selector
    should call it in small batches to keep memory and CPU usage predictable.
    """
    return [
        simulate_circuit(
            circuit=circuit,
            coherent_params=coherent_params,
            config=config,
            keep_final_density=keep_final_density,
        )
        for circuit in circuits
    ]


def flatten_results(results: Sequence[SimulationResult]) -> Tuple[List[str], np.ndarray]:
    """
    Flatten batch results into labels and vector f.

    Labels have form:
        circuit_name/measurement_name
    """
    labels: List[str] = []
    values: List[float] = []

    for result in results:
        for measurement_name, value in result.values.items():
            labels.append(f"{result.circuit_name}/{measurement_name}")
            values.append(float(value))

    return labels, np.asarray(values, dtype=float)


def simulate_observable_vector(
    circuits: Sequence[CircuitSpec],
    coherent_params: TwoQubitCoherentParams,
    config: CPTPModelConfig,
) -> Tuple[List[str], np.ndarray]:
    """
    Simulate circuits and return f1...fk vector.
    """
    results = simulate_circuit_batch(
        circuits=circuits,
        coherent_params=coherent_params,
        config=config,
    )
    return flatten_results(results)


def add_binomial_shot_noise(
    values: Dict[str, float],
    shots: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Add approximate shot noise to expectation values in [-1,1].

    For an expectation value m = P(+1)-P(-1), sample k ~ Binomial(shots,p)
    with p=(1+m)/2 and return 2k/shots - 1.

    This is appropriate for Pauli-like binary observables. Leakage
    probabilities are clipped and sampled as probabilities.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")

    noisy: Dict[str, float] = {}

    for name, value in values.items():
        clipped = float(np.clip(value, -1.0, 1.0))
        probability = 0.5 * (1.0 + clipped)
        counts = rng.binomial(shots, probability)
        noisy[name] = float(2.0 * counts / shots - 1.0)

    return noisy


def with_repetitions(circuit: CircuitSpec, repetitions: int) -> CircuitSpec:
    """
    Return copy of circuit with different repetitions.
    """
    return replace(circuit, repetitions=repetitions)


def make_phi_det_probe_circuits(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    depths: Sequence[int] = (2, 4, 8, 12),
    dd_pauli: str = "XX",
) -> List[CircuitSpec]:
    """
    Construct lightweight phi-sensitive probe circuits.

    These are not the final determinant reconstruction protocol, but they
    provide matrix-element-style observables that are useful for the next
    SVD selector.

    For target pair (a,b), use initial |+0...> on qubit a with qubit b fixed
    to 0 or 1 and measure X/Y on qubit a. This probes phase accumulation
    relative to a spectator branch.

    For true MEADD phi extraction, the later fitter should reconstruct the
    odd-subspace matrix elements and take determinant phase slopes.
    """
    circuits: List[CircuitSpec] = []

    pauli = ["I"] * num_qubits
    pauli[target_pair[0]] = dd_pauli[0]
    pauli[target_pair[1]] = dd_pauli[1]
    pauli_string = "".join(pauli)

    for depth in depths:
        for fixed_b in (0, 1):
            base = [0] * num_qubits
            base[target_pair[1]] = fixed_b

            measurements = (
                MeasurementSpec(
                    name=f"X_q{target_pair[0]}",
                    kind="pauli",
                    pauli_string="".join(
                        "X" if q == target_pair[0] else "I"
                        for q in range(num_qubits)
                    ),
                ),
                MeasurementSpec(
                    name=f"Y_q{target_pair[0]}",
                    kind="pauli",
                    pauli_string="".join(
                        "Y" if q == target_pair[0] else "I"
                        for q in range(num_qubits)
                    ),
                ),
                MeasurementSpec(
                    name="leakage",
                    kind="leakage",
                ),
            )

            circuits.append(
                CircuitSpec(
                    name=f"phi_probe_b{fixed_b}_n{depth}",
                    num_qubits=num_qubits,
                    local_dim=local_dim,
                    target_pair=target_pair,
                    repetitions=depth,
                    initial_state=InitialStateSpec(
                        kind="plus_on_qubit",
                        target=target_pair[0],
                        base_occupation=tuple(base),
                    ),
                    interleaving=(
                        InterleavingStep(
                            kind="pauli",
                            pauli_string=pauli_string,
                        ),
                    ),
                    measurements=measurements,
                )
            )

    return circuits


def make_theta_chi_probe_circuits(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    depths: Sequence[int] = (2, 4, 8, 12),
) -> List[CircuitSpec]:
    """
    Construct swap_x/swap_y-sensitive odd-subspace circuits.

For small residual swap:
    swap_x = theta cos(chi)
    swap_y = theta sin(chi)

The XX and YX DD choices are intended to select these two quadratures.
The function name is retained for backward compatibility.

    Inspired by MEADD:
        prepare |01> or |10>
        interleave XX or YX
        measure odd-subspace Bloch X/Y/Z

    The paper notes theta and chi can be separated by choosing relative DD
    phases that select real and imaginary parts of the swap matrix element.
    """
    circuits: List[CircuitSpec] = []

    dd_options = {
        "XX": ("X", "X"),
        "YX": ("Y", "X"),
    }

    for depth in depths:
        for init_label, init_pair_values in {
            "01": (0, 1),
            "10": (1, 0),
        }.items():
            for dd_label, dd_pair in dd_options.items():
                occupation = [0] * num_qubits
                occupation[target_pair[0]] = init_pair_values[0]
                occupation[target_pair[1]] = init_pair_values[1]

                pauli = ["I"] * num_qubits
                pauli[target_pair[0]] = dd_pair[0]
                pauli[target_pair[1]] = dd_pair[1]
                pauli_string = "".join(pauli)

                circuits.append(
                    CircuitSpec(
                        name=f"theta_chi_{dd_label}_init{init_label}_n{depth}",
                        num_qubits=num_qubits,
                        local_dim=local_dim,
                        target_pair=target_pair,
                        repetitions=depth,
                        initial_state=InitialStateSpec(
                            kind="basis",
                            occupation=tuple(occupation),
                        ),
                        interleaving=(
                            InterleavingStep(
                                kind="pauli",
                                pauli_string=pauli_string,
                            ),
                        ),
                        measurements=(
                            MeasurementSpec(
                                name="X_odd",
                                kind="odd_bloch_x",
                                targets=target_pair,
                            ),
                            MeasurementSpec(
                                name="Y_odd",
                                kind="odd_bloch_y",
                                targets=target_pair,
                            ),
                            MeasurementSpec(
                                name="Z_odd",
                                kind="odd_bloch_z",
                                targets=target_pair,
                            ),
                            MeasurementSpec(
                                name="leakage",
                                kind="leakage",
                            ),
                        ),
                    )
                )

    return circuits

def make_swap_xy_probe_circuits(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    depths: Sequence[int] = (2, 4, 8, 12),
) -> List[CircuitSpec]:
    """
    Construct swap_x/swap_y-sensitive odd-subspace circuits.

    XX DD probes the theta cos(chi) direction.
    YX DD probes the theta sin(chi) direction.
    """
    return make_theta_chi_probe_circuits(
        num_qubits=num_qubits,
        local_dim=local_dim,
        target_pair=target_pair,
        depths=depths,
    )


def make_readout_z_probe_circuits(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    depths: Sequence[int] = (1, 2, 4, 8),
) -> List[CircuitSpec]:
    """
    Construct simple computational-basis/readout Z probes.

    These are useful for detecting population transfer, leakage, and readout
    sensitivity. They are not sufficient alone for coherent phase extraction.
    """
    circuits: List[CircuitSpec] = []

    z_masks = []
    for q in range(num_qubits):
        mask = [0] * num_qubits
        mask[q] = 1
        z_masks.append(tuple(mask))

    pair_mask = [0] * num_qubits
    pair_mask[target_pair[0]] = 1
    pair_mask[target_pair[1]] = 1
    z_masks.append(tuple(pair_mask))

    initial_occupations = []
    for value_a, value_b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        occ = [0] * num_qubits
        occ[target_pair[0]] = value_a
        occ[target_pair[1]] = value_b
        initial_occupations.append(tuple(occ))

    for depth in depths:
        for occupation in initial_occupations:
            measurements = tuple(
                MeasurementSpec(
                    name="Z_" + "".join(str(bit) for bit in mask),
                    kind="readout_z",
                    z_mask=mask,
                )
                for mask in z_masks
            ) + (
                MeasurementSpec(
                    name="leakage",
                    kind="leakage",
                ),
            )

            circuits.append(
                CircuitSpec(
                    name=f"readout_z_init{''.join(map(str, occupation))}_n{depth}",
                    num_qubits=num_qubits,
                    local_dim=local_dim,
                    target_pair=target_pair,
                    repetitions=depth,
                    initial_state=InitialStateSpec(
                        kind="basis",
                        occupation=occupation,
                    ),
                    measurements=measurements,
                )
            )

    return circuits


def make_default_probe_circuits(
    num_qubits: int = 3,
    local_dim: int = 3,
    target_pair: Tuple[int, int] = (0, 1),
    max_depth: int = 12,
) -> List[CircuitSpec]:
    """
    Build a compact default set of probe circuits.

    Depths are intentionally modest for a normal PC.
    """
    if max_depth <= 4:
        depths = (1, 2, 4)
    elif max_depth <= 8:
        depths = (1, 2, 4, 8)
    else:
        depths = (2, 4, 8, min(12, max_depth))

    circuits: List[CircuitSpec] = []
    circuits.extend(
        make_phi_det_probe_circuits(
            num_qubits=num_qubits,
            local_dim=local_dim,
            target_pair=target_pair,
            depths=tuple(depth for depth in depths if depth % 2 == 0),
        )
    )
    circuits.extend(
        make_theta_chi_probe_circuits(
            num_qubits=num_qubits,
            local_dim=local_dim,
            target_pair=target_pair,
            depths=tuple(depth for depth in depths if depth % 2 == 0),
        )
    )
    circuits.extend(
        make_readout_z_probe_circuits(
            num_qubits=num_qubits,
            local_dim=local_dim,
            target_pair=target_pair,
            depths=depths,
        )
    )
    return circuits


def summarize_results(results: Sequence[SimulationResult], max_lines: int = 20) -> None:
    """
    Print compact simulation results.
    """
    print("\nSimulation results")
    print("-" * 88)
    printed = 0

    for result in results:
        if printed >= max_lines:
            remaining = sum(len(r.values) for r in results) - printed
            print(f"... omitted {remaining} additional values")
            break

        for measurement_name, value in result.values.items():
            if printed >= max_lines:
                break
            print(
                f"{result.circuit_name:42s} "
                f"{measurement_name:14s} "
                f"{value: .8f} "
                f"trace_err={result.final_trace_error:.2e} "
                f"leak={result.final_leakage_probability:.2e}"
            )
            printed += 1


def run_single_circuit_demo() -> None:
    """
    Demonstrate one MEADD-like theta/chi circuit.
    """
    print("\nSingle circuit demo")
    print("=" * 88)

    circuit = make_theta_chi_probe_circuits(
        num_qubits=3,
        local_dim=3,
        target_pair=(0, 1),
        depths=(8,),
    )[0]

    coherent = TwoQubitCoherentParams(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
    )
    config = default_cptp_config(local_dim=3)

    result = simulate_circuit(
        circuit=circuit,
        coherent_params=coherent,
        config=config,
        keep_final_density=False,
    )

    summarize_results([result])


def run_batch_demo() -> None:
    """
    Demonstrate a compact 3-qutrit batch simulation.
    """
    print("\nBatch demo: 3 qutrits")
    print("=" * 88)

    circuits = make_default_probe_circuits(
        num_qubits=3,
        local_dim=3,
        target_pair=(0, 1),
        max_depth=8,
    )

    coherent = TwoQubitCoherentParams(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
    )
    config = default_cptp_config(local_dim=3)

    results = simulate_circuit_batch(
        circuits=circuits[:12],
        coherent_params=coherent,
        config=config,
    )

    summarize_results(results, max_lines=24)

    labels, vector = flatten_results(results)
    print(f"\nflattened observable count = {len(vector)}")
    print(f"first label/value = {labels[0]} -> {vector[0]: .8f}")


def run_4q_leakage_smoke_test() -> None:
    """
    Smoke test for 4 qubits with leakage.

    Keeps circuit count and depth small to stay PC-friendly.
    """
    print("\nSmoke test: 4 qutrits with leakage")
    print("=" * 88)

    circuits = make_default_probe_circuits(
        num_qubits=4,
        local_dim=3,
        target_pair=(1, 2),
        max_depth=4,
    )

    coherent = TwoQubitCoherentParams(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
    )
    config = default_cptp_config(local_dim=3)

    results = simulate_circuit_batch(
        circuits=circuits[:8],
        coherent_params=coherent,
        config=config,
    )

    max_trace_error = max(result.final_trace_error for result in results)
    max_leakage = max(result.final_leakage_probability for result in results)

    print(f"circuits simulated = {len(results)}")
    print(f"max trace error    = {max_trace_error:.6e}")
    print(f"max leakage prob   = {max_leakage:.6e}")

    if max_trace_error > 1e-10:
        raise AssertionError("4-qutrit smoke test failed trace preservation")


def run_zero_noise_consistency_test() -> None:
    """
    Check that zero-noise 3-qubit simulation remains physical.
    """
    print("\nZero-noise consistency test")
    print("=" * 88)

    circuit = CircuitSpec(
        name="zero_noise_test",
        num_qubits=3,
        local_dim=2,
        target_pair=(0, 1),
        repetitions=4,
        initial_state=InitialStateSpec(
            kind="basis",
            occupation=(0, 1, 0),
        ),
        interleaving=(
            InterleavingStep(
                kind="pauli",
                pauli_string="XXI",
            ),
        ),
        measurements=(
            MeasurementSpec(
                name="Z0",
                kind="pauli",
                pauli_string="ZII",
            ),
            MeasurementSpec(
                name="Z1",
                kind="pauli",
                pauli_string="IZI",
            ),
            MeasurementSpec(
                name="X_odd",
                kind="odd_bloch_x",
                targets=(0, 1),
            ),
            MeasurementSpec(
                name="Y_odd",
                kind="odd_bloch_y",
                targets=(0, 1),
            ),
            MeasurementSpec(
                name="Z_odd",
                kind="odd_bloch_z",
                targets=(0, 1),
            ),
        ),
    )

    coherent = TwoQubitCoherentParams(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
    )
    config = zero_noise_config(local_dim=2)

    result = simulate_circuit(
        circuit=circuit,
        coherent_params=coherent,
        config=config,
    )

    summarize_results([result], max_lines=10)

    if result.final_trace_error > 1e-12:
        raise AssertionError("zero-noise simulation failed trace preservation")


def main() -> None:
    """
    CLI entry point.
    """
    run_single_circuit_demo()
    run_batch_demo()
    run_4q_leakage_smoke_test()
    run_zero_noise_consistency_test()


if __name__ == "__main__":
    main()


# In[ ]:




