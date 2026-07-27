#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
lightweight_cptp_channel.py

Lightweight CPTP channel utilities for 3-4 qubit MEADD-style calibration
simulation.

Run:

    python lightweight_cptp_channel.py

Purpose
-------
This file implements density-matrix-level CPTP noise models for the next
stage of the PPT plan:

    1. 3-4 qubit simulation
    2. computational basis + leakage level
    3. CPTP noise channel supplied/parameterized by experiment team
    4. Pauli/single-qubit interleaving
    5. finite-difference Jacobian and SVD circuit selection

Important design choice
-----------------------
This module does NOT construct dense Liouville superoperators. Channels are
applied as:

    rho -> U rho U^dagger
    rho -> sum_k K_k rho K_k^dagger

This avoids the memory blowup of d^2 x d^2 superoperators, especially for
4 qubits with leakage:

    local_dim = 3
    num_qubits = 4
    Hilbert dim = 81
    dense rho/operator = 81 x 81
    dense superoperator = 6561 x 6561

The previous observable-level decoherence envelope is not used here. T1,
dephasing, depolarizing, leakage, and seepage are applied directly to the
density matrix inside the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from cptp_noise_models import (
    CZParams,
    cz_leakage_kraus,
    cz_unitary_qutrit,
    depolarizing_kraus,
)
import numpy as np

from multi_qubit_state_space import (
    Array,
    MultiQubitStateSpace,
    apply_unitary_to_density,
    computational_cz,
    computational_swap_like_error,
    embed_local_operator,
    embed_two_local_operator,
    expectation_value,
    local_identity,
    local_level_projector,
    local_transition,
    local_x,
    local_y,
    local_z,
    pauli_string_operator,
    two_qubit_cz_like_unitary,
)


@dataclass(frozen=True)
class LocalNoiseRates:
    """
    Local per-cycle CPTP noise rates.

    Parameters are probabilities per application of the local channel.

    amplitude_damping:
        T1-like relaxation |1> -> |0|.

    pure_dephasing:
        Z-type phase damping on the |0>, |1> coherence.

    depolarizing:
        Single-qubit depolarizing probability within computational subspace.

    leakage:
        Population transfer |1> -> |2>. Requires local_dim >= 3.

    seepage:
        Population transfer |2> -> |1>. Requires local_dim >= 3.
    """

    amplitude_damping: float = 0.0
    pure_dephasing: float = 0.0
    depolarizing: float = 0.0
    leakage: float = 0.0
    seepage: float = 0.0

    def validate(self, local_dim: int) -> None:
        """
        Validate rates.
        """
        values = {
            "amplitude_damping": self.amplitude_damping,
            "pure_dephasing": self.pure_dephasing,
            "depolarizing": self.depolarizing,
            "leakage": self.leakage,
            "seepage": self.seepage,
        }

        for name, value in values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if local_dim < 3 and (self.leakage > 0.0 or self.seepage > 0.0):
            raise ValueError("leakage/seepage require local_dim >= 3")


@dataclass(frozen=True)
class TwoQubitCoherentParams:
    """
    Lightweight coherent CZ-like parameters.

    theta and chi are retained for compatibility. New fitting code should
    prefer swap_x/swap_y, where:

        swap_x = theta cos(chi)
        swap_y = theta sin(chi)
    """

    dphi: float = 0.0
    theta: float = 0.0
    chi: float = 0.0
    swap_x: Optional[float] = None
    swap_y: Optional[float] = None

    def effective_theta_chi(self) -> Tuple[float, float]:
        if self.swap_x is None or self.swap_y is None:
            return float(self.theta), float(self.chi)

        swap_x = float(self.swap_x)
        swap_y = float(self.swap_y)
        return float(np.hypot(swap_x, swap_y)), float(np.arctan2(swap_y, swap_x))

    def to_cz_params(self) -> CZParams:
        if self.swap_x is None or self.swap_y is None:
            theta = float(self.theta)
            chi = float(self.chi)
            swap_x = theta * float(np.cos(chi))
            swap_y = theta * float(np.sin(chi))
        else:
            swap_x = float(self.swap_x)
            swap_y = float(self.swap_y)

        return CZParams(
            dphi=float(self.dphi),
            swap_x=swap_x,
            swap_y=swap_y,
        )


@dataclass(frozen=True)
class TwoQubitNoiseRates:
    """
    Two-qubit noise rates applied after the coherent two-qubit operation.

    two_qubit_depolarizing:
        Depolarizing probability inside the two-qubit computational subspace.

    zz_phase_jitter_std:
        Optional quasi-static Gaussian jitter in residual ZZ phase. This is
        sampled outside this class by the circuit simulator if needed.
    """

    two_qubit_depolarizing: float = 0.0
    zz_phase_jitter_std: float = 0.0

    def validate(self) -> None:
        """
        Validate rates.
        """
        if not 0.0 <= self.two_qubit_depolarizing <= 1.0:
            raise ValueError("two_qubit_depolarizing must be between 0 and 1")

        if self.zz_phase_jitter_std < 0.0:
            raise ValueError("zz_phase_jitter_std must be non-negative")


@dataclass(frozen=True)
class ReadoutNoise:
    """
    Classical readout assignment error.

    This is not a quantum channel applied before measurement. It is used to
    transform ideal computational-basis probabilities into observed
    probabilities or noisy expectation values.

    assignment_error_0_to_1:
        Probability of reporting 1 when the actual computational state is 0.

    assignment_error_1_to_0:
        Probability of reporting 0 when the actual computational state is 1.

    leakage_report_as_1:
        Probability of reporting leaked level |2> as bit 1. The remaining
        probability reports it as bit 0.
    """

    assignment_error_0_to_1: float = 0.0
    assignment_error_1_to_0: float = 0.0
    leakage_report_as_1: float = 0.5

    def validate(self) -> None:
        """
        Validate readout rates.
        """
        values = {
            "assignment_error_0_to_1": self.assignment_error_0_to_1,
            "assignment_error_1_to_0": self.assignment_error_1_to_0,
            "leakage_report_as_1": self.leakage_report_as_1,
        }

        for name, value in values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class CPTPModelConfig:
    """
    Full lightweight CPTP model configuration.
    """

    local_noise: LocalNoiseRates
    two_qubit_noise: TwoQubitNoiseRates
    readout_noise: ReadoutNoise

    def validate(self, local_dim: int) -> None:
        """
        Validate the full model.
        """
        self.local_noise.validate(local_dim=local_dim)
        self.two_qubit_noise.validate()
        self.readout_noise.validate()


def default_cptp_config(local_dim: int = 3) -> CPTPModelConfig:
    """
    Conservative default CPTP parameters for synthetic validation.

    These are not experimental values. They are small placeholder rates for
    debugging the simulator and SVD selector.
    """
    local_noise = LocalNoiseRates(
        amplitude_damping=2.0e-4,
        pure_dephasing=3.0e-4,
        depolarizing=1.0e-4,
        leakage=5.0e-5 if local_dim >= 3 else 0.0,
        seepage=1.0e-4 if local_dim >= 3 else 0.0,
    )

    two_qubit_noise = TwoQubitNoiseRates(
        two_qubit_depolarizing=5.0e-4,
        zz_phase_jitter_std=0.0,
    )

    readout_noise = ReadoutNoise(
        assignment_error_0_to_1=0.01,
        assignment_error_1_to_0=0.015,
        leakage_report_as_1=0.5,
    )

    config = CPTPModelConfig(
        local_noise=local_noise,
        two_qubit_noise=two_qubit_noise,
        readout_noise=readout_noise,
    )
    config.validate(local_dim=local_dim)

    return config


def zero_noise_config(local_dim: int = 3) -> CPTPModelConfig:
    """
    Zero-noise configuration.
    """
    config = CPTPModelConfig(
        local_noise=LocalNoiseRates(),
        two_qubit_noise=TwoQubitNoiseRates(),
        readout_noise=ReadoutNoise(),
    )
    config.validate(local_dim=local_dim)
    return config


def apply_kraus_channel(rho: Array, kraus_ops: Sequence[Array]) -> Array:
    """
    Apply Kraus channel:

        rho -> sum_k K_k rho K_k^dagger
    """
    rho = np.asarray(rho, dtype=complex)
    output = np.zeros_like(rho, dtype=complex)

    for kraus in kraus_ops:
        kraus = np.asarray(kraus, dtype=complex)
        output += kraus @ rho @ np.conjugate(kraus.T)

    return output


def kraus_trace_check(kraus_ops: Sequence[Array], atol: float = 1e-10) -> float:
    """
    Return ||sum K^dagger K - I||_F for a Kraus set.
    """
    if not kraus_ops:
        raise ValueError("kraus_ops must be non-empty")

    dim = kraus_ops[0].shape[0]
    accum = np.zeros((dim, dim), dtype=complex)

    for kraus in kraus_ops:
        accum += np.conjugate(kraus.T) @ kraus

    return float(np.linalg.norm(accum - np.eye(dim, dtype=complex)))


def local_amplitude_damping_kraus(local_dim: int, probability: float) -> List[Array]:
    """
    Local T1-like amplitude damping |1> -> |0>.

    Leakage levels are left unchanged.
    """
    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    k0 = np.eye(local_dim, dtype=complex)
    k0[1, 1] = np.sqrt(1.0 - p)

    k1 = np.zeros((local_dim, local_dim), dtype=complex)
    k1[0, 1] = np.sqrt(p)

    return [k0, k1]


def local_pure_dephasing_kraus(local_dim: int, probability: float) -> List[Array]:
    """
    Local phase damping on computational |0>, |1> coherence.

    This implements the channel:

        rho -> (1-p) rho + p Z rho Z

    on the computational subspace. Leakage levels are left unchanged by both
    Kraus operators.
    """
    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    z = local_z(local_dim)
    k0 = np.sqrt(1.0 - p) * np.eye(local_dim, dtype=complex)
    k1 = np.sqrt(p) * z

    return [k0, k1]


def local_depolarizing_kraus(local_dim: int, probability: float) -> List[Array]:
    """
    Local depolarizing channel on computational |0>, |1> subspace.

    Uses:
        rho -> (1-p) rho + p/3 * (X rho X + Y rho Y + Z rho Z)

    For local_dim=3, X/Y/Z leave leakage unchanged. This is a simple
    embedded model, not a full qutrit depolarizing channel.
    """
    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    return [
        np.sqrt(1.0 - p) * np.eye(local_dim, dtype=complex),
        np.sqrt(p / 3.0) * local_x(local_dim),
        np.sqrt(p / 3.0) * local_y(local_dim),
        np.sqrt(p / 3.0) * local_z(local_dim),
    ]


def local_leakage_kraus(local_dim: int, probability: float) -> List[Array]:
    """
    Local leakage channel |1> -> |2>.

    Requires local_dim >= 3.
    """
    if local_dim < 3:
        raise ValueError("local leakage requires local_dim >= 3")

    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    k0 = np.eye(local_dim, dtype=complex)
    k0[1, 1] = np.sqrt(1.0 - p)

    k1 = np.zeros((local_dim, local_dim), dtype=complex)
    k1[2, 1] = np.sqrt(p)

    return [k0, k1]


def local_seepage_kraus(local_dim: int, probability: float) -> List[Array]:
    """
    Local seepage channel |2> -> |1>.

    Requires local_dim >= 3.
    """
    if local_dim < 3:
        raise ValueError("local seepage requires local_dim >= 3")

    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    k0 = np.eye(local_dim, dtype=complex)
    k0[2, 2] = np.sqrt(1.0 - p)

    k1 = np.zeros((local_dim, local_dim), dtype=complex)
    k1[1, 2] = np.sqrt(p)

    return [k0, k1]


def embed_local_kraus(
    state_space: MultiQubitStateSpace,
    local_kraus_ops: Sequence[Array],
    target: int,
) -> List[Array]:
    """
    Embed local Kraus operators into full Hilbert space.
    """
    return [
        embed_local_operator(
            state_space=state_space,
            local_operator=kraus,
            target=target,
        )
        for kraus in local_kraus_ops
    ]


def apply_local_channel(
    state_space: MultiQubitStateSpace,
    rho: Array,
    local_kraus_ops: Sequence[Array],
    target: int,
) -> Array:
    """
    Apply a local Kraus channel to one qubit/qudit.
    """
    full_kraus = embed_local_kraus(
        state_space=state_space,
        local_kraus_ops=local_kraus_ops,
        target=target,
    )

    return apply_kraus_channel(rho, full_kraus)


def apply_local_noise_to_qubit(
    state_space: MultiQubitStateSpace,
    rho: Array,
    target: int,
    rates: LocalNoiseRates,
) -> Array:
    """
    Apply local noise stack to one qubit/qudit.

    Order:
        amplitude damping
        pure dephasing
        local depolarizing
        leakage
        seepage
    """
    rates.validate(local_dim=state_space.local_dim)
    output = np.asarray(rho, dtype=complex)

    if rates.amplitude_damping > 0.0:
        output = apply_local_channel(
            state_space=state_space,
            rho=output,
            local_kraus_ops=local_amplitude_damping_kraus(
                local_dim=state_space.local_dim,
                probability=rates.amplitude_damping,
            ),
            target=target,
        )

    if rates.pure_dephasing > 0.0:
        output = apply_local_channel(
            state_space=state_space,
            rho=output,
            local_kraus_ops=local_pure_dephasing_kraus(
                local_dim=state_space.local_dim,
                probability=rates.pure_dephasing,
            ),
            target=target,
        )

    if rates.depolarizing > 0.0:
        output = apply_local_channel(
            state_space=state_space,
            rho=output,
            local_kraus_ops=local_depolarizing_kraus(
                local_dim=state_space.local_dim,
                probability=rates.depolarizing,
            ),
            target=target,
        )

    if rates.leakage > 0.0:
        output = apply_local_channel(
            state_space=state_space,
            rho=output,
            local_kraus_ops=local_leakage_kraus(
                local_dim=state_space.local_dim,
                probability=rates.leakage,
            ),
            target=target,
        )

    if rates.seepage > 0.0:
        output = apply_local_channel(
            state_space=state_space,
            rho=output,
            local_kraus_ops=local_seepage_kraus(
                local_dim=state_space.local_dim,
                probability=rates.seepage,
            ),
            target=target,
        )

    return output


def apply_all_local_noise(
    state_space: MultiQubitStateSpace,
    rho: Array,
    rates: LocalNoiseRates,
) -> Array:
    """
    Apply local noise to all qubits/qudits.
    """
    output = np.asarray(rho, dtype=complex)

    for target in range(state_space.num_qubits):
        output = apply_local_noise_to_qubit(
            state_space=state_space,
            rho=output,
            target=target,
            rates=rates,
        )

    return output


def two_qubit_depolarizing_kraus(
    local_dim: int,
    probability: float,
) -> List[Array]:
    """
    Two-qubit depolarizing channel in the embedded computational subspace.

    Uses 15 non-identity two-qubit Pauli products on |0>, |1>. For local_dim=3,
    local Pauli operators leave leakage unchanged.

    This is a lightweight embedded model, not a full qutrit-qutrit
    depolarizing channel.
    """
    p = float(probability)

    if not 0.0 <= p <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    identity = local_identity(local_dim)
    paulis = [
        local_x(local_dim),
        local_y(local_dim),
        local_z(local_dim),
    ]

    kraus = [np.sqrt(1.0 - p) * np.kron(identity, identity)]

    weight = np.sqrt(p / 15.0)

    local_basis = [identity] + paulis

    for left_idx, left in enumerate(local_basis):
        for right_idx, right in enumerate(local_basis):
            if left_idx == 0 and right_idx == 0:
                continue
            kraus.append(weight * np.kron(left, right))

    return kraus


def apply_two_qubit_channel(
    state_space: MultiQubitStateSpace,
    rho: Array,
    two_qubit_kraus_ops: Sequence[Array],
    targets: Tuple[int, int],
) -> Array:
    """
    Apply two-qubit Kraus channel to target pair.
    """
    full_kraus = [
        embed_two_local_operator(
            state_space=state_space,
            two_qubit_operator=kraus,
            targets=targets,
        )
        for kraus in two_qubit_kraus_ops
    ]

    return apply_kraus_channel(rho, full_kraus)


def apply_two_qubit_noise(
    state_space: MultiQubitStateSpace,
    rho: Array,
    targets: Tuple[int, int],
    rates: TwoQubitNoiseRates,
) -> Array:
    """
    Apply two-qubit noise stack to one target pair.
    """
    rates.validate()
    output = np.asarray(rho, dtype=complex)

    if rates.two_qubit_depolarizing > 0.0:
        output = apply_two_qubit_channel(
            state_space=state_space,
            rho=output,
            two_qubit_kraus_ops=two_qubit_depolarizing_kraus(
                local_dim=state_space.local_dim,
                probability=rates.two_qubit_depolarizing,
            ),
            targets=targets,
        )

    return output


def apply_two_qubit_cz_like_cycle(
    state_space: MultiQubitStateSpace,
    rho: Array,
    targets: Tuple[int, int],
    coherent_params: TwoQubitCoherentParams,
    config: CPTPModelConfig,
    apply_local_noise_before: bool = False,
    apply_local_noise_after: bool = True,
) -> Array:
    """
    Apply one noisy CZ-like cycle.

    Cycle structure:
        optional local noise
        coherent CZ-like unitary (or unified Kraus model for Qutrits)
        two-qubit noise
        optional local noise

    This is the first CPTP replacement for the old observable-level
    decoherence envelope.
    """
    config.validate(local_dim=state_space.local_dim)

    output = np.asarray(rho, dtype=complex)

    # 1. 应用门前局部噪声
    if apply_local_noise_before:
        output = apply_all_local_noise(
            state_space=state_space,
            rho=output,
            rates=config.local_noise,
        )

    # 2. 应用二比特门操作及相关门噪声
    if state_space.local_dim == 3:
        # 使用统一的新版 qutrit CZ 泄漏 + 门极去极化 Kraus 组合模型
        cz_params = coherent_params.to_cz_params()
        unitary = cz_unitary_qutrit(cz_params)
        
        # 获取由当前门驱动带来的 qutrit 泄漏 Kraus 算子
        leakage_kraus = cz_leakage_kraus(leakage=float(config.local_noise.leakage))
        
        # 获取 9 维双比特去极化 Kraus 算子 (去极化率来自 two_qubit_depolarizing)
        depol_kraus = depolarizing_kraus(
            prob=float(config.two_qubit_noise.two_qubit_depolarizing),
            dim=9,
        )

        # 联合链式 Kraus：K_final = K_depol @ K_leakage @ U_coherent
        combined_kraus = [
            kd @ kl @ unitary
            for kl in leakage_kraus
            for kd in depol_kraus
        ]

        output = apply_two_qubit_channel(
            state_space=state_space,
            rho=output,
            two_qubit_kraus_ops=combined_kraus,
            targets=targets,
        )
    else:
        # local_dim == 2 时，完全退回旧版计算子空间仿真逻辑
        theta, chi = coherent_params.effective_theta_chi()

        local_unitary = two_qubit_cz_like_unitary(
            local_dim=state_space.local_dim,
            dphi=coherent_params.dphi,
            theta=theta,
            chi=chi,
        )

        full_unitary = embed_two_local_operator(
            state_space=state_space,
            two_qubit_operator=local_unitary,
            targets=targets,
        )

        output = apply_unitary_to_density(full_unitary, output)

        output = apply_two_qubit_noise(
            state_space=state_space,
            rho=output,
            targets=targets,
            rates=config.two_qubit_noise,
        )

    # 3. 应用门后局部噪声
    if apply_local_noise_after:
        output = apply_all_local_noise(
            state_space=state_space,
            rho=output,
            rates=config.local_noise,
        )

    return output


def apply_pauli_interleaving(
    state_space: MultiQubitStateSpace,
    rho: Array,
    pauli_string: str,
) -> Array:
    """
    Apply a Pauli-string interleaving unitary.
    """
    unitary = pauli_string_operator(
        state_space=state_space,
        pauli_string=pauli_string,
    )
    return apply_unitary_to_density(unitary, rho)


def apply_single_qubit_rotation(
    state_space: MultiQubitStateSpace,
    rho: Array,
    target: int,
    local_unitary: Array,
) -> Array:
    """
    Apply an embedded single-qubit/qudit unitary.
    """
    unitary = embed_local_operator(
        state_space=state_space,
        local_operator=local_unitary,
        target=target,
    )
    return apply_unitary_to_density(unitary, rho)


def trace_distance_from_one(rho: Array) -> float:
    """
    Return |Tr(rho)-1|.
    """
    return float(abs(np.trace(np.asarray(rho, dtype=complex)) - 1.0))


def hermiticity_error(rho: Array) -> float:
    """
    Return ||rho-rho^dagger||_F.
    """
    rho = np.asarray(rho, dtype=complex)
    return float(np.linalg.norm(rho - np.conjugate(rho.T)))


def min_eigenvalue_hermitian(rho: Array) -> float:
    """
    Return minimum eigenvalue after Hermitian symmetrization.
    """
    rho = np.asarray(rho, dtype=complex)
    hermitian = 0.5 * (rho + np.conjugate(rho.T))
    eigvals = np.linalg.eigvalsh(hermitian)
    return float(np.min(np.real(eigvals)))


def computational_basis_probabilities(
    state_space: MultiQubitStateSpace,
    rho: Array,
) -> Dict[Tuple[int, ...], float]:
    """
    Return probabilities for computational occupations only.
    """
    rho = np.asarray(rho, dtype=complex)
    probabilities: Dict[Tuple[int, ...], float] = {}

    for occupation in state_space.computational_occupations():
        index = state_space.occupation_to_index(occupation)
        probabilities[tuple(occupation)] = float(np.real_if_close(rho[index, index]))

    return probabilities


def all_basis_probabilities(
    state_space: MultiQubitStateSpace,
    rho: Array,
) -> Dict[Tuple[int, ...], float]:
    """
    Return probabilities for all occupations, including leakage.
    """
    rho = np.asarray(rho, dtype=complex)
    probabilities: Dict[Tuple[int, ...], float] = {}

    for occupation in state_space.all_occupations():
        index = state_space.occupation_to_index(occupation)
        probabilities[tuple(occupation)] = float(np.real_if_close(rho[index, index]))

    return probabilities


def leakage_probability(
    state_space: MultiQubitStateSpace,
    rho: Array,
) -> float:
    """
    Return total leakage-subspace probability.
    """
    if state_space.local_dim < 3:
        return 0.0

    projector = state_space.leakage_projector()
    return expectation_value(rho, projector)


def assignment_probability_for_bit(
    actual_level: int,
    reported_bit: int,
    readout_noise: ReadoutNoise,
) -> float:
    """
    Probability of reporting one classical bit given actual local level.
    """
    readout_noise.validate()

    if reported_bit not in (0, 1):
        raise ValueError("reported_bit must be 0 or 1")

    if actual_level == 0:
        p_one = readout_noise.assignment_error_0_to_1
    elif actual_level == 1:
        p_one = 1.0 - readout_noise.assignment_error_1_to_0
    else:
        p_one = readout_noise.leakage_report_as_1

    return p_one if reported_bit == 1 else 1.0 - p_one


def observed_bitstring_probabilities(
    state_space: MultiQubitStateSpace,
    rho: Array,
    readout_noise: ReadoutNoise,
) -> Dict[Tuple[int, ...], float]:
    """
    Convert full basis probabilities into observed classical bitstring
    probabilities under independent readout assignment error.
    """
    readout_noise.validate()

    observed: Dict[Tuple[int, ...], float] = {
        bitstring: 0.0
        for bitstring in state_space.computational_occupations()
    }

    full_probs = all_basis_probabilities(state_space, rho)

    for occupation, probability in full_probs.items():
        for bitstring in state_space.computational_occupations():
            assignment_probability = 1.0

            for actual_level, reported_bit in zip(occupation, bitstring):
                assignment_probability *= assignment_probability_for_bit(
                    actual_level=actual_level,
                    reported_bit=reported_bit,
                    readout_noise=readout_noise,
                )

            observed[tuple(bitstring)] += probability * assignment_probability

    return observed


def observed_pauli_z_expectation(
    state_space: MultiQubitStateSpace,
    rho: Array,
    z_mask: Sequence[int],
    readout_noise: ReadoutNoise,
) -> float:
    """
    Observed Z-string expectation after classical readout assignment error.

    z_mask contains 0/1 entries. A 1 means include Z on that measured bit.
    """
    if len(z_mask) != state_space.num_qubits:
        raise ValueError("z_mask length does not match num_qubits")

    probabilities = observed_bitstring_probabilities(
        state_space=state_space,
        rho=rho,
        readout_noise=readout_noise,
    )

    value = 0.0

    for bitstring, probability in probabilities.items():
        parity = 1.0

        for bit, include_z in zip(bitstring, z_mask):
            if include_z:
                parity *= 1.0 if bit == 0 else -1.0

        value += parity * probability

    return float(value)


def make_noisy_config_from_params(
    local_dim: int,
    params: Dict[str, float],
) -> CPTPModelConfig:
    """
    Build a CPTPModelConfig from a flat parameter dictionary.

    This helper is useful for later finite-difference Jacobian code.
    Missing parameters default to zero.
    """
    config = CPTPModelConfig(
        local_noise=LocalNoiseRates(
            amplitude_damping=float(params.get("t1", params.get("amplitude_damping", 0.0))),
            pure_dephasing=float(params.get("tphi", params.get("pure_dephasing", 0.0))),
            depolarizing=float(params.get("local_depolarizing", 0.0)),
            leakage=float(params.get("leakage", 0.0)) if local_dim >= 3 else 0.0,
            seepage=float(params.get("seepage", 0.0)) if local_dim >= 3 else 0.0,
        ),
        two_qubit_noise=TwoQubitNoiseRates(
            two_qubit_depolarizing=float(params.get("two_qubit_depolarizing", 0.0)),
            zz_phase_jitter_std=float(params.get("zz_phase_jitter_std", 0.0)),
        ),
        readout_noise=ReadoutNoise(
            assignment_error_0_to_1=float(params.get("readout_0_to_1", 0.0)),
            assignment_error_1_to_0=float(params.get("readout_1_to_0", 0.0)),
            leakage_report_as_1=float(params.get("leakage_report_as_1", 0.5)),
        ),
    )

    config.validate(local_dim=local_dim)
    return config


def summarize_density_matrix(
    state_space: MultiQubitStateSpace,
    rho: Array,
    label: str,
) -> None:
    """
    Print compact diagnostics for a density matrix.
    """
    print(f"\n{label}")
    print("-" * 72)
    print(f"trace_error        = {trace_distance_from_one(rho):.6e}")
    print(f"hermiticity_error  = {hermiticity_error(rho):.6e}")
    print(f"min_eigenvalue     = {min_eigenvalue_hermitian(rho):.6e}")
    print(f"leakage_probability= {leakage_probability(state_space, rho):.6e}")


def run_trace_preservation_tests() -> None:
    """
    Test local and two-qubit Kraus trace preservation.
    """
    print("\nRunning Kraus trace-preservation tests...")

    for local_dim in (2, 3):
        kraus_sets = [
            ("amplitude_damping", local_amplitude_damping_kraus(local_dim, 0.1)),
            ("pure_dephasing", local_pure_dephasing_kraus(local_dim, 0.1)),
            ("depolarizing", local_depolarizing_kraus(local_dim, 0.1)),
        ]

        if local_dim >= 3:
            kraus_sets.append(("leakage", local_leakage_kraus(local_dim, 0.1)))
            kraus_sets.append(("seepage", local_seepage_kraus(local_dim, 0.1)))

        for name, kraus_ops in kraus_sets:
            error = kraus_trace_check(kraus_ops)
            print(f"local_dim={local_dim}, {name:18s}: {error:.3e}")

            if error > 1e-10:
                raise AssertionError(f"{name} Kraus set is not trace preserving")

        two_kraus = two_qubit_depolarizing_kraus(local_dim, 0.1)
        error = kraus_trace_check(two_kraus)
        print(f"local_dim={local_dim}, two_qubit_depol   : {error:.3e}")

        if error > 1e-10:
            raise AssertionError("two-qubit depolarizing Kraus set is not trace preserving")

    print("Kraus trace-preservation tests passed.")


def run_density_channel_tests() -> None:
    """
    Test density-matrix channel application.
    """
    print("\nRunning density-channel tests...")

    for num_qubits in (3, 4):
        for local_dim in (2, 3):
            state_space = MultiQubitStateSpace(
                num_qubits=num_qubits,
                local_dim=local_dim,
                leakage_level=2 if local_dim == 3 else None,
            )

            rho = state_space.basis_density([1] + [0] * (num_qubits - 1))

            config = default_cptp_config(local_dim=local_dim)
            coherent = TwoQubitCoherentParams(
                dphi=-0.01,
                theta=0.004,
                chi=0.3,
            )

            output = apply_two_qubit_cz_like_cycle(
                state_space=state_space,
                rho=rho,
                targets=(0, 1),
                coherent_params=coherent,
                config=config,
            )

            trace_error = trace_distance_from_one(output)
            herm_error = hermiticity_error(output)
            min_eig = min_eigenvalue_hermitian(output)

            print(
                f"num_qubits={num_qubits}, local_dim={local_dim}, "
                f"trace_error={trace_error:.3e}, "
                f"hermiticity_error={herm_error:.3e}, "
                f"min_eig={min_eig:.3e}"
            )

            if trace_error > 1e-10:
                raise AssertionError("channel failed trace preservation")

            if herm_error > 1e-10:
                raise AssertionError("channel failed hermiticity preservation")

            if min_eig < -1e-10:
                raise AssertionError("channel produced non-positive state")

    print("Density-channel tests passed.")


def run_readout_tests() -> None:
    """
    Test readout probability transformation.
    """
    print("\nRunning readout tests...")

    state_space = MultiQubitStateSpace(
        num_qubits=3,
        local_dim=3,
        leakage_level=2,
    )

    rho = state_space.basis_density((1, 0, 2))
    readout = ReadoutNoise(
        assignment_error_0_to_1=0.01,
        assignment_error_1_to_0=0.02,
        leakage_report_as_1=0.5,
    )

    probabilities = observed_bitstring_probabilities(
        state_space=state_space,
        rho=rho,
        readout_noise=readout,
    )

    total = sum(probabilities.values())
    print(f"observed probability sum = {total:.12f}")

    if not np.isclose(total, 1.0):
        raise AssertionError("readout probabilities do not sum to 1")

    zzi = observed_pauli_z_expectation(
        state_space=state_space,
        rho=rho,
        z_mask=(1, 1, 0),
        readout_noise=readout,
    )

    print(f"example observed ZZ expectation = {zzi:.6f}")
    print("Readout tests passed.")


def run_example_cycle() -> None:
    """
    Demonstrate one noisy cycle and diagnostics.
    """
    print("\nRunning example noisy CZ-like cycle...")

    state_space = MultiQubitStateSpace(
        num_qubits=4,
        local_dim=3,
        leakage_level=2,
    )

    initial_state = state_space.basis_state((0, 1, 0, 0))
    rho = state_space.density_from_state(initial_state)

    config = default_cptp_config(local_dim=3)
    coherent = TwoQubitCoherentParams(
        dphi=-0.01,
        theta=0.004,
        chi=0.3,
    )

    summarize_density_matrix(state_space, rho, "initial rho")

    for _ in range(8):
        rho = apply_two_qubit_cz_like_cycle(
            state_space=state_space,
            rho=rho,
            targets=(0, 1),
            coherent_params=coherent,
            config=config,
        )
        rho = apply_pauli_interleaving(
            state_space=state_space,
            rho=rho,
            pauli_string="XXII",
        )

    summarize_density_matrix(state_space, rho, "after 8 noisy interleaved cycles")

    z0 = observed_pauli_z_expectation(
        state_space=state_space,
        rho=rho,
        z_mask=(1, 0, 0, 0),
        readout_noise=config.readout_noise,
    )

    z1 = observed_pauli_z_expectation(
        state_space=state_space,
        rho=rho,
        z_mask=(0, 1, 0, 0),
        readout_noise=config.readout_noise,
    )

    print(f"observed <Z0> = {z0:.6f}")
    print(f"observed <Z1> = {z1:.6f}")


def main() -> None:
    """
    CLI entry point.
    """
    run_trace_preservation_tests()
    run_density_channel_tests()
    run_readout_tests()
    run_example_cycle()


if __name__ == "__main__":
    main()


# In[ ]:




