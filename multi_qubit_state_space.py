#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
multi_qubit_state_space.py

Lightweight state-space utilities for 3-4 qubit MEADD-style calibration
simulation.

Run:

    python multi_qubit_state_space.py

Purpose
-------
This file provides the basis and operator infrastructure needed for the
next stages of the PPT plan:

    1. 3-4 qubit simulation
    2. computational basis + leakage level
    3. lightweight CPTP channel application
    4. Pauli / single-qubit interleaving
    5. Jacobian + SVD circuit selection

Design constraints
------------------
The target development machine is a normal PC:

    CPU: Intel i7-1165G7
    RAM: 16 GB
    GPU: integrated Intel Iris Xe

Therefore this module intentionally avoids dense superoperators of size
d^2 x d^2. It only builds Hilbert-space operators of size d x d when needed.

Typical dimensions:

    3 qubits, two levels each:
        dim = 2^3 = 8

    4 qubits, two levels each:
        dim = 2^4 = 16

    3 qubits with one leakage level each:
        dim = 3^3 = 27

    4 qubits with one leakage level each:
        dim = 3^4 = 81

A dense 81 x 81 complex matrix is small enough for a PC. A dense
6561 x 6561 superoperator is not appropriate for repeated simulations.

Convention
----------
Basis states are ordered lexicographically by occupation tuple.

For num_qubits=3, local_dim=2:

    index 0 -> (0, 0, 0)
    index 1 -> (0, 0, 1)
    index 2 -> (0, 1, 0)
    index 3 -> (0, 1, 1)
    index 4 -> (1, 0, 0)
    ...

The leftmost qubit is qubit 0.

This matches standard tensor-product ordering:

    |q0> tensor |q1> tensor ... tensor |qN-1>
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Array = np.ndarray
Occupation = Tuple[int, ...]


@dataclass(frozen=True)
class MultiQubitStateSpace:
    """
    Hilbert-space description for N qudits.

    Parameters
    ----------
    num_qubits:
        Number of physical qubits/qudits.

    local_dim:
        Local Hilbert-space dimension.
        Use local_dim=2 for computational subspace.
        Use local_dim=3 for computational + one leakage level.

    leakage_level:
        Optional local level used as leakage state.
        For local_dim=3 this is usually 2.
    """

    num_qubits: int
    local_dim: int = 2
    leakage_level: int | None = None

    def __post_init__(self) -> None:
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")

        if self.local_dim < 2:
            raise ValueError("local_dim must be at least 2")

        if self.leakage_level is not None:
            if not 0 <= self.leakage_level < self.local_dim:
                raise ValueError("leakage_level must be within local dimension")

    @property
    def dim(self) -> int:
        """
        Total Hilbert-space dimension.
        """
        return self.local_dim ** self.num_qubits

    @property
    def computational_dim(self) -> int:
        """
        Dimension of the computational subspace.
        """
        return 2 ** self.num_qubits

    def index_to_occupation(self, index: int) -> Occupation:
        """
        Convert basis index to occupation tuple.

        Example for num_qubits=3, local_dim=2:

            5 -> (1, 0, 1)
        """
        if not 0 <= index < self.dim:
            raise IndexError("basis index out of range")

        values = []
        remainder = int(index)

        for power in reversed(range(self.num_qubits)):
            base = self.local_dim ** power
            value = remainder // base
            remainder = remainder % base
            values.append(value)

        return tuple(values)

    def occupation_to_index(self, occupation: Sequence[int]) -> int:
        """
        Convert occupation tuple to basis index.

        Example for num_qubits=3, local_dim=2:

            (1, 0, 1) -> 5
        """
        if len(occupation) != self.num_qubits:
            raise ValueError("occupation length does not match num_qubits")

        index = 0

        for value in occupation:
            if not 0 <= int(value) < self.local_dim:
                raise ValueError("occupation value outside local dimension")
            index = index * self.local_dim + int(value)

        return index

    def all_occupations(self) -> List[Occupation]:
        """
        Return all basis occupations in lexicographic order.
        """
        return list(product(range(self.local_dim), repeat=self.num_qubits))

    def computational_occupations(self) -> List[Occupation]:
        """
        Return occupations in the computational subspace.
        """
        return list(product((0, 1), repeat=self.num_qubits))

    def leakage_occupations(self) -> List[Occupation]:
        """
        Return occupations containing at least one non-computational level.
        """
        return [
            occupation
            for occupation in self.all_occupations()
            if any(level >= 2 for level in occupation)
        ]

    def is_computational_occupation(self, occupation: Sequence[int]) -> bool:
        """
        Check whether an occupation belongs to the computational subspace.
        """
        return all(level in (0, 1) for level in occupation)

    def computational_indices(self) -> np.ndarray:
        """
        Return full-space indices belonging to the computational subspace.
        """
        return np.asarray(
            [
                self.occupation_to_index(occupation)
                for occupation in self.computational_occupations()
            ],
            dtype=int,
        )

    def leakage_indices(self) -> np.ndarray:
        """
        Return full-space indices belonging to leakage subspace.
        """
        return np.asarray(
            [
                self.occupation_to_index(occupation)
                for occupation in self.leakage_occupations()
            ],
            dtype=int,
        )

    def basis_state(self, occupation: Sequence[int]) -> Array:
        """
        Return ket vector for a basis occupation.
        """
        vector = np.zeros(self.dim, dtype=complex)
        vector[self.occupation_to_index(occupation)] = 1.0
        return vector

    def density_from_state(self, state: Array) -> Array:
        """
        Return density matrix |psi><psi| from a state vector.
        """
        state = np.asarray(state, dtype=complex)
        if state.shape != (self.dim,):
            raise ValueError("state has wrong dimension")

        return np.outer(state, np.conjugate(state))

    def basis_density(self, occupation: Sequence[int]) -> Array:
        """
        Return density matrix for a basis occupation.
        """
        state = self.basis_state(occupation)
        return self.density_from_state(state)

    def projector_on_indices(self, indices: Sequence[int]) -> Array:
        """
        Return projector onto a set of basis indices.
        """
        projector = np.zeros((self.dim, self.dim), dtype=complex)

        for index in indices:
            if not 0 <= int(index) < self.dim:
                raise IndexError("projector index out of range")
            projector[int(index), int(index)] = 1.0

        return projector

    def computational_projector(self) -> Array:
        """
        Return projector onto computational subspace.
        """
        return self.projector_on_indices(self.computational_indices())

    def leakage_projector(self) -> Array:
        """
        Return projector onto leakage subspace.
        """
        return self.projector_on_indices(self.leakage_indices())


def local_identity(local_dim: int) -> Array:
    """
    Local identity operator.
    """
    return np.eye(local_dim, dtype=complex)


def local_zero_projector(local_dim: int) -> Array:
    """
    Local |0><0| projector.
    """
    op = np.zeros((local_dim, local_dim), dtype=complex)
    op[0, 0] = 1.0
    return op


def local_one_projector(local_dim: int) -> Array:
    """
    Local |1><1| projector.
    """
    op = np.zeros((local_dim, local_dim), dtype=complex)
    op[1, 1] = 1.0
    return op


def local_level_projector(local_dim: int, level: int) -> Array:
    """
    Local |level><level| projector.
    """
    if not 0 <= level < local_dim:
        raise ValueError("level outside local dimension")

    op = np.zeros((local_dim, local_dim), dtype=complex)
    op[level, level] = 1.0
    return op


def local_transition(local_dim: int, bra_level: int, ket_level: int) -> Array:
    """
    Local |bra_level><ket_level| operator.
    """
    if not 0 <= bra_level < local_dim:
        raise ValueError("bra_level outside local dimension")
    if not 0 <= ket_level < local_dim:
        raise ValueError("ket_level outside local dimension")

    op = np.zeros((local_dim, local_dim), dtype=complex)
    op[bra_level, ket_level] = 1.0
    return op


def local_x(local_dim: int = 2) -> Array:
    """
    Embedded X operator acting on |0>, |1> subspace.

    For local_dim=3, leakage level |2> is left unchanged.
    """
    op = np.eye(local_dim, dtype=complex)
    op[0, 0] = 0.0
    op[1, 1] = 0.0
    op[0, 1] = 1.0
    op[1, 0] = 1.0
    return op


def local_y(local_dim: int = 2) -> Array:
    """
    Embedded Y operator acting on |0>, |1> subspace.

    For local_dim=3, leakage level |2> is left unchanged.
    """
    op = np.eye(local_dim, dtype=complex)
    op[0, 0] = 0.0
    op[1, 1] = 0.0
    op[0, 1] = -1.0j
    op[1, 0] = 1.0j
    return op


def local_z(local_dim: int = 2) -> Array:
    """
    Embedded Z operator acting on |0>, |1> subspace.

    For local_dim=3, leakage level |2> is left unchanged.
    """
    op = np.eye(local_dim, dtype=complex)
    op[0, 0] = 1.0
    op[1, 1] = -1.0
    return op


def local_h(local_dim: int = 2) -> Array:
    """
    Embedded Hadamard operator on |0>, |1> subspace.

    For local_dim=3, leakage level |2> is left unchanged.
    """
    op = np.eye(local_dim, dtype=complex)
    factor = 1.0 / np.sqrt(2.0)
    op[0, 0] = factor
    op[0, 1] = factor
    op[1, 0] = factor
    op[1, 1] = -factor
    return op


def local_phase_rotation(angle: float, local_dim: int = 2) -> Array:
    """
    Embedded virtual Z-like phase rotation.

    Uses diag(1, exp(i angle)) on the computational subspace and leaves
    leakage levels unchanged.
    """
    op = np.eye(local_dim, dtype=complex)
    op[1, 1] = np.exp(1.0j * float(angle))
    return op


def local_xy_rotation(angle: float, phase: float, local_dim: int = 2) -> Array:
    """
    Embedded XY rotation on the computational subspace.

    The computational block implements:

        exp[-i angle/2 * (cos(phase) X + sin(phase) Y)]

    Leakage levels are left unchanged.
    """
    c = np.cos(0.5 * float(angle))
    s = np.sin(0.5 * float(angle))
    phase = float(phase)

    axis = np.cos(phase) * local_x(2) + np.sin(phase) * local_y(2)
    block = c * np.eye(2, dtype=complex) - 1.0j * s * axis

    op = np.eye(local_dim, dtype=complex)
    op[:2, :2] = block
    return op


def kron_all(operators: Sequence[Array]) -> Array:
    """
    Kronecker product of a sequence of operators.
    """
    if not operators:
        raise ValueError("operators must be non-empty")

    result = np.asarray(operators[0], dtype=complex)

    for op in operators[1:]:
        result = np.kron(result, np.asarray(op, dtype=complex))

    return result


def embed_local_operator(
    state_space: MultiQubitStateSpace,
    local_operator: Array,
    target: int,
) -> Array:
    """
    Embed one local operator into the full Hilbert space.
    """
    if not 0 <= target < state_space.num_qubits:
        raise ValueError("target qubit out of range")

    local_operator = np.asarray(local_operator, dtype=complex)

    expected_shape = (state_space.local_dim, state_space.local_dim)
    if local_operator.shape != expected_shape:
        raise ValueError("local_operator has wrong shape")

    operators = []

    for qubit in range(state_space.num_qubits):
        if qubit == target:
            operators.append(local_operator)
        else:
            operators.append(local_identity(state_space.local_dim))

    return kron_all(operators)


def embed_two_local_operator(
    state_space: MultiQubitStateSpace,
    two_qubit_operator: Array,
    targets: Tuple[int, int],
) -> Array:
    """
    Embed a two-qudit operator into full Hilbert space.

    This implementation is index-based rather than a large tensor reshaping
    routine. It is simple and robust for dim <= 81.

    targets order matters. If targets=(a, b), then two_qubit_operator is
    interpreted in basis |qa> tensor |qb>.
    """
    target_a, target_b = targets

    if target_a == target_b:
        raise ValueError("targets must be distinct")

    if not 0 <= target_a < state_space.num_qubits:
        raise ValueError("target_a out of range")

    if not 0 <= target_b < state_space.num_qubits:
        raise ValueError("target_b out of range")

    local_dim = state_space.local_dim
    expected_shape = (local_dim * local_dim, local_dim * local_dim)

    two_qubit_operator = np.asarray(two_qubit_operator, dtype=complex)
    if two_qubit_operator.shape != expected_shape:
        raise ValueError("two_qubit_operator has wrong shape")

    full = np.zeros((state_space.dim, state_space.dim), dtype=complex)

    for input_index in range(state_space.dim):
        input_occ = list(state_space.index_to_occupation(input_index))

        local_input_index = input_occ[target_a] * local_dim + input_occ[target_b]

        for local_output_index in range(local_dim * local_dim):
            amp = two_qubit_operator[local_output_index, local_input_index]
            if abs(amp) == 0.0:
                continue

            output_a = local_output_index // local_dim
            output_b = local_output_index % local_dim

            output_occ = list(input_occ)
            output_occ[target_a] = output_a
            output_occ[target_b] = output_b

            output_index = state_space.occupation_to_index(output_occ)
            full[output_index, input_index] += amp

    return full


def computational_cz(local_dim: int = 2, phase: float = np.pi) -> Array:
    """
    Two-qudit controlled phase on computational levels.

    Uses diag(1, 1, 1, exp(i phase)) on |00>, |01>, |10>, |11>.
    Leakage states are left unchanged.
    """
    dim = local_dim * local_dim
    op = np.eye(dim, dtype=complex)

    index_11 = 1 * local_dim + 1
    op[index_11, index_11] = np.exp(1.0j * float(phase))

    return op


def computational_swap_like_error(
    local_dim: int = 2,
    theta: float = 0.0,
    chi: float = 0.0,
) -> Array:
    """
    Excitation-preserving swap-like coherent error on |01>, |10>.

    The computational odd-parity block is:

        [[cos(theta), i exp(i chi) sin(theta)],
         [i exp(-i chi) sin(theta), cos(theta)]]

    This is useful for testing theta/chi sensitivity in multi-qubit
    candidate circuits. Leakage states are left unchanged.
    """
    dim = local_dim * local_dim
    op = np.eye(dim, dtype=complex)

    index_01 = 0 * local_dim + 1
    index_10 = 1 * local_dim + 0

    c = np.cos(float(theta))
    s = np.sin(float(theta))
    phase = float(chi)

    op[index_01, index_01] = c
    op[index_10, index_10] = c
    op[index_01, index_10] = 1.0j * np.exp(1.0j * phase) * s
    op[index_10, index_01] = 1.0j * np.exp(-1.0j * phase) * s

    return op


def computational_zz_phase_error(
    local_dim: int = 2,
    dphi: float = 0.0,
) -> Array:
    """
    Residual controlled phase error around CZ.

    This applies diag(1, 1, 1, exp(i dphi)) in the computational two-qubit
    subspace and leaves leakage levels unchanged.
    """
    return computational_cz(local_dim=local_dim, phase=float(dphi))


def two_qubit_cz_like_unitary(
    local_dim: int = 2,
    dphi: float = 0.0,
    theta: float = 0.0,
    chi: float = 0.0,
) -> Array:
    """
    Lightweight CZ-like two-qudit unitary for candidate-circuit simulation.

    It combines:
        ideal CZ phase pi,
        residual controlled phase dphi,
        small odd-subspace swap theta, chi.

    This is not the final experimental CPTP channel. It is a compact coherent
    test unitary used before the experiment team supplies the full CPTP
    parameterization.
    """
    cz = computational_cz(local_dim=local_dim, phase=np.pi + float(dphi))
    swap = computational_swap_like_error(
        local_dim=local_dim,
        theta=float(theta),
        chi=float(chi),
    )
    return swap @ cz


def expectation_value(rho: Array, observable: Array) -> float:
    """
    Real expectation value Tr(rho observable).
    """
    value = np.trace(np.asarray(rho, dtype=complex) @ np.asarray(observable, dtype=complex))
    return float(np.real_if_close(value))


def state_expectation_value(state: Array, observable: Array) -> float:
    """
    Real expectation value <psi|observable|psi>.
    """
    state = np.asarray(state, dtype=complex)
    value = np.vdot(state, np.asarray(observable, dtype=complex) @ state)
    return float(np.real_if_close(value))


def apply_unitary_to_state(unitary: Array, state: Array) -> Array:
    """
    Apply unitary to state vector.
    """
    return np.asarray(unitary, dtype=complex) @ np.asarray(state, dtype=complex)


def apply_unitary_to_density(unitary: Array, rho: Array) -> Array:
    """
    Apply unitary to density matrix.
    """
    unitary = np.asarray(unitary, dtype=complex)
    rho = np.asarray(rho, dtype=complex)
    return unitary @ rho @ np.conjugate(unitary.T)


def normalize_state(state: Array) -> Array:
    """
    Normalize a state vector.
    """
    state = np.asarray(state, dtype=complex)
    norm = np.linalg.norm(state)

    if norm == 0.0:
        raise ValueError("cannot normalize zero state")

    return state / norm


def ket_plus_on_qubit(
    state_space: MultiQubitStateSpace,
    target: int,
    base_occupation: Sequence[int] | None = None,
) -> Array:
    """
    Prepare a state with target qubit in |+> and other qubits fixed.

    If base_occupation is None, all other qubits are initialized to |0>.
    """
    if base_occupation is None:
        occupation0 = [0] * state_space.num_qubits
    else:
        occupation0 = list(base_occupation)

    if len(occupation0) != state_space.num_qubits:
        raise ValueError("base_occupation has wrong length")

    occupation1 = list(occupation0)
    occupation0[target] = 0
    occupation1[target] = 1

    state = (
        state_space.basis_state(occupation0)
        + state_space.basis_state(occupation1)
    ) / np.sqrt(2.0)

    return state


def bell_odd_state(
    state_space: MultiQubitStateSpace,
    qubit_a: int,
    qubit_b: int,
    phase: complex = 1.0,
    base_occupation: Sequence[int] | None = None,
) -> Array:
    """
    Prepare odd-parity Bell-like state on qubit_a and qubit_b.

    State:

        (|01> + phase |10>) / sqrt(2)

    Other qubits are fixed by base_occupation or initialized to |0>.
    """
    if qubit_a == qubit_b:
        raise ValueError("qubits must be distinct")

    if base_occupation is None:
        occ_01 = [0] * state_space.num_qubits
    else:
        occ_01 = list(base_occupation)

    if len(occ_01) != state_space.num_qubits:
        raise ValueError("base_occupation has wrong length")

    occ_10 = list(occ_01)

    occ_01[qubit_a] = 0
    occ_01[qubit_b] = 1

    occ_10[qubit_a] = 1
    occ_10[qubit_b] = 0

    state = (
        state_space.basis_state(occ_01)
        + complex(phase) * state_space.basis_state(occ_10)
    )

    return normalize_state(state)


def pauli_string_operator(
    state_space: MultiQubitStateSpace,
    pauli_string: str,
) -> Array:
    """
    Build an embedded Pauli string.

    Supported characters:
        I, X, Y, Z

    For local_dim=3, Pauli operators act on |0>, |1> and leave leakage
    level |2> unchanged.
    """
    if len(pauli_string) != state_space.num_qubits:
        raise ValueError("pauli_string length does not match num_qubits")

    local_ops = []

    for char in pauli_string.upper():
        if char == "I":
            local_ops.append(local_identity(state_space.local_dim))
        elif char == "X":
            local_ops.append(local_x(state_space.local_dim))
        elif char == "Y":
            local_ops.append(local_y(state_space.local_dim))
        elif char == "Z":
            local_ops.append(local_z(state_space.local_dim))
        else:
            raise ValueError(f"unsupported Pauli character: {char}")

    return kron_all(local_ops)


def estimate_dense_matrix_memory_mb(dim: int, dtype_bytes: int = 16) -> float:
    """
    Estimate memory of one dim x dim dense complex matrix in MB.

    complex128 uses 16 bytes per entry.
    """
    return float(dim * dim * dtype_bytes / 1024.0 / 1024.0)


def estimate_superoperator_memory_mb(dim: int, dtype_bytes: int = 16) -> float:
    """
    Estimate memory of one dense Liouville superoperator in MB.
    """
    super_dim = dim * dim
    return estimate_dense_matrix_memory_mb(super_dim, dtype_bytes=dtype_bytes)


def print_dimension_report() -> None:
    """
    Print dimension and memory estimates.
    """
    print("\nDimension report")
    print("-" * 72)
    print(
        f"{'num_qubits':>10s} "
        f"{'local_dim':>10s} "
        f"{'hilbert_dim':>12s} "
        f"{'matrix_MB':>12s} "
        f"{'superop_MB':>12s}"
    )

    for num_qubits in (3, 4):
        for local_dim in (2, 3):
            space = MultiQubitStateSpace(
                num_qubits=num_qubits,
                local_dim=local_dim,
                leakage_level=2 if local_dim == 3 else None,
            )

            matrix_mb = estimate_dense_matrix_memory_mb(space.dim)
            superop_mb = estimate_superoperator_memory_mb(space.dim)

            print(
                f"{num_qubits:10d} "
                f"{local_dim:10d} "
                f"{space.dim:12d} "
                f"{matrix_mb:12.3f} "
                f"{superop_mb:12.3f}"
            )


def run_self_tests() -> None:
    """
    Lightweight self-tests.
    """
    print("Running multi_qubit_state_space.py self-tests...")

    for num_qubits in (3, 4):
        for local_dim in (2, 3):
            space = MultiQubitStateSpace(
                num_qubits=num_qubits,
                local_dim=local_dim,
                leakage_level=2 if local_dim == 3 else None,
            )

            for index in range(space.dim):
                occupation = space.index_to_occupation(index)
                recovered = space.occupation_to_index(occupation)
                if recovered != index:
                    raise AssertionError("index/occupation conversion failed")

            computational_indices = space.computational_indices()
            if computational_indices.size != 2 ** num_qubits:
                raise AssertionError("wrong computational subspace size")

            projector = space.computational_projector()
            if projector.shape != (space.dim, space.dim):
                raise AssertionError("projector shape mismatch")

            if not np.allclose(projector @ projector, projector):
                raise AssertionError("computational projector is not idempotent")

    space = MultiQubitStateSpace(num_qubits=3, local_dim=2)

    x0 = embed_local_operator(space, local_x(2), target=0)
    state_000 = space.basis_state((0, 0, 0))
    state_100 = space.basis_state((1, 0, 0))

    if not np.allclose(x0 @ state_000, state_100):
        raise AssertionError("embedded X on qubit 0 failed")

    x2 = embed_local_operator(space, local_x(2), target=2)
    state_001 = space.basis_state((0, 0, 1))

    if not np.allclose(x2 @ state_000, state_001):
        raise AssertionError("embedded X on qubit 2 failed")

    cz01 = embed_two_local_operator(
        space,
        computational_cz(local_dim=2, phase=np.pi),
        targets=(0, 1),
    )
    state_110 = space.basis_state((1, 1, 0))
    state_after = cz01 @ state_110

    if not np.allclose(state_after, -state_110):
        raise AssertionError("embedded CZ phase failed")

    state_101 = space.basis_state((1, 0, 1))
    state_after = cz01 @ state_101

    if not np.allclose(state_after, state_101):
        raise AssertionError("embedded CZ affected wrong basis state")

    plus = ket_plus_on_qubit(space, target=1)
    norm = np.linalg.norm(plus)

    if not np.isclose(norm, 1.0):
        raise AssertionError("|+> state is not normalized")

    bell = bell_odd_state(space, qubit_a=0, qubit_b=1, phase=1.0)
    if not np.isclose(np.linalg.norm(bell), 1.0):
        raise AssertionError("Bell odd state is not normalized")

    zzz = pauli_string_operator(space, "ZZZ")
    value = state_expectation_value(state_000, zzz)

    if not np.isclose(value, 1.0):
        raise AssertionError("Pauli string expectation failed")

    print("All self-tests passed.")


def main() -> None:
    """
    CLI entry point.
    """
    print_dimension_report()
    run_self_tests()


if __name__ == "__main__":
    main()


# In[ ]:




