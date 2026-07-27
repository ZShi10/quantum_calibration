#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, exp, isfinite, pi, sin, sqrt
from typing import List, Sequence

import numpy as np
from scipy.linalg import expm

Array = np.ndarray

@dataclass(frozen=True)
class X2PParams:
    delta: float = 0.0
    eta01: float = 0.0
    epsilon: float = 0.0
    eta12: float = 0.0
    include_leakage_unitary: bool = False

@dataclass(frozen=True)
class CZParams:
    dphi: float = 0.0
    swap_x: float = 0.0
    swap_y: float = 0.0
    gamma: float = 0.0
    zeta: float = 0.0

def dagger(a: Array) -> Array:
    return np.conjugate(a.T)

def basis(dim: int, index: int) -> Array:
    v = np.zeros((dim,), dtype=complex)
    v[index] = 1.0
    return v

def projector(dim: int, ket_index: int, bra_index: int) -> Array:
    return np.outer(basis(dim, ket_index), np.conjugate(basis(dim, bra_index)))

def kron_all(ops: Sequence[Array]) -> Array:
    out = np.asarray(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def embed_one_qudit(op: Array, target: int, num_qudits: int, local_dim: int = 3) -> Array:
    eye = np.eye(local_dim, dtype=complex)
    return kron_all([op if q == target else eye for q in range(num_qudits)])

def is_unitary(u: Array, atol: float = 1.0e-10) -> bool:
    eye = np.eye(u.shape[0], dtype=complex)
    return np.allclose(dagger(u) @ u, eye, atol=atol)

def is_cptp(kraus: Sequence[Array], atol: float = 1.0e-10) -> bool:
    dim = kraus[0].shape[1]
    acc = np.zeros((dim, dim), dtype=complex)
    for k in kraus:
        acc += dagger(k) @ k
    return np.allclose(acc, np.eye(dim, dtype=complex), atol=atol)

def x2p_unitary(params: X2PParams) -> Array:
    delta = params.delta
    eta01 = params.eta01
    angle = pi / 4.0 + delta / 2.0

    u0 = np.eye(3, dtype=complex)
    u0[0, 0] = cos(angle) * np.exp(-0.5j * eta01)
    u0[0, 1] = -1j * sin(angle) * np.exp(-0.5j * eta01)
    u0[1, 0] = -1j * sin(angle) * np.exp(0.5j * eta01)
    u0[1, 1] = cos(angle) * np.exp(0.5j * eta01)

    if not params.include_leakage_unitary:
        return u0

    epsilon = params.epsilon
    eta12 = params.eta12

    u1 = np.eye(3, dtype=complex)
    u1[1, 1] = cos(epsilon / 2.0) * np.exp(-0.5j * eta12)
    u1[1, 2] = -1j * sin(epsilon / 2.0) * np.exp(-0.5j * eta12)
    u1[2, 1] = -1j * sin(epsilon / 2.0) * np.exp(0.5j * eta12)
    u1[2, 2] = cos(epsilon / 2.0) * np.exp(0.5j * eta12)

    return u1 @ u0

def cz_unitary(params: CZParams) -> Array:
    theta = sqrt(params.swap_x * params.swap_x + params.swap_y * params.swap_y)
    chi = atan2(params.swap_y, params.swap_x) if theta > 0.0 else 0.0
    phi = pi + params.dphi
    gamma = params.gamma
    zeta = params.zeta

    u = np.zeros((4, 4), dtype=complex)
    u[0, 0] = np.exp(1j * gamma)
    u[1, 1] = cos(theta) * np.exp(-1j * zeta)
    u[1, 2] = -1j * sin(theta) * np.exp(-1j * chi)
    u[2, 1] = -1j * sin(theta) * np.exp(1j * chi)
    u[2, 2] = cos(theta) * np.exp(1j * zeta)
    u[3, 3] = np.exp(-1j * (phi + gamma))
    return u

def cz_unitary_qutrit(params: CZParams) -> Array:
    u4 = cz_unitary(params)
    u9 = np.eye(9, dtype=complex)
    comp = [0, 1, 3, 4]
    for i, row in enumerate(comp):
        for j, col in enumerate(comp):
            u9[row, col] = u4[i, j]
    return u9

def cz_leakage_kraus(leakage: float) -> List[Array]:
    if not 0.0 <= leakage <= 1.0:
        raise ValueError("leakage must be in [0, 1].")

    dim = 9
    idx_11 = 4
    idx_20 = 6

    c0 = np.eye(dim, dtype=complex)
    c0[idx_11, idx_11] = sqrt(1.0 - leakage)

    c1 = np.zeros((dim, dim), dtype=complex)
    c1[idx_20, idx_11] = sqrt(leakage)

    return [c0, c1]

def _validate_duration_and_time_constant(
    duration: float,
    time_constant: float,
    name: str,
) -> None:
    if not isfinite(float(duration)) or float(duration) < 0.0:
        raise ValueError("duration must be a finite non-negative number")
    if not isfinite(float(time_constant)) or float(time_constant) <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")


def _validate_local_dim(local_dim: int) -> None:
    if int(local_dim) not in (2, 3):
        raise ValueError("local_dim must be 2 or 3")


def _kraus_from_superoperator(superoperator: Array, dim: int) -> List[Array]:
    choi = np.zeros((dim * dim, dim * dim), dtype=complex)
    for row in range(dim):
        for col in range(dim):
            basis_op = np.zeros((dim, dim), dtype=complex)
            basis_op[row, col] = 1.0
            image_vec = superoperator @ basis_op.reshape(dim * dim, order="F")
            image = image_vec.reshape((dim, dim), order="F")
            choi += np.kron(basis_op, image)

    choi = 0.5 * (choi + dagger(choi))
    eigenvalues, eigenvectors = np.linalg.eigh(choi)
    kraus: List[Array] = []
    threshold = 1.0e-14
    for value, vector in zip(eigenvalues, eigenvectors.T):
        if value > threshold:
            kraus.append(sqrt(float(value)) * vector.reshape((dim, dim), order="F"))

    if not kraus:
        return [np.eye(dim, dtype=complex)]
    return kraus


def _lindblad_kraus(local_dim: int, duration: float, jump_ops: Sequence[Array]) -> List[Array]:
    dim = int(local_dim)
    identity = np.eye(dim, dtype=complex)
    generator = np.zeros((dim * dim, dim * dim), dtype=complex)

    for jump in jump_ops:
        jump = np.asarray(jump, dtype=complex)
        rate_op = dagger(jump) @ jump
        generator += np.kron(np.conjugate(jump), jump)
        generator -= 0.5 * np.kron(identity, rate_op)
        generator -= 0.5 * np.kron(rate_op.T, identity)

    superoperator = expm(float(duration) * generator)
    return _kraus_from_superoperator(superoperator, dim)


def generalized_amplitude_damping_kraus(
    local_dim: int,
    duration: float,
    t1: float,
    thermal_excited_population: float = 0.0,
) -> List[Array]:
    """
    Finite-time generalized amplitude damping from the thermal Lindblad model.

    The internal unit for duration and t1 is seconds. thermal_excited_population
    is the fixed-point population of |1>.
    """
    _validate_local_dim(local_dim)
    _validate_duration_and_time_constant(duration, t1, "t1")
    p_thermal = float(thermal_excited_population)
    if not isfinite(p_thermal) or not 0.0 <= p_thermal <= 1.0:
        raise ValueError("thermal_excited_population must be in [0, 1]")

    dim = int(local_dim)
    if float(duration) == 0.0:
        return [np.eye(dim, dtype=complex)]

    if dim == 2:
        damping_probability = 1.0 - exp(-float(duration) / float(t1))
        ground_population = 1.0 - p_thermal
        excited_population = p_thermal

        k0 = sqrt(ground_population) * np.asarray(
            [[1.0, 0.0], [0.0, sqrt(1.0 - damping_probability)]],
            dtype=complex,
        )
        k1 = sqrt(ground_population * damping_probability) * projector(2, 0, 1)
        k2 = sqrt(excited_population) * np.asarray(
            [[sqrt(1.0 - damping_probability), 0.0], [0.0, 1.0]],
            dtype=complex,
        )
        k3 = sqrt(excited_population * damping_probability) * projector(2, 1, 0)
        return [k0, k1, k2, k3]

    l_down = sqrt((1.0 - p_thermal) / float(t1)) * projector(dim, 0, 1)
    l_up = sqrt(p_thermal / float(t1)) * projector(dim, 1, 0)
    jumps = [l_down]
    if p_thermal > 0.0:
        jumps.append(l_up)
    return _lindblad_kraus(dim, float(duration), jumps)


def projector_dephasing_kraus(
    local_dim: int,
    duration: float,
    tphi: float,
) -> List[Array]:
    """
    Finite-time pure dephasing generated by sqrt(1/Tphi) |1><1|.
    """
    _validate_local_dim(local_dim)
    _validate_duration_and_time_constant(duration, tphi, "tphi")
    dim = int(local_dim)
    if float(duration) == 0.0:
        return [np.eye(dim, dtype=complex)]

    prob = 0.5 * (1.0 - exp(-float(duration) / (2.0 * float(tphi))))
    z = np.eye(dim, dtype=complex)
    z[1, 1] = -1.0

    return [
        sqrt(1.0 - prob) * np.eye(dim, dtype=complex),
        sqrt(prob) * z,
    ]


def amplitude_damping_kraus(t_gate: float, t1: float, p_thermal: float = 0.0) -> List[Array]:
    """
    Compatibility wrapper for qutrit generalized amplitude damping.
    """
    return generalized_amplitude_damping_kraus(
        local_dim=3,
        duration=t_gate,
        t1=t1,
        thermal_excited_population=p_thermal,
    )


def dephasing_kraus(t_gate: float, tphi: float) -> List[Array]:
    """
    Compatibility wrapper for qutrit projector dephasing.
    """
    return projector_dephasing_kraus(local_dim=3, duration=t_gate, tphi=tphi)

def depolarizing_kraus(prob: float, dim: int) -> List[Array]:
    if not 0.0 <= prob <= 1.0:
        raise ValueError("prob must be in [0, 1].")

    kraus = [sqrt(1.0 - prob) * np.eye(dim, dtype=complex)]
    weight = sqrt(prob / (dim * dim - 1))

    for a in range(dim):
        for b in range(dim):
            if a == 0 and b == 0:
                continue
            x = np.roll(np.eye(dim, dtype=complex), shift=a, axis=1)
            z = np.diag([np.exp(2j * pi * b * k / dim) for k in range(dim)])
            kraus.append(weight * x @ z)

    return kraus

def apply_kraus(rho: Array, kraus: Sequence[Array]) -> Array:
    out = np.zeros_like(rho, dtype=complex)
    for k in kraus:
        out += k @ rho @ dagger(k)
    return out

def _run_basic_checks() -> None:
    assert is_unitary(x2p_unitary(X2PParams()))
    assert is_unitary(x2p_unitary(X2PParams(epsilon=0.02, include_leakage_unitary=True)))

    assert is_unitary(cz_unitary(CZParams()))
    assert is_unitary(cz_unitary_qutrit(CZParams(dphi=1.0e-2, swap_x=3.0e-3, swap_y=2.0e-3)))

    assert is_cptp(cz_leakage_kraus(0.0))
    assert is_cptp(cz_leakage_kraus(1.0e-3))

    assert is_cptp(amplitude_damping_kraus(t_gate=32.0e-9, t1=50.0e-6))
    assert is_cptp(dephasing_kraus(t_gate=32.0e-9, tphi=60.0e-6))
    assert is_cptp(depolarizing_kraus(prob=1.0e-4, dim=3))

    rho_11 = projector(9, 4, 4)
    leaked = apply_kraus(rho_11, cz_leakage_kraus(0.02))
    assert np.isclose(np.real(leaked[4, 4]), 0.98)
    assert np.isclose(np.real(leaked[6, 6]), 0.02)

    print("basic CPTP/unitary checks passed")

if __name__ == "__main__":
    _run_basic_checks()


# In[ ]:



