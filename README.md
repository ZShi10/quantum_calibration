# Quantum Calibration

A research prototype for automatic quantum-gate calibration and
physics-informed experiment design.

## Current milestone

Milestone 0: CZ-core synthetic recovery closed loop.

The current validation workflow covers:

- physics-informed circuit generation
- finite-difference Jacobian construction
- SVD-based circuit selection
- synthetic finite-shot measurements
- bounded parameter fitting
- covariance estimation
- multi-start recovery
- held-out circuit prediction

## CZ-core parameters

The current model fits seven parameters:

- `dphi`
- `swap_x`
- `swap_y`
- `t1`
- `tphi`
- `leakage`
- `two_qubit_depolarizing`

## Project files

- `cptp_noise_models.py`: CPTP noise-channel construction
- `multi_qubit_state_space.py`: multi-qubit and multi-qutrit state-space utilities
- `lightweight_cptp_channel.py`: lightweight channel implementation
- `multi_qubit_circuit_simulator.py`: circuit and observable simulation
- `sensitivity_svd_selector.py`: Jacobian analysis and circuit selection
- `synthetic_parameter_fit.py`: synthetic data generation and parameter fitting
- `cz_core_7_recovery_study.py`: formal CZ-core-7 recovery validation

## Requirements

- Python 3.10 or later
- NumPy
- SciPy

## Roadmap
M0: CZ-core synthetic recovery closed loop
M1: unified parameter, circuit, and observation interfaces
M2: physics-informed automatic experiment design
M3: single-qubit calibration
M4: expanded CZ noise models
M5: global multi-gate calibration
M6: adaptive Bayesian calibration
