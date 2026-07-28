"""
Central metadata registry for model parameters.

This module intentionally contains metadata only. Parameter values and
coordinate synchronization remain the responsibility of ParameterPoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

@dataclass(frozen=True)
class ParameterSpec:
    """
    Metadata for one model parameter.

    finite_difference_step:
        Default perturbation used to estimate Jacobian columns.

    fit_scale / lower_bound / upper_bound:
        Optional fitting metadata. These fields are populated together for
        parameters supported by the default fitting configuration.

    minimum_local_dim:
        Minimum local Hilbert-space dimension required by the parameter.
        Leakage and seepage require qutrits and therefore use 3.
    """

    name: str
    finite_difference_step: float
    fit_scale: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    minimum_local_dim: int = 2

    @property
    def supports_default_fit(self) -> bool:
        return self.fit_scale is not None

    def validate(self) -> None:
        if not self.name:
            raise ValueError("parameter name must be non-empty")

        if self.finite_difference_step <= 0.0:
            raise ValueError(
                f"finite-difference step for {self.name} must be positive"
            )

        if self.minimum_local_dim < 2:
            raise ValueError(
                f"minimum local dimension for {self.name} must be at least 2"
            )

        fit_fields = (
            self.fit_scale,
            self.lower_bound,
            self.upper_bound,
        )
        has_any_fit_field = any(value is not None for value in fit_fields)
        has_all_fit_fields = all(value is not None for value in fit_fields)

        if has_any_fit_field and not has_all_fit_fields:
            raise ValueError(
                f"incomplete fitting metadata for {self.name}"
            )

        if has_all_fit_fields:
            assert self.fit_scale is not None
            assert self.lower_bound is not None
            assert self.upper_bound is not None

            if self.fit_scale <= 0.0:
                raise ValueError(
                    f"fit scale for {self.name} must be positive"
                )

            if self.lower_bound >= self.upper_bound:
                raise ValueError(
                    f"invalid fitting bounds for {self.name}"
                )

class ParameterRegistry:
    """
    Immutable-by-convention collection of parameter metadata.
    """

    def __init__(self, specs: Iterable[ParameterSpec]):
        entries: Dict[str, ParameterSpec] = {}

        for spec in specs:
            spec.validate()

            if spec.name in entries:
                raise ValueError(
                    f"duplicate parameter specification: {spec.name}"
                )

            entries[spec.name] = spec

        if not entries:
            raise ValueError("parameter registry must be non-empty")

        self._entries = entries

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(self._entries)

    def get(self, name: str) -> ParameterSpec:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown parameter: {name}") from exc

    def finite_difference_step(self, name: str) -> float:
        return self.get(name).finite_difference_step

    def supports_local_dim(self, name: str, local_dim: int) -> bool:
        return local_dim >= self.get(name).minimum_local_dim

DEFAULT_PARAMETER_REGISTRY = ParameterRegistry(
    (
        ParameterSpec(
            name="dphi",
            finite_difference_step=1.0e-4,
            fit_scale=1.0e-2,
            lower_bound=-0.1,
            upper_bound=0.1,
        ),
        ParameterSpec(
            name="theta",
            finite_difference_step=5.0e-5,
        ),
        ParameterSpec(
            name="chi",
            finite_difference_step=1.0e-3,
        ),
        ParameterSpec(
            name="swap_x",
            finite_difference_step=5.0e-5,
            fit_scale=5.0e-3,
            lower_bound=-0.05,
            upper_bound=0.05,
        ),
        ParameterSpec(
            name="swap_y",
            finite_difference_step=5.0e-5,
            fit_scale=5.0e-3,
            lower_bound=-0.05,
            upper_bound=0.05,
        ),
        ParameterSpec(
            name="t1",
            finite_difference_step=2.0e-5,
            fit_scale=2.0e-4,
            lower_bound=0.0,
            upper_bound=5.0e-3,
        ),
        ParameterSpec(
            name="tphi",
            finite_difference_step=2.0e-5,
            fit_scale=3.0e-4,
            lower_bound=0.0,
            upper_bound=5.0e-3,
        ),
        ParameterSpec(
            name="local_depolarizing",
            finite_difference_step=1.0e-5,
        ),
        ParameterSpec(
            name="leakage",
            finite_difference_step=1.0e-5,
            fit_scale=5.0e-5,
            lower_bound=0.0,
            upper_bound=2.0e-3,
            minimum_local_dim=3,
        ),
        ParameterSpec(
            name="seepage",
            finite_difference_step=1.0e-5,
            minimum_local_dim=3,
        ),
        ParameterSpec(
            name="two_qubit_depolarizing",
            finite_difference_step=2.0e-5,
            fit_scale=5.0e-4,
            lower_bound=0.0,
            upper_bound=1.0e-2,
        ),
        ParameterSpec(
            name="readout_0_to_1",
            finite_difference_step=1.0e-4,
        ),
        ParameterSpec(
            name="readout_1_to_0",
            finite_difference_step=1.0e-4,
        ),
    )
)