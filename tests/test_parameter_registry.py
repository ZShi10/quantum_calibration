import unittest

from parameter_registry import (
    DEFAULT_PARAMETER_REGISTRY,
    ParameterRegistry,
    ParameterSpec,
)
from sensitivity_svd_selector import FiniteDifferenceConfig

EXPECTED_PARAMETER_NAMES = {
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
}

class ParameterSpecTests(unittest.TestCase):
    def test_rejects_non_positive_finite_difference_step(self):
        with self.assertRaises(ValueError):
            ParameterSpec(
                name="invalid",
                finite_difference_step=0.0,
            ).validate()

    def test_rejects_incomplete_fit_metadata(self):
        with self.assertRaises(ValueError):
            ParameterSpec(
                name="invalid",
                finite_difference_step=1.0e-4,
                fit_scale=1.0,
            ).validate()

    def test_rejects_invalid_fit_bounds(self):
        with self.assertRaises(ValueError):
            ParameterSpec(
                name="invalid",
                finite_difference_step=1.0e-4,
                fit_scale=1.0,
                lower_bound=1.0,
                upper_bound=1.0,
            ).validate()

class ParameterRegistryTests(unittest.TestCase):
    def test_default_registry_contains_all_parameter_names(self):
        self.assertEqual(
            set(DEFAULT_PARAMETER_REGISTRY.names),
            EXPECTED_PARAMETER_NAMES,
        )

    def test_default_finite_difference_steps_are_positive(self):
        for name in DEFAULT_PARAMETER_REGISTRY.names:
            self.assertGreater(
                DEFAULT_PARAMETER_REGISTRY.finite_difference_step(name),
                0.0,
            )

    def test_core_fit_metadata_matches_existing_configuration(self):
        expected = {
            "dphi": (1.0e-2, -0.1, 0.1),
            "swap_x": (5.0e-3, -0.05, 0.05),
            "swap_y": (5.0e-3, -0.05, 0.05),
            "t1": (2.0e-4, 0.0, 5.0e-3),
            "tphi": (3.0e-4, 0.0, 5.0e-3),
            "leakage": (5.0e-5, 0.0, 2.0e-3),
            "two_qubit_depolarizing": (
                5.0e-4,
                0.0,
                1.0e-2,
            ),
        }

        for name, values in expected.items():
            spec = DEFAULT_PARAMETER_REGISTRY.get(name)
            self.assertEqual(
                (spec.fit_scale, spec.lower_bound, spec.upper_bound),
                values,
            )

    def test_leakage_and_seepage_require_qutrit_dimension(self):
        for name in ("leakage", "seepage"):
            self.assertFalse(
                DEFAULT_PARAMETER_REGISTRY.supports_local_dim(name, 2)
            )
            self.assertTrue(
                DEFAULT_PARAMETER_REGISTRY.supports_local_dim(name, 3)
            )

    def test_unknown_parameter_raises_clear_error(self):
        with self.assertRaisesRegex(KeyError, "unknown parameter"):
            DEFAULT_PARAMETER_REGISTRY.get("not_a_parameter")

    def test_duplicate_parameter_is_rejected(self):
        spec = ParameterSpec(
            name="duplicate",
            finite_difference_step=1.0e-4,
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            ParameterRegistry((spec, spec))

class FiniteDifferenceConfigTests(unittest.TestCase):
    def test_uses_registry_default(self):
        config = FiniteDifferenceConfig()
        self.assertEqual(config.step_for("dphi"), 1.0e-4)
        self.assertEqual(config.step_for("tphi"), 2.0e-5)

    def test_explicit_override_has_priority(self):
        config = FiniteDifferenceConfig(
            step_sizes={"dphi": 7.5e-5}
        )
        self.assertEqual(config.step_for("dphi"), 7.5e-5)

    def test_rejects_non_positive_override(self):
        config = FiniteDifferenceConfig(
            step_sizes={"dphi": 0.0}
        )

        with self.assertRaisesRegex(ValueError, "must be positive"):
            config.step_for("dphi")

if __name__ == "__main__":
    unittest.main()