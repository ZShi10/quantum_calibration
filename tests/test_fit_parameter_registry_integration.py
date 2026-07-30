import unittest

from cz_core_7_recovery_study import seven_parameter_fit_spec
from synthetic_parameter_fit import (
    CZ_CORE_7_PARAMETER_NAMES,
    default_fit_parameter_spec,
    fit_parameter_spec_from_registry,
)

EXPECTED_SCALES = {
    "dphi": 1.0e-2,
    "swap_x": 5.0e-3,
    "swap_y": 5.0e-3,
    "t1": 2.0e-4,
    "tphi": 3.0e-4,
    "leakage": 5.0e-5,
    "two_qubit_depolarizing": 5.0e-4,
}

EXPECTED_LOWER_BOUNDS = {
    "dphi": -0.1,
    "swap_x": -0.05,
    "swap_y": -0.05,
    "t1": 0.0,
    "tphi": 0.0,
    "leakage": 0.0,
    "two_qubit_depolarizing": 0.0,
}

EXPECTED_UPPER_BOUNDS = {
    "dphi": 0.1,
    "swap_x": 0.05,
    "swap_y": 0.05,
    "t1": 5.0e-3,
    "tphi": 5.0e-3,
    "leakage": 2.0e-3,
    "two_qubit_depolarizing": 1.0e-2,
}

class FitParameterRegistryIntegrationTests(unittest.TestCase):
    def assert_core_spec(self, spec):
        self.assertEqual(spec.names, CZ_CORE_7_PARAMETER_NAMES)
        self.assertEqual(spec.scales, EXPECTED_SCALES)
        self.assertEqual(spec.lower_bounds, EXPECTED_LOWER_BOUNDS)
        self.assertEqual(spec.upper_bounds, EXPECTED_UPPER_BOUNDS)

    def test_default_fit_spec_uses_registered_metadata(self):
        self.assert_core_spec(default_fit_parameter_spec())

    def test_recovery_study_uses_same_registered_metadata(self):
        self.assert_core_spec(seven_parameter_fit_spec())

    def test_parameter_without_fit_metadata_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "theta has no default fitting metadata",
        ):
            fit_parameter_spec_from_registry(("theta",))

    def test_requested_parameter_order_is_preserved(self):
        names = ("tphi", "dphi", "t1")
        spec = fit_parameter_spec_from_registry(names)

        self.assertEqual(spec.names, names)
        self.assertEqual(tuple(spec.scales), names)
        self.assertEqual(tuple(spec.lower_bounds), names)
        self.assertEqual(tuple(spec.upper_bounds), names)

if __name__ == "__main__":
    unittest.main()