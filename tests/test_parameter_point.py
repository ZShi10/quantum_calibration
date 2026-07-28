import unittest

import numpy as np

from sensitivity_svd_selector import ParameterPoint

class ParameterPointTests(unittest.TestCase):
    def test_swap_coordinates_update_theta_and_chi(self):
        point = ParameterPoint()
        point = point.with_update("swap_x", 3.0e-3, local_dim=3)
        point = point.with_update("swap_y", 4.0e-3, local_dim=3)

        self.assertAlmostEqual(point.theta, 5.0e-3)
        self.assertAlmostEqual(point.chi, np.arctan2(4.0, 3.0))

    def test_polar_coordinates_update_swap_quadratures(self):
        point = ParameterPoint()
        point = point.with_update("theta", 5.0e-3, local_dim=3)
        point = point.with_update("chi", np.pi / 2.0, local_dim=3)

        self.assertAlmostEqual(point.swap_x, 0.0, places=15)
        self.assertAlmostEqual(point.swap_y, 5.0e-3)

    def test_physical_rates_are_clipped_to_unit_interval(self):
        point = ParameterPoint()

        below = point.with_update("t1", -0.1, local_dim=3)
        above = point.with_update("leakage", 1.1, local_dim=3)

        self.assertEqual(below.t1, 0.0)
        self.assertEqual(above.leakage, 1.0)

    def test_qubit_model_disables_leakage_and_seepage(self):
        point = ParameterPoint(leakage=0.2, seepage=0.3)
        point = point.with_update("leakage", 0.4, local_dim=2)
        point = point.with_update("seepage", 0.5, local_dim=2)

        self.assertEqual(point.leakage, 0.0)
        self.assertEqual(point.seepage, 0.0)

    def test_qutrit_config_preserves_noise_rates(self):
        point = ParameterPoint(t1=2.0e-4, tphi=3.0e-4, leakage=5.0e-5)
        config = point.cptp_config(local_dim=3)

        self.assertAlmostEqual(config.local_noise.amplitude_damping, point.t1)
        self.assertAlmostEqual(config.local_noise.pure_dephasing, point.tphi)
        self.assertAlmostEqual(config.local_noise.leakage, point.leakage)

if __name__ == "__main__":
    unittest.main()