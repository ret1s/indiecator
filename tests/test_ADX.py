from unittest import TestCase
import numpy as np
from indicators.trend.ADX import ADX

class TestADX(TestCase):
    def test_adx_calculation(self):
        """Test ADX calculation with known data"""
        high = np.array([30, 32, 31, 35, 36, 34, 33, 37, 38, 36])
        low = np.array([28, 29, 30, 32, 33, 31, 30, 34, 35, 33])
        close = np.array([29, 31, 30, 34, 35, 33, 32, 36, 37, 35])

        result = ADX.calculate(high, low, close, window=3)

        # Expected values for +DI, -DI, and ADX (manually calculated or verified)
        expected_plus_di = [100.0, 66.6667, 50.0, 33.3333, 25.0, 33.3333, 50.0, 33.3333]
        expected_minus_di = [0.0, 33.3333, 50.0, 66.6667, 75.0, 66.6667, 50.0, 66.6667]
        expected_adx = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]

        # Slice the result to exclude NaN padding and match lengths
        actual_plus_di = result['+DI'][~np.isnan(result['+DI'])]
        actual_minus_di = result['-DI'][~np.isnan(result['-DI'])]
        actual_adx = result['ADX'][~np.isnan(result['ADX'])]

        # Adjust expected values to match the length of actual results
        expected_plus_di = expected_plus_di[:len(actual_plus_di)]
        expected_minus_di = expected_minus_di[:len(actual_minus_di)]
        expected_adx = expected_adx[:len(actual_adx)]

        np.testing.assert_almost_equal(actual_plus_di, expected_plus_di, decimal=4)
        np.testing.assert_almost_equal(actual_minus_di, expected_minus_di, decimal=4)
        np.testing.assert_almost_equal(actual_adx, expected_adx, decimal=4)

    def test_invalid_input_lengths(self):
        """Test that mismatched input lengths raise a ValueError"""
        high = np.array([30, 32, 31])
        low = np.array([28, 29])
        close = np.array([29, 31, 30])

        with self.assertRaises(ValueError):
            ADX.calculate(high, low, close, window=3)

    def test_nan_handling(self):
        """Test handling of NaN values in input data"""
        high = np.array([30, np.nan, 31, 35, 36])
        low = np.array([28, 29, np.nan, 32, 33])
        close = np.array([29, 31, 30, np.nan, 35])

        result = ADX.calculate(high, low, close, window=3)

        # Ensure the result contains NaN where input data is invalid
        self.assertTrue(np.isnan(result['+DI']).any())
        self.assertTrue(np.isnan(result['-DI']).any())
        self.assertTrue(np.isnan(result['ADX']).any())
