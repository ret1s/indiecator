import unittest
import numpy as np
from indicators.trend.MACD import MovingAverageConvergenceDivergence


class TestMovingAverageConvergenceDivergence(unittest.TestCase):
    def test_calculate(self):
        # Test with simple price data
        price_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                               20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
                               30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0])

        # Calculate MACD with default parameters
        result = MovingAverageConvergenceDivergence.calculate(price_data)

        # Check if result contains all expected keys
        self.assertIn('MACD', result)
        self.assertIn('Signal Line', result)
        self.assertIn('Histogram', result)

        # Check if the output arrays have the correct length
        self.assertEqual(len(result['MACD']), len(price_data))
        self.assertEqual(len(result['Signal Line']), len(price_data))
        self.assertEqual(len(result['Histogram']), len(price_data))

        # Test with custom periods
        custom_result = MovingAverageConvergenceDivergence.calculate(price_data,
                                                                     fast_period=5,
                                                                     slow_period=10,
                                                                     signal_period=3)

        # Basic validation that values are different with different parameters
        self.assertFalse(np.array_equal(result['MACD'], custom_result['MACD']))

        # Test error case - data too short
        short_data = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            MovingAverageConvergenceDivergence.calculate(short_data)

        # Test with real-world data and verify some known values
        # These expected values should be calculated separately to ensure correctness
        known_prices = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 109.0, 108.0,
                                 110.0, 112.0, 111.0, 113.0, 115.0, 117.0, 118.0, 120.0,
                                 119.0, 121.0, 123.0, 125.0, 124.0, 126.0, 128.0, 127.0,
                                 129.0, 130.0, 132.0, 133.0, 135.0, 137.0])

        macd_result = MovingAverageConvergenceDivergence.calculate(known_prices)

        # Verify MACD is not all NaN
        self.assertTrue(np.any(~np.isnan(macd_result['MACD'])))

        # Test increasing trend should eventually show positive MACD values
        self.assertTrue(np.any(macd_result['MACD'][20:] > 0))