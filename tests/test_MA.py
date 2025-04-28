import unittest
import numpy as np
import pandas as pd
from indicators.trend.MA import MovingAverage

class TestMovingAverage(unittest.TestCase):

    def test_sma_calculation(self):
        # Test with list data
        data = [1, 2, 3, 4, 5]
        result = MovingAverage.sma(data, window=3)
        # First two values as Nan, then 3-day averages
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result[2:], expected[2:])
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

    def test_sma_different_inputs(self):
        # Test with numpy array and pandas Series
        np_data = np.array([1, 2, 3, 4, 5])
        pd_data = pd.Series([1, 2, 3, 4, 5])

        np_result = MovingAverage.sma(np_data, window=2)
        pd_result = MovingAverage.sma(pd_data, window=2)

        expected = np.array([np.nan, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_equal(np_result[1:], expected[1:])
        np.testing.assert_array_equal(pd_result[1:], expected[1:])

    def test_sma_window_validation(self):
        data = [1, 2, 3, 4, 5]
        # Test error for invalid window sizes
        with self.assertRaises(ValueError):
            MovingAverage.sma(data, window=0)
        with self.assertRaises(ValueError):
            MovingAverage.sma(data, window=-1)

    def test_ema_calculation(self):
        """Test Exponential Moving Average (EMA) calculation"""
        data = [1, 2, 3, 4, 5]
        result = MovingAverage.ema(data, window=3)
        expected = pd.Series(data).ewm(span=3, adjust=False).mean().to_numpy()
        np.testing.assert_array_almost_equal(result, expected)

    def test_wma_calculation(self):
        """Test Weighted Moving Average (WMA) calculation"""
        data = [1, 2, 3, 4, 5]
        result = MovingAverage.wma(data, window=3)
        expected = np.array([np.nan, np.nan, 2.33333333, 3.33333333, 4.33333333])
        np.testing.assert_array_almost_equal(result[2:], expected[2:])
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isnan(result[1]))

    def test_cma_calculation(self):
        """Test Cumulative Moving Average (CMA) calculation"""
        data = [1, 2, 3, 4, 5]
        result = MovingAverage.cma(data)
        expected = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_window_for_ema_and_wma(self):
        """Test invalid window sizes for EMA and WMA"""
        data = [1, 2, 3, 4, 5]
        with self.assertRaises(ValueError):
            MovingAverage.ema(data, window=0)
        with self.assertRaises(ValueError):
            MovingAverage.wma(data, window=-1)

    def test_crossover_bullish_bearish(self):
        """Test detection of bullish and bearish crossovers"""
        ma1 = np.array([np.nan, 1, 2, 3, 4, 5])
        ma2 = np.array([np.nan, 2, 2, 2, 2, 2])
        result = MovingAverage.crossover(ma1, ma2)
        expected = [0, 0, 0, 1, 0, 0]  # Bullish crossover at index 3
        self.assertEqual(result, expected)

        ma1 = np.array([np.nan, 5, 4, 1, 2, 1])
        ma2 = np.array([np.nan, 2, 2, 2, 2, 2])
        result = MovingAverage.crossover(ma1, ma2)
        expected = [0, 0, 0, -1, 0, -1]  # Bearish crossover at index 3 and 5
        self.assertEqual(result, expected)

    def test_crossover_no_cross(self):
        """Test no crossovers when moving averages do not intersect"""
        ma1 = np.array([1, 2, 3, 4, 5])
        ma2 = np.array([5, 5, 5, 5, 5])
        result = MovingAverage.crossover(ma1, ma2)
        expected = [0, 0, 0, 0, 0]  # No crossovers
        self.assertEqual(result, expected)

    def test_crossover_nan_handling(self):
        """Test handling of NaN values in moving averages"""
        ma1 = np.array([np.nan, 1, 3, np.nan, 4, 5])
        ma2 = np.array([np.nan, 2, 2, np.nan, 2, 2])
        result = MovingAverage.crossover(ma1, ma2)
        expected = [0, 0, 1, 0, 0, 0]  # Bullish crossover at index 2
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()