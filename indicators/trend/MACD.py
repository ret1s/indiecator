from indicators.trend.MA import MovingAverage
from typing import Optional, Union, Any
import numpy as np
from numpy import ndarray, dtype

class MovingAverageConvergenceDivergence:
    """
    A class that implements the Moving Average Convergence Divergence (MACD) indicator for technical analysis.
    """

    @staticmethod
    def calculate(data: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[dict[str, Union[ndarray[Any, dtype[Any]], ndarray]]]:
        """
        Calculate MACD using the formula:
        MACD = EMA(fast_period) - EMA(slow_period)
        Signal Line = EMA(MACD, signal_period)
        Histogram = MACD - Signal Line
        """
        if len(data) < max(fast_period, slow_period, signal_period):
            raise ValueError("Data length must be greater than the maximum of fast, slow, and signal periods")

        fast_ema = MovingAverage.ema(data, fast_period)
        slow_ema = MovingAverage.ema(data, slow_period)
        macd_line = fast_ema - slow_ema
        signal_line = MovingAverage.ema(macd_line, signal_period)
        histogram = macd_line - signal_line

        return {
            'MACD': macd_line,
            'Signal Line': signal_line,
            'Histogram': histogram
        }
