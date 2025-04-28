import numpy as np
import pandas as pd
from typing import Union, List

class MovingAverage:
    """
    A class that implements various moving average techniques for technical analysis.
    """

    @staticmethod
    def sma(data: Union[List[float], np.ndarray, pd.Series], window: int) -> np.ndarray:
        """
        Calculate the Simple Moving Average (SMA)
        """
        if window <= 0:
            raise ValueError("Period must be a positive integer")

        data = np.array(data)
        return pd.Series(data).rolling(window).mean().to_numpy()

    @staticmethod
    def ema(data: Union[List[float], np.ndarray, pd.Series], window: int,
            smoothing: float = 2.0) -> np.ndarray:
        """Calculate Exponential Moving Average (EMA)"""
        if window <= 0:
            raise ValueError("Window period must be positive")

        data = np.array(data)
        alpha = smoothing / (1 + window)
        return pd.Series(data).ewm(alpha=alpha, adjust=False).mean().to_numpy()

    @staticmethod
    def wma(data: Union[List[float], np.ndarray, pd.Series], window: int) -> np.ndarray:
        """Calculate Weighted Moving Average (WMA)"""
        if window <= 0:
            raise ValueError("Window period must be positive")

        data = np.array(data)
        weights = np.arange(1, window + 1)
        result = np.full_like(data, np.nan, dtype=float)

        for i in range(window - 1, len(data)):
            result[i] = np.sum(data[i - window + 1:i + 1] * weights) / weights.sum()

        return result

    @staticmethod
    def cma(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """Calculate Cumulative Moving Average (CMA)"""
        data = np.array(data)
        result = np.full_like(data, np.nan, dtype=float)

        for i in range(len(data)):
            result[i] = np.mean(data[:i + 1])

        return result

    @staticmethod
    def crossover(ma1: np.ndarray, ma2: np.ndarray) -> List[int]:
        """
        Identify crossover points between two moving averages
        Returns indices where crossovers occur (1 for bullish, -1 for bearish, 0 for no crossover)
        """
        if len(ma1) != len(ma2):
            raise ValueError("Moving averages must have the same length")

        crossovers = [0]  # Start with no crossover

        for i in range(1, len(ma1)):
            if np.isnan(ma1[i - 1]) or np.isnan(ma2[i - 1]) or np.isnan(ma1[i]) or np.isnan(ma2[i]):
                crossovers.append(0)
            elif ma1[i - 1] <= ma2[i - 1] and ma1[i] > ma2[i]:
                crossovers.append(1)  # Bullish crossover
            elif ma1[i - 1] >= ma2[i - 1] and ma1[i] < ma2[i]:
                crossovers.append(-1)  # Bearish crossover
            else:
                crossovers.append(0)  # No crossover

        return crossovers