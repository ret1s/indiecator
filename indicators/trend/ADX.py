import numpy as np

class ADX:
    """
    A class that implements the Average Directional Index (ADX) for technical analysis.
    """

    @staticmethod
    def calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> dict:
        """
        Calculate the Average Directional Index (ADX).
        """
        if len(high) != len(low) or len(low) != len(close):
            raise ValueError("High, low, and close arrays must have the same length")

        # Calculate True Range (TR)
        tr = np.maximum.reduce([
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        ])

        # Calculate +DM and -DM
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)

        # Smooth TR, +DM, and -DM using exponential moving average
        tr_smooth = np.convolve(tr, np.ones(window) / window, mode='valid')
        plus_dm_smooth = np.convolve(plus_dm, np.ones(window) / window, mode='valid')
        minus_dm_smooth = np.convolve(minus_dm, np.ones(window) / window, mode='valid')

        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth

        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX
        adx = np.convolve(dx, np.ones(window) / window, mode='valid')

        # Pad results with NaN to match input length
        nan_padding = [np.nan] * (2 * window - 1)
        return {
            '+DI': np.concatenate((nan_padding, plus_di)),
            '-DI': np.concatenate((nan_padding, minus_di)),
            'ADX': np.concatenate((nan_padding, adx))
        }