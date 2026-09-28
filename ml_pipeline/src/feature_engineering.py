from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


FEATURE_COLUMNS = [
    "return_1",
    "return_4",
    "log_return",
    "sma_10_ratio",
    "sma_20_ratio",
    "ema_10_ratio",
    "ema_20_ratio",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "volatility_20",
    "volume_change",
    "volume_ratio_20",
    "high_low_range",
    "open_close_range",
    "atr_14_ratio",
    "hour_sin",
    "hour_cos",
]


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    price_change = close.diff()

    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_loss = losses.rolling(
        window=period,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100,
    )

    rsi = rsi.mask(
        (average_gain == 0) & (average_loss > 0),
        0,
    )

    return rsi


def calculate_atr(
    frame: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    previous_close = frame["close"].shift(1)

    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        window=period,
        min_periods=period,
    ).mean()


def create_features(
    frame: pd.DataFrame,
    target_threshold: float = 0.0015,
    target_horizon: int = 1,
    same_session_target: bool = False,
) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if target_threshold <= 0:
        raise ValueError(
            "target_threshold must be positive."
        )

    if (
        not isinstance(target_horizon, int)
        or isinstance(target_horizon, bool)
        or target_horizon < 1
    ):
        raise ValueError(
            "target_horizon must be a positive integer."
        )

    data = frame.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data = (
        data.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data["return_1"] = data["close"].pct_change(
        periods=1,
        fill_method=None,
    )

    data["return_4"] = data["close"].pct_change(
        periods=4,
        fill_method=None,
    )

    data["log_return"] = np.log(
        data["close"] / data["close"].shift(1)
    )

    sma_10 = data["close"].rolling(
        window=10,
        min_periods=10,
    ).mean()

    sma_20 = data["close"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    ema_10 = data["close"].ewm(
        span=10,
        adjust=False,
    ).mean()

    ema_20 = data["close"].ewm(
        span=20,
        adjust=False,
    ).mean()

    data["sma_10_ratio"] = data["close"] / sma_10 - 1
    data["sma_20_ratio"] = data["close"] / sma_20 - 1
    data["ema_10_ratio"] = data["close"] / ema_10 - 1
    data["ema_20_ratio"] = data["close"] / ema_20 - 1

    data["rsi_14"] = calculate_rsi(data["close"])

    ema_12 = data["close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    ema_26 = data["close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    data["macd"] = ema_12 - ema_26

    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["macd_histogram"] = (
        data["macd"] - data["macd_signal"]
    )

    data["volatility_20"] = data["return_1"].rolling(
        window=20,
        min_periods=20,
    ).std()

    data["volume_change"] = data["volume"].pct_change(
        periods=1,
        fill_method=None,
    )

    volume_average_20 = data["volume"].rolling(
        window=20,
        min_periods=20,
    ).mean()

    data["volume_ratio_20"] = (
        data["volume"] / volume_average_20.replace(0, np.nan)
    )

    data["high_low_range"] = (
        data["high"] - data["low"]
    ) / data["close"]

    data["open_close_range"] = (
        data["close"] - data["open"]
    ) / data["open"]

    atr_14 = calculate_atr(data)

    data["atr_14_ratio"] = atr_14 / data["close"]

    india_timestamp = data["timestamp"].dt.tz_convert(
        "Asia/Kolkata"
    )

    decimal_hour = (
        india_timestamp.dt.hour
        + india_timestamp.dt.minute / 60
    )

    data["hour_sin"] = np.sin(
        2 * np.pi * decimal_hour / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * decimal_hour / 24
    )

    if same_session_target:
        trading_date = india_timestamp.dt.date

        grouped_data = data.groupby(
            trading_date,
            sort=False,
        )

        future_timestamp = grouped_data[
            "timestamp"
        ].shift(-target_horizon)

        future_close = grouped_data[
            "close"
        ].shift(-target_horizon)

    else:
        future_timestamp = data[
            "timestamp"
        ].shift(-target_horizon)

        future_close = data[
            "close"
        ].shift(-target_horizon)

    data["target_time"] = future_timestamp

    data["target_return"] = (
        future_close / data["close"] - 1
    )

    data["target_class"] = np.select(
        [
            data["target_return"] < -target_threshold,
            data["target_return"] > target_threshold,
        ],
        [
            0,  # SELL
            2,  # BUY
        ],
        default=1,  # HOLD
    )

    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    required_output_columns = (
        FEATURE_COLUMNS
        + [
            "target_time",
            "target_return",
            "target_class",
        ]
    )

    data = data.dropna(
        subset=required_output_columns
    ).reset_index(drop=True)

    data["target_class"] = data[
        "target_class"
    ].astype(int)

    return data