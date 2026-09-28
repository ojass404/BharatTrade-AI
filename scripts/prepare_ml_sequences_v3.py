from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.src.dataset_preparation import (
    fit_training_scaler,
)
from ml_pipeline.src.dataset_preparation_v3 import (
    prepare_stock_sequences_v3,
)
from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


FEATURE_DIRECTORY = Path(
    "ml_pipeline/data/processed/features_v3"
)

SEQUENCE_DIRECTORY = Path(
    "ml_pipeline/data/processed/sequences_v3"
)

ARTIFACT_DIRECTORY = Path(
    "ml_pipeline/artifacts/models/v3"
)

SCALER_PATH = ARTIFACT_DIRECTORY / "feature_scaler_v3.joblib"
METADATA_PATH = ARTIFACT_DIRECTORY / "sequence_metadata_v3.json"

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def main() -> None:
    feature_files = sorted(
        FEATURE_DIRECTORY.glob("*_features_v3.csv")
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No V3 feature files found in "
            f"{FEATURE_DIRECTORY}."
        )

    frames: dict[str, pd.DataFrame] = {}

    print("Loading V3 feature datasets...")

    for path in feature_files:
        symbol = path.stem.replace(
            "_features_v3",
            "",
        ).upper()

        frame = pd.read_csv(path)
        frames[symbol] = frame

        print(
            f"  {symbol:<12} rows={len(frame):,}"
        )

    print("\nFitting scaler using training rows only...")

    scaler = fit_training_scaler(
        frames=list(frames.values()),
        feature_columns=FEATURE_COLUMNS,
        train_ratio=TRAIN_RATIO,
    )

    SEQUENCE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(scaler, SCALER_PATH)

    metadata: dict[str, object] = {
        "version": "v3",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "target_horizon_candles": 4,
        "target_threshold": 0.004,
        "same_session_target": True,
        "stocks": {},
    }

    print("\nPreparing leakage-safe V3 sequences...")

    for symbol, frame in frames.items():
        prepared = prepare_stock_sequences_v3(
            frame=frame,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
        )

        output_path = (
            SEQUENCE_DIRECTORY
            / f"{symbol.lower()}_sequences_v3.npz"
        )

        np.savez_compressed(
            output_path,
            x_train=prepared["x_train"],
            y_train=prepared["y_train"],
            train_times=prepared["train_times"],
            train_target_times=prepared[
                "train_target_times"
            ],
            x_validation=prepared["x_validation"],
            y_validation=prepared["y_validation"],
            validation_times=prepared[
                "validation_times"
            ],
            validation_target_times=prepared[
                "validation_target_times"
            ],
            x_test=prepared["x_test"],
            y_test=prepared["y_test"],
            test_times=prepared["test_times"],
            test_target_times=prepared[
                "test_target_times"
            ],
        )

        train_count = len(prepared["y_train"])
        validation_count = len(
            prepared["y_validation"]
        )
        test_count = len(prepared["y_test"])

        metadata["stocks"][symbol] = {
            "train_samples": train_count,
            "validation_samples": validation_count,
            "test_samples": test_count,
            "validation_start_time": str(
                prepared["validation_start_time"]
            ),
            "test_start_time": str(
                prepared["test_start_time"]
            ),
        }

        print(
            f"{symbol:<12} "
            f"train={train_count:>6,} | "
            f"validation={validation_count:>5,} | "
            f"test={test_count:>5,}"
        )

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nV3 sequence generation complete")
    print(f"Scaler  : {SCALER_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Output  : {SEQUENCE_DIRECTORY}")


if __name__ == "__main__":
    main()