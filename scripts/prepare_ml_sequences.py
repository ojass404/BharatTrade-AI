import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_pipeline.src.dataset_preparation import (
    fit_training_scaler,
    prepare_stock_sequences,
)
from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


FEATURE_DIRECTORY = Path(
    "ml_pipeline/data/processed/features"
)

SEQUENCE_DIRECTORY = Path(
    "ml_pipeline/data/processed/sequences"
)

ARTIFACT_DIRECTORY = Path(
    "ml_pipeline/artifacts/preprocessing"
)

SEQUENCE_LENGTH = 60
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def class_distribution(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(
        labels,
        minlength=3,
    )

    return {
        "sell": int(counts[0]),
        "hold": int(counts[1]),
        "buy": int(counts[2]),
    }


def main() -> None:
    feature_files = sorted(
        FEATURE_DIRECTORY.glob("*_features.csv")
    )

    if not feature_files:
        raise SystemExit(
            "No feature CSV files were found. "
            "Run scripts.build_feature_dataset first."
        )

    frames: dict[str, pd.DataFrame] = {}

    for file_path in feature_files:
        symbol = file_path.stem.removesuffix(
            "_features"
        ).upper()

        frame = pd.read_csv(file_path)

        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
        )

        frame["target_time"] = pd.to_datetime(
            frame["target_time"],
            utc=True,
        )

        frame = frame.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        frames[symbol] = frame

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

    scaler_path = ARTIFACT_DIRECTORY / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    metadata: dict[str, Any] = {
        "sequence_length": SEQUENCE_LENGTH,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": (
            1 - TRAIN_RATIO - VALIDATION_RATIO
        ),
        "feature_columns": FEATURE_COLUMNS,
        "stocks": {},
    }

    print("ML SEQUENCE PREPARATION")
    print("=" * 70)
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Feature count  : {len(FEATURE_COLUMNS)}")

    for symbol, frame in frames.items():
        prepared = prepare_stock_sequences(
            frame=frame,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
            train_ratio=TRAIN_RATIO,
            validation_ratio=VALIDATION_RATIO,
        )

        boundaries = prepared["boundaries"]

        output_path = (
            SEQUENCE_DIRECTORY
            / f"{symbol.lower()}_sequences.npz"
        )

        np.savez_compressed(
            output_path,
            x_train=prepared["x_train"],
            y_train=prepared["y_train"],
            train_times=prepared["train_times"],
            x_validation=prepared["x_validation"],
            y_validation=prepared["y_validation"],
            validation_times=prepared[
                "validation_times"
            ],
            x_test=prepared["x_test"],
            y_test=prepared["y_test"],
            test_times=prepared["test_times"],
        )

        stock_metadata = {
            "total_feature_rows": len(frame),
            "train_boundary": boundaries.train_end,
            "validation_boundary": (
                boundaries.validation_end
            ),
            "train_sequences": len(
                prepared["y_train"]
            ),
            "validation_sequences": len(
                prepared["y_validation"]
            ),
            "test_sequences": len(
                prepared["y_test"]
            ),
            "train_distribution": class_distribution(
                prepared["y_train"]
            ),
            "validation_distribution": class_distribution(
                prepared["y_validation"]
            ),
            "test_distribution": class_distribution(
                prepared["y_test"]
            ),
        }

        metadata["stocks"][symbol] = stock_metadata

        print(
            f"{symbol:<12} "
            f"train={stock_metadata['train_sequences']:>6,} | "
            f"validation="
            f"{stock_metadata['validation_sequences']:>5,} | "
            f"test={stock_metadata['test_sequences']:>5,}"
        )

    metadata_path = (
        ARTIFACT_DIRECTORY
        / "sequence_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("-" * 70)
    print(f"Scaler saved   : {scaler_path}")
    print(f"Metadata saved : {metadata_path}")
    print(f"Sequences saved: {SEQUENCE_DIRECTORY}")


if __name__ == "__main__":
    main()