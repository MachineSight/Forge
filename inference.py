from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crusher_lstm_autoencoder.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"
CONFIG_PATH = BASE_DIR / "config.pkl"


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]


class Decoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        decoded, _ = self.lstm(latent)
        return self.output_layer(decoded)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, sequence_length: int):
        super().__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = Decoder(hidden_size=hidden_size, output_size=input_size, sequence_length=sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


@dataclass(frozen=True)
class InferenceResult:
    score: float
    health_state: str
    status_label: str
    threshold: float
    reconstruction_error: float
    window_index: int
    total_windows: int
    feature_contributions: pd.DataFrame
    top_sensor: str
    anomaly_hint: str
    scored_windows: pd.DataFrame
    preview: pd.DataFrame


def load_artifacts() -> tuple[LSTMAutoencoder, object, dict]:
    config = joblib.load(CONFIG_PATH)
    scaler = joblib.load(SCALER_PATH)

    input_size = int(config.get("input_size", 5))
    hidden_size = int(config.get("hidden_size", 32))
    sequence_length = int(config.get("sequence_length", 60))

    model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size, sequence_length=sequence_length)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, config


def _normalize_columns(columns: Iterable[str]) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in columns}


def prepare_dataframe(dataframe: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    normalized = _normalize_columns(dataframe.columns)
    resolved = {}
    missing = []

    for feature in feature_names:
        source_name = normalized.get(feature.lower())
        if source_name is None:
            missing.append(feature)
        else:
            resolved[feature] = source_name

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    prepared = dataframe.copy()
    prepared = prepared[[resolved[feature] for feature in feature_names]]
    prepared.columns = feature_names
    prepared = prepared.apply(pd.to_numeric, errors="coerce")
    prepared = prepared.interpolate(limit_direction="both").ffill().bfill()
    if prepared.isna().any().any():
        raise ValueError("Input file still contains non-numeric or missing values after cleaning.")
    return prepared


def create_windows(values: np.ndarray, sequence_length: int) -> np.ndarray:
    if len(values) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} rows to score the model, received {len(values)}."
        )

    return np.stack([values[start : start + sequence_length] for start in range(len(values) - sequence_length + 1)])


def score_health(reconstruction_error: float, threshold: float) -> float:
    return float(100 / (1 + reconstruction_error / threshold))


def classify_health(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Healthy", "healthy"
    if score >= 50:
        return "Warning", "warning"
    return "Critical", "critical"


def interpret_score(score: float) -> str:
    if score >= 80:
        return "The crusher is operating in a stable range. Continue routine monitoring."
    if score >= 50:
        return "The crusher shows early degradation signals. Inspect the highlighted sensors soon."
    return "The crusher is in a critical condition. Plan intervention immediately."


def sensor_anomaly_hint(sensor: str) -> str:
    hints = {
        "bearing_temperature": "Bearing is overheating",
        "motor_current": "Motor is overloaded",
        "vibration": "Mechanical imbalance is likely",
        "rpm": "Rotational speed is unstable",
        "lubrication_pressure": "Lubrication is insufficient",
    }
    return hints.get(sensor, f"{sensor.replace('_', ' ')} is showing abnormal behaviour")


def infer_from_dataframe(dataframe: pd.DataFrame) -> InferenceResult:
    model, scaler, config = load_artifacts()
    feature_names = list(config["feature_names"])
    sequence_length = int(config["sequence_length"])
    threshold = float(config["threshold"])

    cleaned = prepare_dataframe(dataframe, feature_names)
    values = cleaned[feature_names].values.astype(np.float32)
    scaled = scaler.transform(values)
    windows = create_windows(scaled, sequence_length)

    tensor = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        reconstruction = model(tensor)

    squared_error = (tensor - reconstruction) ** 2
    window_errors = squared_error.mean(dim=(1, 2)).cpu().numpy()
    window_scores = np.array([score_health(error, threshold) for error in window_errors], dtype=np.float32)

    latest_window_index = len(window_errors) - 1
    latest_error = float(window_errors[latest_window_index])
    latest_score = float(window_scores[latest_window_index])
    health_state, status_label = classify_health(latest_score)

    latest_window = tensor[latest_window_index : latest_window_index + 1].detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    latest_reconstruction = model(latest_window)
    latest_loss = torch.mean((latest_reconstruction - latest_window) ** 2)
    latest_loss.backward()

    gradients = latest_window.grad.detach().cpu().numpy()[0]
    latest_inputs = latest_window.detach().cpu().numpy()[0]
    gradient_scores = np.mean(np.abs(gradients * latest_inputs), axis=0)

    contribution_total = float(gradient_scores.sum())
    if contribution_total > 0:
        contribution_percentages = 100.0 * gradient_scores / contribution_total
    else:
        contribution_percentages = np.zeros_like(gradient_scores)

    top_sensor_index = int(np.argmax(contribution_percentages))
    top_sensor = feature_names[top_sensor_index]
    anomaly_hint = sensor_anomaly_hint(top_sensor)

    feature_contributions = pd.DataFrame(
        {
            "sensor": feature_names,
            "contribution_percent": contribution_percentages,
            "gradient_score": gradient_scores,
        }
    ).sort_values("contribution_percent", ascending=False, ignore_index=True)

    scored_windows = pd.DataFrame(
        {
            "window": np.arange(1, len(window_errors) + 1),
            "reconstruction_error": window_errors,
            "health_score": window_scores,
        }
    )

    return InferenceResult(
        score=latest_score,
        health_state=health_state,
        status_label=status_label,
        threshold=threshold,
        reconstruction_error=latest_error,
        window_index=latest_window_index + 1,
        total_windows=len(window_errors),
        feature_contributions=feature_contributions,
        top_sensor=top_sensor,
        anomaly_hint=anomaly_hint,
        scored_windows=scored_windows,
        preview=cleaned.tail(10).reset_index(drop=True),
    )
