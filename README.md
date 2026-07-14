# Crusher Equipment Health Monitor

A **real-time anomaly detection dashboard** for industrial crusher machinery, powered by an **LSTM Autoencoder** deep learning model.

This project ingests multi-channel sensor telemetry (bearing temperature, motor current, vibration, RPM, lubrication pressure), runs inference through a trained PyTorch autoencoder, and surfaces a **Streamlit dashboard** with health scores, failure predictions, and per-sensor fault attribution.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Sensor CSV     │───▶│  Data Preprocess │───▶│  LSTM Autoencoder    │
│  (telemetry)    │     │  (normalize,     │     │  (PyTorch)           │
│                 │     │   window, align) │     │                      │
└─────────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                            |
                                                            ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Streamlit UI   │◀───│  Health Scoring  │◀───│  Reconstruction      │
│  Dashboard      │     │  + Gradient      │     │  Error + Sensitivity │
│                 │     │  Attribution     │     │  Analysis            │
└─────────────────┘     └──────────────────┘     └──────────────────────┘
```

### Components

| File | Purpose |
|---|---|
| `app.py` | **Streamlit frontend** — dashboard UI, data ingestion, LED annunciator, gauges, RUL forecast |
| `inference.py` | **Inference engine** — LSTM autoencoder model, data preprocessing, windowed scoring, gradient-based sensor attribution |
| `patch_scalar.py` | **Utility** — re-fits the `StandardScaler` on healthy baseline data (for scikit-learn version migrations) |
| `healthy_sample.csv` | Sample telemetry from normal crusher operation |
| `faulty_sample.csv` | Sample telemetry from a degrading/failing crusher |
| `config.pkl` | Model configuration (features, sequence length, threshold) |
| `scaler.pkl` | Fitted `StandardScaler` for input normalization |
| `crusher_lstm_autoencoder.pth` | Trained PyTorch LSTM autoencoder weights |

---

## Model: LSTM Autoencoder

The anomaly detector is an **unsupervised LSTM autoencoder** trained on healthy (normal) operating data.

### Architecture

```
Input: [batch, sequence_length=60, input_size=5]
           │
           ▼
    ┌──────────────┐
    │   Encoder    │  LSTM(5 → 32)
    │   (LSTM)     │  Returns last hidden state
    └──────┬───────┘
           │  latent vector (32-d)
           ▼
    ┌──────────────┐
    │   Decoder    │  LSTM(32 → 32) → Linear(32 → 5)
    │   (LSTM)     │  Repeats latent across 60 timesteps
    └──────┬───────┘
           │
           ▼
Output: [batch, 60, 5]  (reconstructed sequence)
```

### How it works

1. **Training phase:** The autoencoder learns to reconstruct normal sensor patterns with low error.
2. **Inference phase:** New sensor windows are passed through the model. Windows with **high reconstruction error** indicate anomalous behavior (the model can't reconstruct what it never learned).
3. **Health scoring:** Reconstruction error is converted to a 0–100 health score. Higher scores = healthier.
4. **Fault attribution:** Gradients of the reconstruction loss with respect to the input are computed via `backward()`, producing per-sensor contribution percentages that pinpoint the root cause.

### Health Score Interpretation

| Score | State | Action |
|---|---|---|
| 80–100 | **Healthy** | Continue routine monitoring |
| 50–79 | **Warning** | Inspect highlighted sensors soon |
| 0–49 | **Critical** | Plan immediate intervention |

### Sensor Channels

| Channel | Normal Range | Faulty Range |
|---|---|---|
| Bearing Temperature | ~62 °C | 65–75 °C |
| Motor Current | ~220 A | 224–239 A |
| Vibration | ~2.1 mm/s | 2.4–3.6 mm/s |
| RPM | ~1475 | 1466–1474 |
| Lubrication Pressure | ~3.5 bar | 3.25–3.45 bar |

---

## Dashboard Features

The Streamlit dashboard (`app.py`) provides:

- **Telemetry Ingestion** — Upload a CSV file or load a demo sample (healthy / faulty)
- **LED Annunciator Panel** — Solid-state indicator lamps (GREEN / AMBER / RED)
- **Health Gauge** — Plotly gauge with threshold markers
- **Fault Contribution Bars** — Per-sensor gradient amplitude chart
- **Remaining Useful Life (RUL)** — Estimated days to failure with projected date
- **Dominant Fault Diagnosis** — Highlights the most anomalous sensor with actionable hints
- **Raw Data Bus** — Preview of the ingested sensor matrix
- **CSV Export** — Download scored windows for offline analysis

---

## Setup & Usage

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd crusher-health-monitor

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
streamlit
plotly
pandas
numpy
torch
scikit-learn
joblib
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

### Preventing Streamlit Sleep (Keep-Alive Automation)

This repo includes a keep-alive script at `scripts/keep_streamlit_awake.py` and a scheduled GitHub Actions workflow at `.github/workflows/keep_streamlit_awake.yml`.

#### Option A: GitHub Actions (recommended)

The workflow runs every 10 minutes and sends a ping to:

```text
https://forge-team.streamlit.app/
```

To enable it:

1. Push this repo to GitHub.
2. In GitHub, open **Actions** and ensure workflows are enabled.
3. Manually run **Keep Streamlit Awake** once from **Actions** -> **Run workflow**.

#### Option B: Local/background runner

Run continuously from any machine that stays online:

```bash
python scripts/keep_streamlit_awake.py --url "https://forge-team.streamlit.app/"
```

Useful flags:

```bash
# One ping and exit
python scripts/keep_streamlit_awake.py --once

# Custom interval (seconds)
python scripts/keep_streamlit_awake.py --interval-seconds 480
```

Note: this is best-effort. If Streamlit Cloud platform policies change or scheduled jobs are delayed, occasional cold starts can still happen.

### Using the App

1. **Load demo data:** Click *"Ingest Matrix: Healthy Window"* or *"Ingest Matrix: Faulty Window"* to load sample telemetry.
2. **Upload your own:** Use the file uploader to ingest a CSV with matching sensor columns.
3. **Read the dashboard:** Check the health score, LED annunciator, gauge, and fault contribution chart.
4. **Export results:** Click *"Export Complete Analysis Logs (.CSV)"* to download per-window scores.

### Expected CSV Format

Your CSV must include the following columns (case-insensitive, order-independent):

```csv
time,bearing_temperature,motor_current,vibration,rpm,lubrication_pressure
0,62,220,2.1,1475,3.5
3,62,220,2.1,1475,3.5
...
```

The file must contain at least **60 rows** (the model's sequence length).

### Repairing the Scaler

If you encounter a `StandardScaler` compatibility error after upgrading scikit-learn:

```bash
python patch_scalar.py
```

This re-fits the scaler on `healthy_sample.csv` and overwrites `scaler.pkl`.

---

## Project Structure

```
├── app.py                          # Streamlit dashboard
├── inference.py                    # Inference engine + model definition
├── patch_scalar.py                 # Scaler repair utility
├── requirements.txt                # Python dependencies
├── config.pkl                      # Model configuration (pickle)
├── scaler.pkl                      # Fitted StandardScaler (pickle)
├── crusher_lstm_autoencoder.pth    # Trained model weights (PyTorch)
├── healthy_sample.csv              # Normal operating data sample
├── faulty_sample.csv               # Faulty operating data sample
└── README.md                       # This file
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **PyTorch** | LSTM autoencoder model definition & inference |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Gauges, bar charts, and data visualization |
| **scikit-learn** | `StandardScaler` for input normalization |
| **pandas / numpy** | Data manipulation and windowing |
| **joblib** | Model artifact serialization |

---

## License

MIT
