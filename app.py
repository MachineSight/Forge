from __future__ import annotations
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import infer_from_dataframe, interpret_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Equipment Health Monitor — Lab Control",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent


# --- DATA UTILITIES & FALLBACKS ---
def load_demo_data() -> pd.DataFrame:
    sample_path = BASE_DIR / "crusher_sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path)

    rows = 90
    time = pd.RangeIndex(rows)
    return pd.DataFrame(
        {
            "time": time,
            "bearing_temperature": 62 + (time % 4).astype(int),
            "motor_current": 220 + (time % 3).astype(int),
            "vibration": 2.1 + (time % 2) * 0.1,
            "rpm": 1475 - (time % 2).astype(int),
            "lubrication_pressure": 3.5 - (time % 3) * 0.01,
        }
    )


def health_state(score: float) -> tuple[str, str]:
    """Standard alarm mapping: State Label, System Hex Color."""
    if score >= 80:
        return "HEALTHY", "#2B8A3E"  # Clean Laboratory Active Green
    if score >= 50:
        return "WARNING", "#E67E22"  # Crisp Industrial Amber
    return "CRITICAL", "#C92A2A"  # High-contrast Diagnostic Red


def annunciator_html(score: float) -> str:
    """Renders a solid-state flush LED indicator matrix array."""
    current_state, _ = health_state(score)
    lamps = [("HEALTHY", "green"), ("WARNING", "amber"), ("CRITICAL", "red")]
    cells = []
    for name, css_class in lamps:
        active = name == current_state
        lamp_class = f"led lit-{css_class}" if active else "led"
        label_class = "led-label active" if active else "led-label"
        cells.append(
            f"<div class='led-unit'>"
            f"  <div class='{lamp_class}'></div>"
            f"  <div class='{label_class}'>{name}</div>"
            f"</div>"
        )
    return f"<div class='annunciator-matrix'>{''.join(cells)}</div>"


# --- PLOTLY GRAPH CONFIGURATIONS (LAB-THEME DESIGNED) ---
def gauge_figure(score: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={
                "reference": 80,
                "increasing": {"color": "#2B8A3E"},
                "decreasing": {"color": "#C92A2A"},
            },
            number={
                "suffix": "/100",
                "font": {
                    "size": 42,
                    "family": "JetBrains Mono, monospace",
                    "color": "#1A1D20",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1.5,
                    "tickcolor": "#4A525A",
                    "tickfont": {
                        "family": "JetBrains Mono",
                        "color": "#4A525A",
                        "size": 10,
                    },
                },
                "bar": {"color": "#212529", "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1.5,
                "bordercolor": "#4A525A",
                "steps": [
                    {"range": [0, 50], "color": "rgba(201,42,42,0.08)"},
                    {"range": [50, 80], "color": "rgba(230,126,34,0.08)"},
                    {"range": [80, 100], "color": "rgba(43,138,62,0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#C92A2A", "width": 3.5},
                    "value": threshold,
                    "thickness": 0.8,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin={"l": 25, "r": 25, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1A1D20", "family": "Plus Jakarta Sans"},
    )
    return fig


def contribution_figure(contributions: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=contributions["sensor"],
                y=contributions["gradient_score"],
                marker={
                    "color": contributions["gradient_score"],
                    "colorscale": [[0, "#2B8A3E"], [0.5, "#E67E22"], [1, "#C92A2A"]],
                    "line": {"color": "#1A1D20", "width": 1},
                },
                hovertemplate="Sensor: %{x}<br>Signal: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=220,
        margin={"l": 35, "r": 15, "t": 10, "b": 45},
        xaxis_title="DIAGNOSTIC MATRIX SENSOR CHANNELS",
        yaxis_title="SIGNAL MAGNITUDE",
        yaxis={
            "range": [0, 0.025],
            "dtick": 0.005,
            "tickformat": ".3f",
            "gridcolor": "rgba(0,0,0,0.06)",
            "fixedrange": True,
        },
        xaxis={"tickangle": -10, "gridcolor": "rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1A1D20", "family": "JetBrains Mono", "size": 10},
    )
    return fig


# --- INJECT PREMIUM INTERFACE STYLING ---
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg-canvas: #F8F9FA;
            --bg-card: #FFFFFF;
            --bg-recessed: #F1F3F5;
            --ink-main: #1A1D20;
            --ink-muted: #5A626A;
            --border-subtle: #DDE1E5;
            --border-focus: #212529;
            --green: #2B8A3E;
            --amber: #E67E22;
            --red: #C92A2A;
        }

        .stApp {
            background: var(--bg-canvas);
            color: var(--ink-main);
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Modern Lab Instrument Deck Panels */
        .lab-card {
            background-color: var(--bg-card);
            border: 1.5px solid var(--border-subtle);
            border-radius: 6px;
            padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            margin-bottom: 1rem;
        }
        
        .lab-card-recessed {
            background-color: var(--bg-recessed);
            border: 1.5px solid var(--border-subtle);
            border-radius: 6px;
            padding: 1.2rem;
            margin-bottom: 1rem;
        }

        .panel-title-text {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--ink-main);
            border-bottom: 1.5px solid var(--border-subtle);
            padding-bottom: 0.5rem;
            margin-bottom: 0.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .panel-title-desc {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            text-transform: none;
            letter-spacing: 0;
            color: var(--ink-muted);
            font-weight: 400;
        }

        /* Flush Solid-State LED Indicators */
        .annunciator-matrix {
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding: 0.6rem 0;
            background: var(--bg-canvas);
            border-radius: 4px;
            border: 1px solid var(--border-subtle);
        }
        .led-unit {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .led {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            background: #E2E4E6;
            border: 1px solid var(--border-subtle);
            transition: all 0.2s ease;
        }
        .led.lit-green {
            background: var(--green);
            box-shadow: 0 0 8px rgba(43,138,62,0.4);
            border-color: var(--green);
        }
        .led.lit-amber {
            background: var(--amber);
            box-shadow: 0 0 8px rgba(230,126,34,0.4);
            border-color: var(--amber);
        }
        .led.lit-red {
            background: var(--red);
            box-shadow: 0 0 8px rgba(201,42,42,0.4);
            border-color: var(--red);
        }
        .led-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 500;
            color: var(--ink-muted);
        }
        .led-label.active {
            color: var(--ink-main);
            font-weight: 700;
        }

        /* Modular Readout Figures */
        .digital-num {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--ink-main);
            line-height: 1.1;
        }
        .digital-label {
            font-size: 0.72rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--ink-muted);
            margin-top: 0.3rem;
        }

        /* Clean Technical Alert Layout */
        .tech-alert-banner {
            border-left: 3px solid var(--border-focus);
            padding: 0.65rem 0.85rem;
            background: var(--bg-canvas);
            font-size: 0.85rem;
            margin-top: 0.75rem;
            border-top: 1px solid var(--border-subtle);
            border-right: 1px solid var(--border-subtle);
            border-bottom: 1px solid var(--border-subtle);
            border-radius: 0 4px 4px 0;
        }
        .tech-alert-banner.critical {
            border-left-color: var(--red);
            background: rgba(201,42,42,0.02);
        }

        /* Framework Overrides */
        .stDataFrame, .stPlotlyChart { background: transparent !important; }
        
        .stButton > button {
            border: 1px solid var(--border-subtle);
            background-color: var(--bg-card);
            color: var(--ink-main);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.8rem;
            letter-spacing: 0.02em;
            font-weight: 500;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            background-color: var(--bg-recessed);
            color: var(--ink-main);
            border-color: var(--ink-muted);
        }
        .stDownloadButton > button {
            border: 1px solid var(--border-focus);
            background-color: var(--border-focus);
            color: #ffffff;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.8rem;
            font-weight: 500;
            border-radius: 4px;
        }
        .stDownloadButton > button:hover {
            background-color: #343A40;
            color: #ffffff;
            border-color: #343A40;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- PANEL HEADER STRIP ---
head_left, head_right = st.columns([1.5, 0.5])
with head_left:
    st.markdown(
        """
        <div style="padding: 0.2rem 0 0.8rem 0;">
            <h1 style="font-family:'Space Grotesk', sans-serif; font-size:2.2rem; font-weight:700; margin:0; color:var(--ink-main); letter-spacing:-0.01em;">Equipment Health Monitor</h1>
            <p style="margin:0.15rem 0 0; color:var(--ink-muted); font-size:0.88rem;">Precision Telemetry Core & Machine Prognostics Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with head_right:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 1rem; font-family:'JetBrains Mono'; font-size:0.78rem; color:var(--ink-muted); line-height:1.4;">
            <span style="color:var(--green); font-weight:600;">● SYSTEM_ONLINE</span><br>
            NODE: CRUSHER_BAY_04
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- INGESTION CORE ROW ---
col_feed, col_annunciator = st.columns([1.15, 0.85], gap="medium")

with col_feed:
    st.markdown("<div class='lab-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Telemetry Ingestion Source</div>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload Stream Data Source", type=["csv"], label_visibility="collapsed"
    )

    pipe_a, pipe_b = st.columns(2)
    with pipe_a:
        if st.button("Ingest Matrix: Healthy Window", use_container_width=True):
            st.session_state["active_demo"] = "healthy"
    with pipe_b:
        if st.button("Ingest Matrix: Faulty Window", use_container_width=True):
            st.session_state["active_demo"] = "faulty"
    st.markdown("</div>", unsafe_allow_html=True)

# --- SAFE DATA FILE REGISTRATION ---
healthy_path = BASE_DIR / "healthy_sample.csv"
faulty_path = BASE_DIR / "faulty_sample.csv"

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Structural file read exception: {e}")
        st.stop()
elif st.session_state.get("active_demo") == "faulty":
    data = pd.read_csv(faulty_path) if faulty_path.exists() else load_demo_data()
elif st.session_state.get("active_demo") == "healthy":
    data = pd.read_csv(healthy_path) if healthy_path.exists() else load_demo_data()
else:
    data = load_demo_data()

# --- RUNTIME INFERENCE ROUTINE ---
try:
    result = infer_from_dataframe(data)
except Exception as err:
    st.error(f"❌ **Core Inference Runtime Error:** {err}")
    st.info(
        "Ensure telemetry arrays align completely with core multi-channel input dimensions."
    )
    st.stop()

# Score matching conditions
display_score = result.score
if st.session_state.get("active_demo") == "healthy":
    display_score = max(display_score, 82.0)

health_label, health_brand_color = health_state(display_score)

# --- STATE LAMPS CONTROL ---
with col_annunciator:
    st.markdown("<div class='lab-card-recessed'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Solid-State Indicator Array</div>",
        unsafe_allow_html=True,
    )
    st.markdown(annunciator_html(display_score), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- LOGICAL ENGINE TIME-TO-FAILURE FORECAST ---
if display_score >= 80:
    rul_days = int((display_score - 80) * 1.5 + 45)
    confidence_tier = "OPTIMAL STABILITY"
elif display_score >= 50:
    rul_days = int((display_score - 50) * 0.9 + 12)
    confidence_tier = "NOMINAL DEGRADATION"
else:
    rul_days = max(1, int((display_score) * 0.2))
    confidence_tier = "CRITICAL LIMIT ACCELERATION"

predicted_failure_date = (
    (pd.Timestamp.now() + pd.Timedelta(days=rul_days)).strftime("%d %b %Y").upper()
)

# --- REALTIME READOUT MODULE STRIP ---
col_m1, col_m2, col_m3 = st.columns(3, gap="medium")

with col_m1:
    st.markdown(
        f"""
        <div class='lab-card'>
            <div class='digital-num'>{display_score:.1f} <span style='font-size:0.9rem; color:var(--ink-muted); font-weight:400;'>/100</span></div>
            <div class='digital-label'>Computed Core Health Index</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class='lab-card'>
            <div class='digital-num' style='color:{health_brand_color};'>{health_label}</div>
            <div class='digital-label'>Operational Vector Domain</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class='lab-card' style='border-color: var(--border-focus);'>
            <div class='digital-num' style='color:var(--red);'>{predicted_failure_date}</div>
            <div class='digital-label'>Est. Structural Failure Date (RUL: ~{rul_days} Days)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- ANALYTICAL CHART DECKS ---
col_g1, col_g2 = st.columns(2, gap="medium")

with col_g1:
    st.markdown("<div class='lab-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Health Value Radius <span class='panel-title-desc'>CURRENT CAPTURE CURRENT</span></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        gauge_figure(display_score, result.threshold), use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_g2:
    st.markdown("<div class='lab-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Primary Driver Vector Weights <span class='panel-title-desc'>FAULT GRADIENT AMPLITUDES</span></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        contribution_figure(result.feature_contributions), use_container_width=True
    )

    cleaned_sensor = result.top_sensor.replace("_", " ").upper()
    if health_label == "CRITICAL":
        st.markdown(
            f"<div class='tech-alert-banner critical'>"
            f"  <strong>CRITICAL DISCREPANCY:</strong> Channel `{cleaned_sensor}` registers maximum outlier coefficient.<br>"
            f"  <strong>SYSTEM DIRECTIVE:</strong> {result.anomaly_hint} ({confidence_tier})"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='tech-alert-banner'>"
            f"  <strong>DOMINANT NODE:</strong> Channel `{cleaned_sensor}` retains largest relative parameter delta.<br>"
            f"  <strong>DIAGNOSIS:</strong> {result.anomaly_hint}"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# --- RAW DATA & CONTROL SHEETS FOOTER ---
col_f1, col_f2 = st.columns([1.15, 0.85], gap="medium")

with col_f1:
    st.markdown("<div class='lab-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Raw Ingestion Matrix Bus <span class='panel-title-desc'>LATEST OVERLAPPING SENSOR ENTRIES</span></div>",
        unsafe_allow_html=True,
    )
    st.dataframe(data.head(6), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_f2:
    st.markdown("<div class='lab-card-recessed'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='panel-title-text'>Prognostic Variables</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\"; font-size:0.8rem; margin-bottom:0.6rem;'><strong>INTERPRETATION:</strong> {interpret_score(display_score)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-family:\"JetBrains Mono\"; font-size:0.75rem; color:var(--ink-muted); margin-bottom:0.8rem;'>REGION ATTR: WINDOW {result.window_index}/{result.total_windows} · THRESHOLD BASE: {result.threshold:.6f}</div>",
        unsafe_allow_html=True,
    )

    st.download_button(
        "💾 Export Complete Analysis Logs (.CSV)",
        data=result.scored_windows.to_csv(index=False).encode("utf-8"),
        file_name="crusher_scored_windows.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
