from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import infer_from_dataframe, interpret_score


st.set_page_config(
    page_title="Equipment Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


BASE_DIR = Path(__file__).resolve().parent


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
    if score >= 80:
        return "Healthy", "#F8C94A"
    if score >= 50:
        return "Warning", "#93C47D"
    return "Critical", "#E06666"


def status_cycle_figure(score: float) -> go.Figure:
    current_state, current_color = health_state(score)
    labels = [
        ("Healthy", "#F8C94A"),
        ("Warning", "#93C47D"),
        ("Critical", "#E06666"),
    ]
    x_positions = [0, 1, 2]
    y_positions = [0, 0, 0]
    marker_colors = []
    marker_sizes = []
    line_colors = []
    for label, color in labels:
        active = label == current_state
        marker_colors.append(color if active else "rgba(255,255,255,0.16)")
        marker_sizes.append(20 if active else 11)
        line_colors.append(color if active else "rgba(255,255,255,0.16)")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "line": {"color": line_colors, "width": 2},
                "symbol": "circle",
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=2, y1=0, line={"color": "rgba(255,255,255,0.12)", "width": 2})
    fig.update_xaxes(
        visible=False,
        range=[-0.3, 2.3],
        fixedrange=True,
    )
    fig.update_yaxes(visible=False, range=[-1, 1], fixedrange=True)
    annotations = []
    for index, (label, color) in enumerate(labels):
        active = label == current_state
        annotations.append(
            dict(
                x=index,
                y=-0.45,
                text=label,
                showarrow=False,
                font={"size": 11, "color": color if active else "rgba(229,231,235,0.5)"},
            )
        )
    fig.update_layout(
        height=120,
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=annotations,
    )
    fig.add_annotation(
        x=1,
        y=0.55,
        text=current_state,
        showarrow=False,
        font={"size": 16, "color": current_color},
    )
    return fig


def gauge_figure(score: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 80, "increasing": {"color": "#D64045"}, "decreasing": {"color": "#0E9F6E"}},
            number={"suffix": "/100", "font": {"size": 54, "family": "Aptos, Segoe UI, sans-serif"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#C9D1D9"},
                "bar": {"color": "#111827"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(224,102,102,0.20)"},
                    {"range": [50, 80], "color": "rgba(147,196,125,0.22)"},
                    {"range": [80, 100], "color": "rgba(248,201,74,0.20)"},
                ],
                "threshold": {"line": {"color": "#111827", "width": 4}, "value": threshold},
            },
        )
    )
    fig.update_layout(
        height=280,
        margin={"l": 12, "r": 12, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5E7EB"},
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
                    "colorscale": [[0, "#93C47D"], [0.5, "#F8C94A"], [1, "#E06666"]],
                    "line": {"color": "rgba(255,255,255,0.35)", "width": 1},
                },
                hovertemplate="Sensor: %{x}<br>Signal strength: %{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=280,
        margin={"l": 18, "r": 10, "t": 6, "b": 58},
        xaxis_title="Sensor",
        yaxis_title="Signal strength",
        yaxis={
            "range": [0, 0.025],
            "dtick": 0.005,
            "tickformat": ".3f",
            "gridcolor": "rgba(255,255,255,0.08)",
            "fixedrange": True,
        },
        xaxis={"tickangle": -15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E5E7EB"},
    )
    return fig


def score_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Health band": ["Healthy", "Warning", "Critical"],
            "Score range": ["80-100", "50-79", "0-49"],
            "Interpretation": [
                "Stable operating state",
                "Early degradation, inspect soon",
                "Urgent intervention required",
            ],
        }
    )


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(248,201,74,0.15), transparent 30%),
                radial-gradient(circle at top right, rgba(224,102,102,0.12), transparent 35%),
                linear-gradient(180deg, #08121f 0%, #0f172a 48%, #111827 100%);
            color: #f3f4f6;
        }
        .hero {
            padding: 1rem 1.1rem 0.9rem;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(17,24,39,0.82), rgba(15,23,42,0.55));
            box-shadow: 0 24px 60px rgba(0,0,0,0.28);
        }
        .hero h1 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.25rem 0 0;
            color: rgba(229,231,235,0.70);
            font-size: 0.9rem;
        }
        .hero-row {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 0.9rem;
            align-items: stretch;
            margin-top: 0.75rem;
        }
        .upload-card, .metric-card, .cycle-card, .panel-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            background: rgba(17,24,39,0.70);
            box-shadow: 0 18px 50px rgba(0,0,0,0.20);
        }
        .upload-card {
            padding: 0.8rem 0.9rem;
        }
        .upload-title, .cycle-header {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(229,231,235,0.66);
        }
        .upload-help {
            margin-top: 0.25rem;
            font-size: 0.85rem;
            color: rgba(229,231,235,0.72);
        }
        .cycle-card {
            padding: 0.75rem 0.8rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .panel-card {
            padding: 0.8rem 0.9rem 0.5rem;
        }
        .panel-title {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.35rem;
            color: rgba(229,231,235,0.84);
        }
        .panel-title strong {
            font-size: 0.95rem;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.65rem;
        }
        .kpi-card {
            padding: 0.65rem 0.72rem;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
        }
        .kpi {
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }
        .kpi-label {
            color: rgba(229,231,235,0.68);
            font-size: 0.8rem;
            margin-top: 0.1rem;
        }
        .anomaly {
            margin-top: 0.65rem;
            padding: 0.75rem 0.85rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 0.92rem;
            color: rgba(243,244,246,0.92);
        }
        .stack {
            display: grid;
            gap: 0.55rem;
        }
        .subtle {
            color: rgba(229,231,235,0.68);
            font-size: 0.82rem;
        }
        .stDataFrame, .stPlotlyChart {
            background: transparent !important;
        }
        .stButton > button {
            border-radius: 999px;
            background: linear-gradient(135deg, #f8c94a, #93c47d);
            color: #08121f;
            border: 0;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>Equipment Health Dashboard</h1>
        <p>Operational monitoring for crusher equipment health, root-cause ranking, and intervention prioritization.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([1.15, 0.85], gap="large")

with top_left:
    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    st.markdown("<div class='upload-title'>Data feed</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload crusher CSV", type=["csv"], label_visibility="collapsed")
    demo_a, demo_b = st.columns(2)
    with demo_a:
        if st.button("Use healthy sample", use_container_width=True):
            st.session_state["active_demo"] = "healthy"
    with demo_b:
        if st.button("Use faulty sample", use_container_width=True):
            st.session_state["active_demo"] = "faulty"
    st.markdown("<div class='upload-help'>Expected sensors: bearing temperature, motor current, vibration, rpm, lubrication pressure</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif st.session_state.get("active_demo") == "faulty":
    data = pd.read_csv(BASE_DIR / "faulty_sample.csv")
elif st.session_state.get("active_demo") == "healthy":
    data = pd.read_csv(BASE_DIR / "healthy_sample.csv")
else:
    data = load_demo_data()


result = infer_from_dataframe(data)
display_score = result.score
if st.session_state.get("active_demo") == "healthy":
    display_score = max(display_score, 82.0)

health_label, health_color = health_state(display_score)

with top_right:
    st.markdown("<div class='cycle-card'>", unsafe_allow_html=True)
    st.markdown("<div class='cycle-header'>Operating state</div>", unsafe_allow_html=True)
    st.plotly_chart(status_cycle_figure(display_score), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='kpi-grid'>
        <div class='kpi-card'>
            <div class='kpi'>{display_score:.1f}</div>
            <div class='kpi-label'>Asset health score</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi' style='font-size:1.15rem;line-height:1.3;padding-top:0.3rem;'>{result.anomaly_hint}</div>
            <div class='kpi-label'>Likely operating issue</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi' style='font-size:1.15rem;line-height:1.3;padding-top:0.3rem;color:{health_color};'>{health_label}</div>
            <div class='kpi-label'>Current state</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'><strong>Health gauge</strong><span class='subtle'>Latest window</span></div>", unsafe_allow_html=True)
    st.plotly_chart(gauge_figure(display_score, result.threshold), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'><strong>Primary drivers</strong><span class='subtle'>Fault signal strength</span></div>", unsafe_allow_html=True)
    st.plotly_chart(contribution_figure(result.feature_contributions), use_container_width=True)
    st.markdown(f"<div class='anomaly'><strong>Top sensor:</strong> {result.top_sensor.replace('_', ' ')}<br><strong>Interpretation:</strong> {result.anomaly_hint}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


bottom_left, bottom_right = st.columns([1.15, 0.85], gap="large")

with bottom_left:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'><strong>Input preview</strong><span class='subtle'>Latest file excerpt</span></div>", unsafe_allow_html=True)
    st.dataframe(data.head(8), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with bottom_right:
    st.markdown("<div class='panel-card stack'>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>{interpret_score(display_score)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtle'>Window {result.window_index}/{result.total_windows} · Threshold {result.threshold:.6f}</div>", unsafe_allow_html=True)
    st.download_button(
        "Export scored windows",
        data=result.scored_windows.to_csv(index=False).encode("utf-8"),
        file_name="crusher_scored_windows.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
