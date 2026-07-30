"""Clean and polished Streamlit dashboard for face expression analytics."""

import base64
import glob
import json
import os
from collections import Counter
from datetime import date, datetime, time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "Outputs")

st.set_page_config(
    page_title="FaceX AI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

VALID_EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Happy": "#F59E0B",
    "Sad": "#14B8A6",
    "Angry": "#EF4444",
    "Fear": "#A78BFA",
    "Surprise": "#FB923C",
    "Neutral": "#22C55E",
}
EMOTION_EMOJI = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Surprise": "😲",
    "Neutral": "😐",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*='css'], .stApp {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    color: #f8fafc;
}
[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border-right: 1px solid rgba(129, 140, 248, 0.25);
    color: #f8fafc;
}
[data-testid='stSidebar'] * {
    color: #f8fafc !important;
}
[data-testid='stSidebar'] .stSelectbox > div > div,
[data-testid='stSidebar'] .stMultiSelect > div > div,
[data-testid='stSidebar'] .stDateInput input,
[data-testid='stSidebar'] .stSlider [data-testid='stThumbValue'] {
    background: #1f2937 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(129, 140, 248, 0.35) !important;
    border-radius: 10px !important;
}
[data-testid='stSidebar'] .stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(129, 140, 248, 0.35);
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 600;
    width: 100%;
    padding: 0.6rem 1rem;
    margin-bottom: 0.3rem;
    transition: all 0.2s ease;
}
[data-testid='stSidebar'] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
[data-testid='stSidebar'] .stMetric {
    background: rgba(17, 24, 39, 0.95);
    border: 1px solid rgba(129, 140, 248, 0.2);
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
}
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
.stMetric {
    background: rgba(17, 24, 39, 0.84);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    padding: 0.8rem 1rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
}
.stDataFrame, .stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(148, 163, 184, 0.16);
}
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.35);
    background: linear-gradient(135deg, #4338ca, #6366f1);
    color: white;
}
.stTextInput > div > div > input, .stSelectbox > div > div, .stDateInput input {
    background: rgba(17, 24, 39, 0.85);
    color: #f8fafc;
    border-radius: 10px;
    border: 1px solid rgba(148, 163, 184, 0.2);
}

/* Navigation button active state */
.nav-active > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    border: 1px solid #818cf8 !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.35) !important;
}
.nav-inactive > button {
    background: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}

/* Section card styling */
.section-card {
    background: rgba(17, 24, 39, 0.84);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #818cf8;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stHeader"] {
    background-color: #ffffff !important;
}
[data-testid="stHeader"]::before {
    content: "AI-Driven Human Behavior Analysis Using Computer Vision";
    display: flex;
    align-items: center;
    color: #0f172a;
    font-size: 1.5rem;
    font-weight: 800;
    padding-left: 2rem;
    height: 100%;
    width: 100%;
    letter-spacing: 0.02em;
}
</style>
"""


@st.cache_data(ttl=30)
def load_data(json_path: str) -> pd.DataFrame:
    json_path = os.path.join(BASE_DIR, json_path) if not os.path.isabs(json_path) else json_path
    if not os.path.exists(json_path):
        return pd.DataFrame()

    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["date"] = df["timestamp"].dt.date
            df["date_str"] = df["timestamp"].dt.strftime("%d/%m/%Y")
            df["hour"] = df["timestamp"].dt.hour

        if "expression" in df.columns:
            df["expression"] = df["expression"].fillna("Neutral").astype(str).str.capitalize()
            df["expression"] = df["expression"].apply(lambda value: value if value in VALID_EMOTIONS else "Neutral")

        return df
    except Exception as exc:
        st.error(f"Unable to load data: {exc}")
        return pd.DataFrame()


def get_runs() -> list[str]:
    run_dirs = [
        os.path.basename(path)
        for path in glob.glob(os.path.join(OUTPUTS_DIR, "run*"))
        if os.path.isdir(path)
    ]
    return sorted(run_dirs, reverse=True)


def resolve_face_image_path(face_image: str, run_dir: str) -> str | None:
    if not face_image:
        return None

    clean_path = str(face_image).strip().replace("\\", "/")
    candidates = []
    
    if run_dir:
        run_name = os.path.basename(run_dir)
        candidates.append(os.path.join(OUTPUTS_DIR, run_name, clean_path))
        candidates.append(os.path.join(BASE_DIR, run_dir, clean_path))
        candidates.append(os.path.join(run_dir, clean_path))

    candidates.extend([
        os.path.join(BASE_DIR, clean_path),
        os.path.join(OUTPUTS_DIR, clean_path),
    ])

    if run_dir and os.path.basename(run_dir):
        run_name = os.path.basename(run_dir)
        if clean_path.startswith(run_name + "/"):
            trimmed = clean_path[len(run_name) + 1 :]
            candidates.append(os.path.join(OUTPUTS_DIR, run_name, trimmed))
            candidates.append(os.path.join(BASE_DIR, run_dir, trimmed))
            candidates.append(os.path.join(run_dir, trimmed))
            candidates.append(os.path.join(os.getcwd(), "Outputs", trimmed))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def img_to_b64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("utf-8")
    except Exception:
        return None


def render_sidebar(df: pd.DataFrame, runs: list[str]):
    with st.sidebar:
        st.markdown("<div style='padding: 1rem 0.25rem 0.5rem;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 1.7rem; font-weight: 800; color: #f8fafc;'>🧠 FaceX AI</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.72rem; letter-spacing: 0.2em; color: #818cf8; margin-bottom: 1rem;'>SMART FACE ANALYTICS</div>", unsafe_allow_html=True)

        # ── Navigation buttons ──
        st.markdown("<div style='margin: 0.5rem 0 0.3rem; font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.18em;'>Navigation</div>", unsafe_allow_html=True)

        if "active_page" not in st.session_state:
            st.session_state.active_page = "Overview"

        col_ov, col_st = st.columns(2)
        with col_ov:
            if st.button("📊 Overview", key="nav_overview", use_container_width=True):
                st.session_state.active_page = "Overview"
        with col_st:
            if st.button("📈 Statistics", key="nav_statistics", use_container_width=True):
                st.session_state.active_page = "Statistics"

        # Show active indicator
        active = st.session_state.active_page
        indicator_ov = "▸ " if active == "Overview" else "  "
        indicator_st = "▸ " if active == "Statistics" else "  "
        st.markdown(
            f"<div style='text-align:center; font-size: 0.75rem; color: #818cf8; margin-bottom: 0.8rem;'>"
            f"Active: <strong>{active}</strong></div>",
            unsafe_allow_html=True,
        )

        st.markdown("---", unsafe_allow_html=True)

        selected_run = st.selectbox("Run", runs if runs else ["No runs found"], key="run_select")

        st.markdown("<div style='margin: 0.7rem 0 0.3rem; font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.18em;'>Filters</div>", unsafe_allow_html=True)
        if not df.empty and "track_id" in df.columns:
            person_options = ["All"] + [f"Person {value}" for value in sorted(df["track_id"].dropna().astype(int).unique().tolist())]
        else:
            person_options = ["All"]
        selected_person = st.selectbox("Person", person_options, key="person_select")

        selected_expressions = st.multiselect(
            "Expressions",
            options=VALID_EMOTIONS,
            default=[],
            key="expression_select",
            help="Pick one or more emotions to view captured faces and analytics."
        )

        if not df.empty and "date" in df.columns:
            min_date = df["date"].min()
            max_date = df["date"].max()
            date_range = st.date_input("Date range", value=(min_date, max_date), key="date_range")
        else:
            date_range = None

        if not df.empty and "timestamp" in df.columns:
            time_range = st.slider(
                "Time of day",
                value=(time(0, 0), time(23, 59)),
                format="HH:mm",
                key="time_range",
            )
        else:
            time_range = None



        st.markdown("<div style='margin-top: 1rem; padding: 0.8rem; border-radius: 12px; background: rgba(17, 24, 39, 0.75);'>", unsafe_allow_html=True)
        if not df.empty:
            st.metric("Persons", int(df["track_id"].nunique()))
            st.metric("Detections", len(df))
            st.metric("Runs", len(runs))
        else:
            st.caption("No data loaded yet")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return selected_run, selected_expressions, date_range, time_range, selected_person


def apply_filters(df: pd.DataFrame, selected_expressions: list[str], date_range, time_range, selected_person: str) -> pd.DataFrame:
    filtered = df.copy()
    if filtered.empty:
        return filtered

    if selected_expressions is None or len(selected_expressions) == 0:
        return filtered.iloc[0:0]

    if len(selected_expressions) != len(VALID_EMOTIONS):
        filtered = filtered[filtered["expression"].isin(selected_expressions)] if "expression" in filtered.columns else filtered


    if selected_person and selected_person != "All" and "track_id" in filtered.columns:
        try:
            person_id = int(selected_person.replace("Person ", ""))
            filtered = filtered[filtered["track_id"].astype(int) == person_id]
        except Exception:
            pass

    if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "date" in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]

    if time_range and isinstance(time_range, tuple) and len(time_range) == 2 and "timestamp" in filtered.columns:
        start_time, end_time = time_range
        try:
            filtered = filtered[(filtered["timestamp"].dt.time >= start_time) & (filtered["timestamp"].dt.time <= end_time)]
        except Exception:
            pass

    return filtered


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div style='padding: 0.2rem 0 0.8rem;'>
        <div style='font-size: 2rem; font-weight: 800; color: #f8fafc;'>{title}</div>
        <div style='font-size: 0.95rem; color: #94a3b8; margin-top: 0.25rem;'>{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(df: pd.DataFrame, selected_run: str) -> None:
    if df.empty:
        st.info("No matching data for the selected run and filters.")
        return

    counts = Counter(df["expression"].tolist()) if "expression" in df.columns else Counter()
    top_expression = counts.most_common(1)[0][0] if counts else "—"

    metrics = [
        ("Unique persons", int(df["track_id"].nunique()), "👥"),
        ("Detections", len(df), "📋"),
        ("Top emotion", top_expression, "🏆"),
        ("Run", selected_run, "📁"),
    ]

    cols = st.columns(4)
    for col, (label, value, emoji) in zip(cols, metrics):
        col.markdown(
            f"""
            <div style='background: rgba(17, 24, 39, 0.84); border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 16px; padding: 1rem; box-shadow: 0 10px 30px rgba(0,0,0,0.18);'>
                <div style='color: #818cf8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.16em;'>{emoji} {label}</div>
                <div style='font-size: 1.5rem; font-weight: 800; color: #f8fafc; margin-top: 0.4rem;'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return

    chart_frame = pd.DataFrame({
        "Expression": [emo for emo in VALID_EMOTIONS if emo in df["expression"].tolist()],
        "Count": [df[df["expression"] == emo].shape[0] for emo in VALID_EMOTIONS if emo in df["expression"].tolist()],
    })

    if chart_frame.empty:
        return

    col_left, col_right = st.columns(2)

    with col_left:
        bar_fig = px.bar(
            chart_frame,
            x="Expression",
            y="Count",
            color="Expression",
            color_discrete_map=EMOTION_COLORS,
            text="Count",
        )
        bar_fig.update_traces(marker_line_width=0, textposition="outside")
        bar_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#e5e7eb"),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_right:
        pie_fig = px.pie(
            chart_frame,
            names="Expression",
            values="Count",
            color="Expression",
            color_discrete_map=EMOTION_COLORS,
            hole=0.55,
        )
        pie_fig.update_traces(textinfo="percent+label", pull=[0.03] * len(chart_frame))
        pie_fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        )
        st.plotly_chart(pie_fig, use_container_width=True)


def render_gallery(df: pd.DataFrame, run_dir: str) -> None:
    if df.empty or "face_image" not in df.columns:
        st.info("No face images are available for the current selection.")
        return

    items = []
    seen = set()
    for _, row in df.iterrows():
        image_path = resolve_face_image_path(str(row.get("face_image", "")), run_dir)
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        items.append({
            "path": image_path,
            "emotion": str(row.get("expression", "Neutral")),
            "id": str(row.get("track_id", "?")),
            "frame": row.get("frame_number", ""),
        })

    if not items:
        st.info("Captured faces were not found for this run yet.")
        return

    st.markdown("<div style='color: #94a3b8; font-size: 0.92rem; margin-bottom: 0.6rem;'>Showing the best captured faces for this selection.</div>", unsafe_allow_html=True)
    cols_per_row = 4
    for start in range(0, len(items), cols_per_row):
        chunk = items[start : start + cols_per_row]
        columns = st.columns(cols_per_row)
        for column, item in zip(columns, chunk):
            image_b64 = img_to_b64(item["path"])
            if not image_b64:
                continue
            color = EMOTION_COLORS.get(item["emotion"], "#6366f1")
            column.markdown(
                f"""
                <div style='background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 16px; padding: 0.7rem; box-shadow: 0 10px 25px rgba(0,0,0,0.18);'>
                    <img src='data:image/jpeg;base64,{image_b64}' style='width: 100%; aspect-ratio: 4 / 4.2; object-fit: cover; border-radius: 12px;' />
                    <div style='margin-top: 0.7rem; color: {color}; font-weight: 700;'>{EMOTION_EMOJI.get(item["emotion"], "🙂")} {item["emotion"]}</div>
                    <div style='color: #64748b; font-size: 0.72rem; margin-top: 0.2rem;'>Frame {item["frame"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recent_activity(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No recent activity to show.")
        return

    display_cols = [col for col in ["timestamp", "track_id", "expression", "frame_number"] if col in df.columns]
    table_df = df[display_cols].copy()
    if "timestamp" in table_df.columns:
        table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    if "track_id" in table_df.columns:
        table_df["track_id"] = table_df["track_id"].astype(str)
    st.dataframe(table_df.head(20), use_container_width=True, hide_index=True)


def has_active_selection(selected_expressions: list[str], selected_person: str) -> bool:
    if selected_person != "All":
        return True
    if selected_expressions is None or len(selected_expressions) == 0:
        return False
    return True


# ═══════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════════════════════
def render_overview_page(df: pd.DataFrame, df_raw: pd.DataFrame, selected_run: str, run_dir: str,
                         selected_expressions, selected_person) -> None:
    """Render the Overview page – hero section, KPI cards, charts, gallery, and recent activity."""

    render_header("📊 Overview", "A bird's-eye view of your face expression analytics.")

    # ── Hero summary banner ──
    if not df_raw.empty:
        total_detections = len(df_raw)
        total_persons = int(df_raw["track_id"].nunique()) if "track_id" in df_raw.columns else 0
        total_emotions = df_raw["expression"].nunique() if "expression" in df_raw.columns else 0
        date_span = ""
        if "timestamp" in df_raw.columns:
            min_ts = df_raw["timestamp"].min()
            max_ts = df_raw["timestamp"].max()
            if pd.notna(min_ts) and pd.notna(max_ts):
                date_span = f"{min_ts.strftime('%b %d, %Y %H:%M')} — {max_ts.strftime('%b %d, %Y %H:%M')}"

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.10));
                    border: 1px solid rgba(129, 140, 248, 0.3); border-radius: 18px;
                    padding: 1.5rem 2rem; margin-bottom: 1.5rem;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);'>
            <div style='font-size: 1.3rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.4rem;'>
                Welcome to Run: <span style='color: #818cf8;'>{selected_run}</span>
            </div>
            <div style='color: #94a3b8; font-size: 0.9rem; line-height: 1.6;'>
                This run contains <strong style='color:#f8fafc;'>{total_detections:,}</strong> detections
                across <strong style='color:#f8fafc;'>{total_persons}</strong> unique person(s),
                with <strong style='color:#f8fafc;'>{total_emotions}</strong> distinct emotions detected.
                {f"<br/>🕐 Time span: <em>{date_span}</em>" if date_span else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No data available for this run. Select a different run from the sidebar.")
        return

    # Check if filters are active
    if not has_active_selection(selected_expressions, selected_person):
        st.info("👈 Select one or more expressions or a person from the sidebar to view detailed analytics.")
        return

    if df.empty:
        st.warning("No records match the chosen filters. Try widening your selection.")
        return

    # ── KPI Metric Cards ──
    render_metrics(df, selected_run)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # ── Emotion Distribution Charts ──
    st.markdown("""
    <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 1rem 0 0.5rem;
         text-transform: uppercase; letter-spacing: 0.08em;'>
        🎭 Emotion Distribution
    </div>""", unsafe_allow_html=True)
    render_charts(df)

    # ── Captured Faces Gallery ──
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
         text-transform: uppercase; letter-spacing: 0.08em;'>
        📸 Captured Faces
    </div>""", unsafe_allow_html=True)
    render_gallery(df, run_dir)

    # ── Recent Activity ──
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
         text-transform: uppercase; letter-spacing: 0.08em;'>
        🕑 Recent Activity
    </div>""", unsafe_allow_html=True)
    render_recent_activity(df)


# ═══════════════════════════════════════════════════════
# STATISTICS PAGE
# ═══════════════════════════════════════════════════════
def render_statistics_page(df: pd.DataFrame, df_raw: pd.DataFrame, selected_run: str,
                           selected_expressions, selected_person) -> None:
    """Render the Statistics page – detailed analytics with advanced charts and breakdowns."""

    render_header("📈 Statistics", "Deep-dive analytics and detailed breakdowns of your expression data.")

    if df_raw.empty:
        st.info("No data available for this run. Select a different run from the sidebar.")
        return

    if not has_active_selection(selected_expressions, selected_person):
        st.info("👈 Select one or more expressions or a person from the sidebar to view statistics.")
        return

    if df.empty:
        st.warning("No records match the chosen filters. Try widening your selection.")
        return

    # ── Summary Stats Cards ──
    counts = Counter(df["expression"].tolist()) if "expression" in df.columns else Counter()
    total = len(df)

    # Top row: per-emotion percentage cards
    st.markdown("""
    <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
         text-transform: uppercase; letter-spacing: 0.08em;'>
        🔢 Emotion Breakdown
    </div>""", unsafe_allow_html=True)

    emotion_cols = st.columns(len(VALID_EMOTIONS))
    for col, emotion in zip(emotion_cols, VALID_EMOTIONS):
        count = counts.get(emotion, 0)
        pct = (count / total * 100) if total > 0 else 0
        color = EMOTION_COLORS.get(emotion, "#6366f1")
        emoji = EMOTION_EMOJI.get(emotion, "🙂")
        col.markdown(f"""
        <div style='background: rgba(17, 24, 39, 0.84); border: 1px solid {color}33;
                    border-radius: 14px; padding: 0.9rem 0.7rem; text-align: center;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);'>
            <div style='font-size: 1.6rem;'>{emoji}</div>
            <div style='color: {color}; font-size: 0.75rem; font-weight: 700; margin-top: 0.3rem;
                        text-transform: uppercase; letter-spacing: 0.1em;'>{emotion}</div>
            <div style='font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;'>{count}</div>
            <div style='font-size: 0.75rem; color: #94a3b8;'>{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Row 1: Timeline + Hourly Heatmap ──
    col_timeline, col_hourly = st.columns(2)

    with col_timeline:
        st.markdown("""
        <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
             text-transform: uppercase; letter-spacing: 0.08em;'>
            📅 Emotion Timeline
        </div>""", unsafe_allow_html=True)

        try:
            if "timestamp" in df.columns and "expression" in df.columns:
                timeline_df = df.copy()
                timeline_df = timeline_df.dropna(subset=["timestamp"])
                if not timeline_df.empty:
                    timeline_df["time_bin"] = timeline_df["timestamp"].dt.floor("5min")
                    timeline_grouped = timeline_df.groupby(["time_bin", "expression"]).size().reset_index(name="count")

                    if not timeline_grouped.empty:
                        fig_timeline = px.area(
                            timeline_grouped,
                            x="time_bin",
                            y="count",
                            color="expression",
                            color_discrete_map=EMOTION_COLORS,
                            labels={"time_bin": "Time", "count": "Detections", "expression": "Emotion"},
                        )
                        fig_timeline.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e5e7eb"),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.caption("Not enough time-series data to plot.")
                else:
                    st.caption("Not enough time-series data to plot.")
            else:
                st.caption("Timestamp data not available.")
        except (ValueError, TypeError, KeyError) as exc:
            st.caption(f"Could not render timeline chart.")

    with col_hourly:
        st.markdown("""
        <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
             text-transform: uppercase; letter-spacing: 0.08em;'>
            🕐 Hourly Activity Heatmap
        </div>""", unsafe_allow_html=True)

        try:
            if "hour" in df.columns and "expression" in df.columns:
                heatmap_df = df.groupby(["hour", "expression"]).size().reset_index(name="count")
                heatmap_pivot = heatmap_df.pivot_table(index="expression", columns="hour", values="count", fill_value=0)

                # Sort emotions to match VALID_EMOTIONS order
                heatmap_pivot = heatmap_pivot.reindex([e for e in VALID_EMOTIONS if e in heatmap_pivot.index])

                if heatmap_pivot.empty or heatmap_pivot.values.size == 0:
                    st.caption("Not enough data for heatmap.")
                else:
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_pivot.values.tolist(),
                        x=[f"{int(h):02d}:00" for h in heatmap_pivot.columns.tolist()],
                        y=heatmap_pivot.index.tolist(),
                        colorscale=[
                            [0, "rgba(15, 23, 42, 0.9)"],
                            [0.25, "#312e81"],
                            [0.5, "#4338ca"],
                            [0.75, "#6366f1"],
                            [1, "#a78bfa"],
                        ],
                        hoverongaps=False,
                        showscale=True,
                        colorbar=dict(title="Count", titlefont=dict(color="#94a3b8"), tickfont=dict(color="#94a3b8")),
                    ))
                    fig_heatmap.update_layout(
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#e5e7eb"),
                        xaxis=dict(title="Hour of Day", showgrid=False),
                        yaxis=dict(title="", showgrid=False),
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.caption("Hourly data not available.")
        except (ValueError, TypeError, KeyError) as exc:
            st.caption(f"Could not render heatmap chart.")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Row 2: Per-Person Breakdown + Emotion Transitions ──
    col_person, col_transitions = st.columns(2)

    with col_person:
        st.markdown("""
        <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
             text-transform: uppercase; letter-spacing: 0.08em;'>
            👤 Per-Person Emotion Breakdown
        </div>""", unsafe_allow_html=True)

        if "track_id" in df.columns and "expression" in df.columns:
            person_expr_df = df.groupby(["track_id", "expression"]).size().reset_index(name="count")
            person_expr_df["track_id"] = person_expr_df["track_id"].apply(lambda x: f"Person {int(x)}")

            fig_person = px.bar(
                person_expr_df,
                x="track_id",
                y="count",
                color="expression",
                color_discrete_map=EMOTION_COLORS,
                barmode="stack",
                labels={"track_id": "Person", "count": "Count", "expression": "Emotion"},
            )
            fig_person.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e5e7eb"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.1)"),
            )
            st.plotly_chart(fig_person, use_container_width=True)
        else:
            st.caption("Person tracking data not available.")

    with col_transitions:
        st.markdown("""
        <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
             text-transform: uppercase; letter-spacing: 0.08em;'>
            🔄 Emotion Transitions
        </div>""", unsafe_allow_html=True)

        try:
            if "track_id" in df.columns and "expression" in df.columns and "frame_number" in df.columns:
                transitions = Counter()
                for tid in df["track_id"].unique():
                    person_df = df[df["track_id"] == tid].sort_values("frame_number")
                    expressions = person_df["expression"].tolist()
                    for i in range(len(expressions) - 1):
                        if expressions[i] != expressions[i + 1]:
                            transitions[(expressions[i], expressions[i + 1])] += 1

                if transitions:
                    sources, targets, values, link_colors = [], [], [], []
                    unique_labels = list(VALID_EMOTIONS)
                    label_index = {label: i for i, label in enumerate(unique_labels)}

                    for (src, tgt), val in transitions.most_common(15):
                        if src in label_index and tgt in label_index:
                            sources.append(label_index[src])
                            targets.append(label_index[tgt])
                            values.append(val)
                            color = EMOTION_COLORS.get(src, "#6366f1")
                            # Make it semi-transparent
                            link_colors.append(color + "66")

                    # Only render if we have valid links
                    if sources and targets and values:
                        node_colors = [EMOTION_COLORS.get(e, "#6366f1") for e in unique_labels]

                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="rgba(0,0,0,0)", width=0),
                                label=[f"{EMOTION_EMOJI.get(e, '')} {e}" for e in unique_labels],
                                color=node_colors,
                            ),
                            link=dict(
                                source=sources,
                                target=targets,
                                value=values,
                                color=link_colors,
                            ),
                        )])
                        fig_sankey.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#e5e7eb", size=11),
                        )
                        st.plotly_chart(fig_sankey, use_container_width=True)
                    else:
                        st.caption("No valid emotion transitions to display.")
                else:
                    st.caption("No emotion transitions detected in this selection.")
            else:
                st.caption("Transition data not available.")
        except (ValueError, TypeError, KeyError) as exc:
            st.caption(f"Could not render transitions chart.")

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # ── Full Data Table ──
    st.markdown("""
    <div class='section-title' style='font-size: 1.1rem; font-weight: 700; color: #818cf8; margin: 0.5rem 0;
         text-transform: uppercase; letter-spacing: 0.08em;'>
        📋 Full Data Table
    </div>""", unsafe_allow_html=True)

    display_cols = [col for col in ["timestamp", "track_id", "expression", "frame_number"] if col in df.columns]
    table_df = df[display_cols].copy()
    if "timestamp" in table_df.columns:
        table_df["timestamp"] = table_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "track_id" in table_df.columns:
        table_df["track_id"] = table_df["track_id"].astype(str)

    st.dataframe(table_df, use_container_width=True, hide_index=True, height=400)

    # ── Download CSV ──
    csv = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download filtered data as CSV",
        data=csv,
        file_name=f"facex_ai_{selected_run}_filtered.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    runs = get_runs()
    selected_run = runs[0] if runs else "No runs found"
    result_path = os.path.join(OUTPUTS_DIR, selected_run, "result.json") if selected_run != "No runs found" else ""
    df_raw = load_data(result_path)

    selected_run, selected_expressions, date_range, time_range, selected_person = render_sidebar(df_raw, runs)

    result_path = os.path.join(OUTPUTS_DIR, selected_run, "result.json") if selected_run != "No runs found" else ""
    df_raw = load_data(result_path)
    df = apply_filters(df_raw, selected_expressions, date_range, time_range, selected_person)

    run_dir = os.path.join("Outputs", selected_run) if selected_run != "No runs found" else ""

    # Route to the active page
    active_page = st.session_state.get("active_page", "Overview")

    if active_page == "Overview":
        render_overview_page(df, df_raw, selected_run, run_dir, selected_expressions, selected_person)
    else:
        render_statistics_page(df, df_raw, selected_run, selected_expressions, selected_person)


if __name__ == "__main__":
    main()
