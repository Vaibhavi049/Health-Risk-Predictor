import streamlit as st
from kafka import KafkaProducer
import pandas as pd
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
from datetime import datetime

# ----------------- SQLite Setup -----------------
DB_PATH = "health_predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT,
            age          INTEGER,
            weight       REAL,
            height       REAL,
            bmi          REAL,
            exercise     INTEGER,
            sleep        REAL,
            sugar_intake INTEGER,
            alcohol      INTEGER,
            smoking      INTEGER,
            married      INTEGER,
            profession   TEXT,
            predicted_risk TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions
            (timestamp, age, weight, height, bmi, exercise, sleep,
             sugar_intake, alcohol, smoking, married, profession, predicted_risk)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["age"], data["weight"], data["height"], data["bmi"],
        data["exercise"], data["sleep"], data["sugar_intake"],
        data["alcohol"], data["smoking"], data["married"],
        data["profession"], data["predicted_risk"],
    ))
    conn.commit()
    conn.close()

def load_predictions() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id ASC", conn)
    conn.close()
    return df

init_db()

# ----------------- Kafka Setup -----------------
try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    KAFKA_AVAILABLE = True
except Exception:
    KAFKA_AVAILABLE = False

# ----------------- Load Model -----------------
try:
    model          = joblib.load("model.pkl")
    le_dict        = joblib.load("encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# ----------------- Session State -----------------
if 'predictions_list' not in st.session_state:
    st.session_state['predictions_list'] = []
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "predictor"

# ----------------- Page Config -----------------
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- Plotly shared theme -----------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#2d5a4a", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="#c4e8d8",
        borderwidth=1,
        font=dict(size=11)
    ),
    hoverlabel=dict(
        bgcolor="#ffffff",
        bordercolor="#c4e8d8",
        font=dict(family="DM Sans, sans-serif", color="#0d6e56", size=12)
    ),
)

# Binary classification: only High and Low
RISK_COLORS = {
    "Low":  "#1a9e78",
    "High": "#e05252",
    "low":  "#1a9e78",
    "high": "#e05252",
}

# ----------------- CSS -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #e8f5f0 0%, #f0faf6 40%, #e3f4ed 100%);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1200px; }

    .hero-header {
        background: linear-gradient(135deg, #0d6e56 0%, #1a9e78 60%, #22c997 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2.5rem;
        display: flex;
        align-items: center;
        gap: 1.5rem;
        box-shadow: 0 8px 32px rgba(13,110,86,0.18);
    }
    .hero-header h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-header p { color: rgba(255,255,255,0.82); margin: 0.4rem 0 0 0; font-size: 1rem; font-weight: 300; }
    .hero-icon { font-size: 3.5rem; line-height: 1; }

    /* ── Tab Navigation ── */
    .nav-bar {
        display: flex; gap: 0.6rem; margin-bottom: 2rem;
        background: #ffffff; border-radius: 14px; padding: 0.45rem;
        border: 1px solid rgba(13,110,86,0.12);
        box-shadow: 0 2px 12px rgba(13,110,86,0.06);
        width: fit-content;
    }
    .nav-btn {
        padding: 0.55rem 1.5rem; border-radius: 10px; font-size: 0.9rem;
        font-weight: 500; cursor: pointer; border: none; transition: all 0.2s;
        font-family: 'DM Sans', sans-serif; letter-spacing: 0.01em;
    }
    .nav-btn-active {
        background: linear-gradient(135deg, #0d6e56, #1a9e78);
        color: #ffffff; box-shadow: 0 3px 10px rgba(13,110,86,0.3);
    }
    .nav-btn-inactive {
        background: transparent; color: #4a8a72;
    }
    .nav-btn-inactive:hover { background: #f0faf6; color: #0d6e56; }

    .section-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
        border: 1px solid rgba(13,110,86,0.1);
        box-shadow: 0 2px 16px rgba(13,110,86,0.06);
    }
    .section-title {
        font-size: 1.05rem; font-weight: 600; color: #0d6e56;
        text-transform: uppercase; letter-spacing: 0.07em;
        margin-bottom: 1.4rem; display: flex; align-items: center; gap: 0.5rem;
    }
    .section-title::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(13,110,86,0.2), transparent);
        margin-left: 0.5rem;
    }

    div[data-testid="stSlider"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        font-size: 0.88rem !important; font-weight: 500 !important;
        color: #2d5a4a !important; letter-spacing: 0.02em;
    }
    div[data-testid="stSlider"] > div > div > div > div { background: #1a9e78 !important; }

    div[data-testid="stSelectbox"] > div > div {
        border: 1.5px solid #b2d8ce !important; border-radius: 10px !important;
        background: #f6fdfb !important; transition: border 0.2s;
    }
    div[data-testid="stSelectbox"] > div > div:hover { border-color: #1a9e78 !important; }
    div[data-testid="stNumberInput"] input {
        border: 1.5px solid #b2d8ce !important; border-radius: 10px !important; background: #f6fdfb !important;
    }

    div[data-testid="stForm"] button[type="submit"], .stButton > button {
        background: linear-gradient(135deg, #0d6e56, #1a9e78) !important;
        color: white !important; border: none !important; border-radius: 12px !important;
        font-size: 1rem !important; font-weight: 500 !important;
        padding: 0.65rem 2.2rem !important; cursor: pointer !important;
        box-shadow: 0 4px 14px rgba(13,110,86,0.3) !important;
        transition: all 0.2s ease !important; width: 100%;
    }
    div[data-testid="stForm"] button[type="submit"]:hover, .stButton > button:hover {
        background: linear-gradient(135deg, #0a5a46, #168f6c) !important;
        box-shadow: 0 6px 18px rgba(13,110,86,0.38) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stAlert"] { border-radius: 12px !important; }

    .risk-badge {
        display: inline-block; padding: 0.55rem 1.4rem;
        border-radius: 50px; font-weight: 600; font-size: 1.1rem; margin-top: 0.3rem;
    }
    .risk-low  { background: #d4f0e5; color: #0a5a46; }
    .risk-high { background: #fde8e8; color: #8b1a1a; }

    /* ── Summary stat cards ── */
    .stat-grid { display: flex; gap: 1rem; margin-bottom: 1.8rem; flex-wrap: wrap; }
    .stat-card {
        flex: 1; min-width: 140px;
        border-radius: 14px; padding: 1.2rem 1.4rem;
        display: flex; flex-direction: column; gap: 0.25rem;
    }
    .stat-card-total { background: linear-gradient(135deg,#e8f5f0,#d0ede3); border: 1px solid #b2d8ce; }
    .stat-card-low   { background: linear-gradient(135deg,#d4f0e5,#c0ebd8); border: 1px solid #a5dfc8; }
    .stat-card-high  { background: linear-gradient(135deg,#fdf0f0,#fadddd); border: 1px solid #f0aeae; }
    .stat-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.07em; color: #4a8a72; }
    .stat-value { font-size: 2rem; font-weight: 700; color: #0d6e56; line-height: 1.1; }
    .stat-pct   { font-size: 0.82rem; color: #6aaa90; font-weight: 500; }

    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .metric-card {
        flex: 1; background: #f2fbf7; border: 1px solid #c4e8d8;
        border-radius: 12px; padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .m-label {
        font-size: 0.75rem; color: #4a8a72; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    .metric-card .m-value { font-size: 1.6rem; font-weight: 600; color: #0d6e56; margin-top: 0.2rem; }

    .chart-label { font-size: 0.82rem; color: #4a8a72; font-weight: 500; margin-bottom: 0.3rem; }
    hr { border-color: rgba(13,110,86,0.1) !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ── Empty state ── */
    .empty-state {
        text-align: center; padding: 4rem 2rem; color: #8ab5a8;
    }
    .empty-state .es-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .empty-state .es-title { font-size: 1.1rem; font-weight: 600; color: #4a8a72; margin-bottom: 0.4rem; }
    .empty-state .es-sub   { font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
    <div class="hero-header">
        <div class="hero-icon">🫀</div>
        <div>
            <h1>Health Risk Predictor</h1>
            <p>Enter your health metrics below to receive a personalised risk assessment powered by machine learning.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════
nav_col1, nav_col2, _ = st.columns([1.1, 1.6, 4])
with nav_col1:
    if st.button("🔍  Risk Predictor", key="nav_predictor",
                 type="primary" if st.session_state['active_tab'] == "predictor" else "secondary"):
        st.session_state['active_tab'] = "predictor"
        st.rerun()
with nav_col2:
    if st.button("📊  User Predictions Dashboard", key="nav_dashboard",
                 type="primary" if st.session_state['active_tab'] == "dashboard" else "secondary"):
        st.session_state['active_tab'] = "dashboard"
        st.rerun()

st.markdown("<hr style='margin: 0.5rem 0 1.8rem 0;'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# TAB 1 – PREDICTOR
# ═══════════════════════════════════════════════════════════
if st.session_state['active_tab'] == "predictor":

    known_professions = ["student", "office", "teacher"]
    col_form, col_results = st.columns([1.05, 0.95], gap="large")

    with col_form:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Your Health Details</div>', unsafe_allow_html=True)

        with st.form("health_form"):
            st.markdown("*Personal*")
            c1, c2 = st.columns(2)
            with c1:
                age    = st.slider("Age (years)", 18, 65, 25)
                height = st.slider("Height (cm)", 140, 200, 170, key="height_slider")
            with c2:
                weight = st.slider("Weight (kg)", 40, 120, 70, key="weight_slider")
                bmi    = weight / ((height / 100) ** 2)
                st.number_input("BMI", min_value=10.0, max_value=50.0,
                                value=round(bmi, 1), step=0.1, key="bmi_input", disabled=True)

            st.markdown("---")
            st.markdown("*Lifestyle*")
            c3, c4 = st.columns(2)
            with c3:
                exercise     = st.slider("Exercise (days/week)", 0, 7, 3)
                sugar_intake = st.slider("Sugar Intake (0–5)", 0, 5, 2)
            with c4:
                sleep   = st.slider("Sleep (hours/day)", 0, 12, 6)
                alcohol = st.slider("Alcohol Intake (0–3)", 0, 3, 1)

            st.markdown("---")
            st.markdown("*Background*")
            c5, c6, c7 = st.columns(3)
            with c5:
                smoking    = st.selectbox("Smoking", ["No", "Yes"])
            with c6:
                married    = st.selectbox("Married", ["No", "Yes"])
            with c7:
                profession = st.selectbox("Profession", known_professions)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍  Predict My Health Risk")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Prediction Result</div>', unsafe_allow_html=True)

        if submitted:
            bmi_at_submission = weight / ((height / 100) ** 2)
            data = {
                "age": age, "weight": weight, "height": height,
                "exercise": exercise, "sleep": sleep,
                "sugar_intake": sugar_intake,
                "smoking": 1 if smoking == "Yes" else 0,
                "alcohol": alcohol,
                "married": 1 if married == "Yes" else 0,
                "profession": profession, "bmi": bmi_at_submission
            }

            if KAFKA_AVAILABLE:
                producer.send('health_topic', value=data)

            if MODEL_AVAILABLE:
                df_pred = pd.DataFrame([data])
                for col, le in le_dict.items():
                    if col in df_pred.columns:
                        df_pred[col] = df_pred[col].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0])
                        df_pred[col] = le.transform(df_pred[col])

                prediction = model.predict(df_pred)
                risk = target_encoder.inverse_transform(prediction)[0]
            else:
                # Fallback demo risk for UI preview when model is absent
                risk = "High"

            data['predicted_risk'] = risk
            st.session_state['predictions_list'].append(data)

            save_prediction(data)

            risk_lower  = risk.lower()
            # Binary: only low or high
            badge_class = "risk-low" if risk_lower == "low" else "risk-high"
            risk_icon   = "✅" if risk_lower == "low" else "⚠️"

            st.markdown(f"""
                <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
                    <div style="font-size:3rem; margin-bottom:0.5rem;">{risk_icon}</div>
                    <div style="color:#4a8a72; font-size:0.9rem; font-weight:500; margin-bottom:0.4rem;">
                        Predicted Health Risk Level
                    </div>
                    <span class="risk-badge {badge_class}">{risk}</span>
                </div>
            """, unsafe_allow_html=True)

            # Lifestyle radar
            bmi_score = max(0, 10 - abs(bmi_at_submission - 22) * 0.8)
            radar_categories = ["Exercise", "Sleep", "Sugar Control", "Alcohol Control", "BMI Score"]
            radar_vals = [
                round(exercise / 7 * 10, 1),
                round(sleep / 12 * 10, 1),
                round((5 - sugar_intake) / 5 * 10, 1),
                round((3 - alcohol) / 3 * 10, 1),
                round(bmi_score, 1),
            ]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                fillcolor='rgba(26,158,120,0.18)',
                line=dict(color="#0d6e56", width=2),
                hovertemplate="%{theta}: %{r}/10<extra></extra>"
            ))
            fig_radar.update_layout(
                **PLOT_LAYOUT,
                polar=dict(
                    bgcolor="rgba(240,250,246,0.6)",
                    radialaxis=dict(visible=True, range=[0, 10],
                                   tickfont=dict(size=9), gridcolor="#c4e8d8"),
                    angularaxis=dict(tickfont=dict(size=11, color="#2d5a4a"), gridcolor="#c4e8d8"),
                ),
                showlegend=False, height=270,
                title=dict(text="Lifestyle Health Score (0–10)",
                           font=dict(size=13, color="#0d6e56"), x=0.5)
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

            st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="m-label">BMI</div>
                        <div class="m-value">{bmi_at_submission:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Exercise</div>
                        <div class="m-value">{exercise}d</div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Sleep</div>
                        <div class="m-value">{sleep}h</div>
                    </div>
                    <div class="metric-card">
                        <div class="m-label">Total Runs</div>
                        <div class="m-value">{len(st.session_state['predictions_list'])}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            if KAFKA_AVAILABLE:
                st.success("✅ Record sent to Kafka pipeline & saved to database.")
            else:
                st.success("✅ Record saved to database successfully.")
        else:
            st.markdown("""
                <div style="text-align:center; padding: 3rem 1rem; color:#8ab5a8;">
                    <div style="font-size:3rem; margin-bottom:0.8rem;">🩺</div>
                    <div style="font-size:1rem; font-weight:500;">
                        Fill in your details and click<br>
                        <strong>Predict My Health Risk</strong> to see your result.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Real-time in-session dashboard ──
    if st.session_state['predictions_list']:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Real-time Session Dashboard</div>', unsafe_allow_html=True)

        known_professions = ["student", "office", "teacher"]
        df_dash = pd.DataFrame(st.session_state['predictions_list'])

        fc1, fc2, _ = st.columns([1.2, 1.2, 2.6])
        with fc1:
            selected_prof = st.selectbox("Profession", ["All"] + known_professions, key="dash_filter")
        with fc2:
            # Binary filter: only High and Low
            selected_risk = st.selectbox("Risk Level", ["All", "Low", "High"], key="risk_filter")

        if selected_prof != "All":
            df_dash = df_dash[df_dash['profession'] == selected_prof]
        if selected_risk != "All":
            df_dash = df_dash[df_dash['predicted_risk'].str.lower() == selected_risk.lower()]

        st.markdown("---")

        if df_dash.empty:
            st.info("No data matches your current filters.")
        else:
            r1a, r1b = st.columns(2, gap="medium")

            with r1a:
                st.markdown('<div class="chart-label">Risk Level Distribution</div>', unsafe_allow_html=True)
                risk_counts = df_dash['predicted_risk'].value_counts().reset_index()
                risk_counts.columns = ['Risk', 'Count']
                fig_donut = px.pie(risk_counts, names='Risk', values='Count', hole=0.52,
                                   color='Risk', color_discrete_map=RISK_COLORS)
                fig_donut.update_traces(
                    textposition='outside', textfont=dict(size=12),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
                    pull=[0.04] * len(risk_counts)
                )
                fig_donut.update_layout(**PLOT_LAYOUT, height=290, showlegend=True)
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

            with r1b:
                st.markdown('<div class="chart-label">Avg BMI & Sleep per Risk Level</div>', unsafe_allow_html=True)
                summary = df_dash.groupby('predicted_risk').agg(
                    avg_bmi=('bmi', 'mean'), avg_sleep=('sleep', 'mean')
                ).reset_index()
                fig_group = go.Figure()
                fig_group.add_trace(go.Bar(
                    name='Avg BMI', x=summary['predicted_risk'], y=summary['avg_bmi'].round(1),
                    marker_color="#1a9e78",
                    hovertemplate="Risk: %{x}<br>Avg BMI: %{y:.1f}<extra></extra>"
                ))
                fig_group.add_trace(go.Bar(
                    name='Avg Sleep (h)', x=summary['predicted_risk'], y=summary['avg_sleep'].round(1),
                    marker_color="#f5a623",
                    hovertemplate="Risk: %{x}<br>Avg Sleep: %{y:.1f}h<extra></extra>"
                ))
                fig_group.update_layout(
                    **PLOT_LAYOUT, barmode='group', height=290,
                    xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_group, use_container_width=True, config={"displayModeBar": False})

            r2a, r2b = st.columns(2, gap="medium")

            with r2a:
                st.markdown('<div class="chart-label">BMI vs Exercise · bubble = Age · colour = Risk</div>', unsafe_allow_html=True)
                fig_scatter = px.scatter(
                    df_dash, x='exercise', y='bmi',
                    color='predicted_risk', color_discrete_map=RISK_COLORS,
                    size='age', size_max=18,
                    custom_data=['age', 'sleep', 'profession', 'predicted_risk'],
                    labels={'exercise': 'Exercise Days/Week', 'bmi': 'BMI', 'predicted_risk': 'Risk'},
                )
                fig_scatter.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "BMI: %{y:.1f} · Exercise: %{x}d/wk<br>"
                        "Age: %{customdata[0]} · Sleep: %{customdata[1]}h<br>"
                        "Profession: %{customdata[2]}<extra></extra>"
                    )
                )
                fig_scatter.update_layout(
                    **PLOT_LAYOUT, height=290,
                    xaxis=dict(title="Exercise (days/week)", gridcolor="#e0f2ea", zeroline=False),
                    yaxis=dict(title="BMI", gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

            with r2b:
                st.markdown('<div class="chart-label">Sugar + Alcohol Burden · Profession × Risk</div>', unsafe_allow_html=True)
                heat_df = df_dash.groupby(['profession', 'predicted_risk']).agg(
                    sugar=('sugar_intake', 'mean'), alcohol=('alcohol', 'mean')
                ).reset_index()
                heat_df['burden'] = (heat_df['sugar'] + heat_df['alcohol']).round(2)
                heat_pivot = heat_df.pivot_table(
                    index='profession', columns='predicted_risk', values='burden', aggfunc='mean'
                ).fillna(0)
                fig_heat = go.Figure(data=go.Heatmap(
                    z=heat_pivot.values,
                    x=heat_pivot.columns.tolist(),
                    y=heat_pivot.index.tolist(),
                    colorscale=[[0, "#e8f5f0"], [0.5, "#1a9e78"], [1, "#0d6e56"]],
                    hovertemplate="Profession: %{y}<br>Risk: %{x}<br>Burden: %{z:.2f}<extra></extra>",
                    showscale=True,
                    colorbar=dict(thickness=12, tickfont=dict(size=10))
                ))
                fig_heat.update_layout(
                    **PLOT_LAYOUT, height=290,
                    xaxis=dict(title="Risk Level"),
                    yaxis=dict(title=""),
                )
                st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

            r3a, r3b = st.columns([1.15, 0.85], gap="medium")

            with r3a:
                st.markdown('<div class="chart-label">BMI Trend Across Submissions (3-point rolling avg)</div>', unsafe_allow_html=True)
                df_trend = df_dash.reset_index(drop=True).copy()
                df_trend['submission'] = df_trend.index + 1
                fig_trend = go.Figure()
                # Binary: iterate only Low and High
                for risk_val, grp in df_trend.groupby('predicted_risk'):
                    fig_trend.add_trace(go.Scatter(
                        x=grp['submission'], y=grp['bmi'],
                        mode='markers', name=risk_val,
                        marker=dict(color=RISK_COLORS.get(risk_val, "#aaa"), size=9, opacity=0.85),
                        hovertemplate=f"<b>{risk_val}</b><br>#%{{x}} · BMI: %{{y:.1f}}<extra></extra>"
                    ))
                rolling = df_trend.sort_values('submission').copy()
                rolling['rolling_bmi'] = rolling['bmi'].rolling(3, min_periods=1).mean().round(1)
                fig_trend.add_trace(go.Scatter(
                    x=rolling['submission'], y=rolling['rolling_bmi'],
                    mode='lines', name='3-pt avg',
                    line=dict(color="#0d6e56", width=2.5, dash='dot'),
                    hovertemplate="3-pt Avg BMI: %{y:.1f}<extra></extra>"
                ))
                fig_trend.update_layout(
                    **PLOT_LAYOUT, height=270,
                    xaxis=dict(title="Submission #", gridcolor="#e0f2ea", zeroline=False, dtick=1),
                    yaxis=dict(title="BMI", gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})

            with r3b:
                st.markdown('<div class="chart-label">Latest Predictions</div>', unsafe_allow_html=True)
                display_cols = ['age', 'bmi', 'exercise', 'sleep', 'profession', 'predicted_risk']
                st.dataframe(
                    df_dash[display_cols].tail(10).reset_index(drop=True),
                    use_container_width=True, height=270, hide_index=True,
                    column_config={
                        "predicted_risk": st.column_config.TextColumn("Risk"),
                        "bmi":  st.column_config.NumberColumn("BMI", format="%.1f"),
                        "age":  st.column_config.NumberColumn("Age"),
                        "exercise": st.column_config.ProgressColumn("Exercise", min_value=0, max_value=7, format="%d d/wk"),
                        "sleep":    st.column_config.ProgressColumn("Sleep",    min_value=0, max_value=12, format="%d h"),
                    }
                )

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("📭  No predictions yet — submit your health details above to populate the dashboard.")


# ═══════════════════════════════════════════════════════════
# TAB 2 – USER PREDICTIONS DASHBOARD  (SQLite-backed)
# ═══════════════════════════════════════════════════════════
elif st.session_state['active_tab'] == "dashboard":

    df_db = load_predictions()

    if df_db.empty:
        st.markdown("""
            <div class="empty-state">
                <div class="es-icon">📭</div>
                <div class="es-title">No predictions stored yet</div>
                <div class="es-sub">
                    Head over to the <strong>Risk Predictor</strong> tab, submit your health details,
                    and the results will appear here automatically.
                </div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Normalise risk column: keep only Low / High (drop any legacy Medium rows)
        df_db['predicted_risk'] = df_db['predicted_risk'].str.capitalize()
        df_db = df_db[df_db['predicted_risk'].isin(["Low", "High"])]

        total = len(df_db)
        risk_counts_db = df_db['predicted_risk'].value_counts()
        n_low  = int(risk_counts_db.get("Low",  0))
        n_high = int(risk_counts_db.get("High", 0))

        pct = lambda n: f"{n/total*100:.1f}%" if total else "—"

        # ── 1. Summary Stats ────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🗂️ Summary Overview</div>', unsafe_allow_html=True)

        r_col, _ = st.columns([0.18, 0.82])
        with r_col:
            if st.button("🔄 Refresh Data"):
                st.rerun()

        # Binary stat grid: Total, Low, High only
        st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card stat-card-total">
                    <div class="stat-label">Total Predictions</div>
                    <div class="stat-value">{total}</div>
                    <div class="stat-pct">All time</div>
                </div>
                <div class="stat-card stat-card-low">
                    <div class="stat-label">Low Risk</div>
                    <div class="stat-value">{n_low}</div>
                    <div class="stat-pct">{pct(n_low)} of total</div>
                </div>
                <div class="stat-card stat-card-high">
                    <div class="stat-label">High Risk</div>
                    <div class="stat-value">{n_high}</div>
                    <div class="stat-pct">{pct(n_high)} of total</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 2. Filters ──────────────────────────────────────
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔎 Filter Records</div>', unsafe_allow_html=True)

        fa, fb, fc, _ = st.columns([1.1, 1.1, 1.1, 1.7])
        with fa:
            # Binary filter: only High and Low
            f_risk = st.selectbox("Risk Level", ["All", "Low", "High"], key="db_risk_filter")
        with fb:
            prof_opts = ["All"] + sorted(df_db['profession'].dropna().unique().tolist())
            f_prof = st.selectbox("Profession", prof_opts, key="db_prof_filter")
        with fc:
            age_min, age_max = int(df_db['age'].min()), int(df_db['age'].max())
            if age_min == age_max:
                age_max = age_min + 1
            f_age = st.slider("Age Range", age_min, age_max, (age_min, age_max), key="db_age_filter")

        df_filtered = df_db.copy()
        if f_risk != "All":
            df_filtered = df_filtered[df_filtered['predicted_risk'] == f_risk]
        if f_prof != "All":
            df_filtered = df_filtered[df_filtered['profession'] == f_prof]
        df_filtered = df_filtered[df_filtered['age'].between(f_age[0], f_age[1])]

        st.markdown(
            f"<div style='color:#4a8a72; font-size:0.83rem; margin-top:-0.5rem;'>"
            f"Showing <strong>{len(df_filtered)}</strong> of <strong>{total}</strong> records</div>",
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if df_filtered.empty:
            st.info("No records match the selected filters.")
        else:
            # ── 3. Chart Row 1: Pie + Bar ──────────────────
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Risk Distribution</div>', unsafe_allow_html=True)

            ch1, ch2 = st.columns(2, gap="large")

            with ch1:
                st.markdown('<div class="chart-label">Pie Chart — Risk Level Share</div>', unsafe_allow_html=True)
                rc = df_filtered['predicted_risk'].value_counts().reset_index()
                rc.columns = ['Risk', 'Count']
                fig_pie = px.pie(
                    rc, names='Risk', values='Count', hole=0.48,
                    color='Risk', color_discrete_map=RISK_COLORS,
                )
                fig_pie.update_traces(
                    textinfo='label+percent',
                    textposition='outside',
                    textfont=dict(size=12),
                    pull=[0.05] * len(rc),
                    hovertemplate="<b>%{label}</b><br>Users: %{value}<br>Share: %{percent}<extra></extra>"
                )
                fig_pie.update_layout(**PLOT_LAYOUT, height=320, showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            with ch2:
                st.markdown('<div class="chart-label">Bar Chart — Count per Risk Level</div>', unsafe_allow_html=True)
                fig_bar = go.Figure()
                # Binary: only Low and High bars
                for risk_label, colour in [("Low", "#1a9e78"), ("High", "#e05252")]:
                    cnt = int((df_filtered['predicted_risk'] == risk_label).sum())
                    fig_bar.add_trace(go.Bar(
                        x=[risk_label], y=[cnt],
                        marker_color=colour,
                        text=[cnt], textposition='outside',
                        textfont=dict(size=14, color="#0d6e56"),
                        hovertemplate=f"<b>{risk_label}</b><br>Count: {cnt}<extra></extra>",
                        name=risk_label,
                        width=0.42
                    ))
                fig_bar.update_layout(
                    **PLOT_LAYOUT, height=320, showlegend=False,
                    barmode='group',
                    xaxis=dict(title="Risk Level", gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(title="Number of Users", gridcolor="#e0f2ea", zeroline=True,
                               zerolinecolor="#c4e8d8", rangemode="tozero"),
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            st.markdown('</div>', unsafe_allow_html=True)

            # ── 4. Chart Row 2: Timeline + Profession stacked bar ──
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Trends & Breakdowns</div>', unsafe_allow_html=True)

            ch3, ch4 = st.columns(2, gap="large")

            with ch3:
                st.markdown('<div class="chart-label">Cumulative Predictions Over Time</div>', unsafe_allow_html=True)
                df_time = df_filtered.copy()
                df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
                df_time = df_time.sort_values('timestamp')
                df_time['cumulative'] = range(1, len(df_time) + 1)

                fig_time = go.Figure()
                # Binary: only Low and High scatter points
                for risk_label, colour in [("Low", "#1a9e78"), ("High", "#e05252")]:
                    grp = df_time[df_time['predicted_risk'] == risk_label]
                    if not grp.empty:
                        fig_time.add_trace(go.Scatter(
                            x=grp['timestamp'], y=grp['cumulative'],
                            mode='markers', name=risk_label,
                            marker=dict(color=colour, size=9, opacity=0.8),
                            hovertemplate=f"<b>{risk_label}</b><br>%{{x|%Y-%m-%d %H:%M}}<br>Record #%{{y}}<extra></extra>"
                        ))
                fig_time.add_trace(go.Scatter(
                    x=df_time['timestamp'], y=df_time['cumulative'],
                    mode='lines', name='All',
                    line=dict(color="#0d6e56", width=2, dash='dot'),
                    hovertemplate="All · Record #%{y}<extra></extra>"
                ))
                fig_time.update_layout(
                    **PLOT_LAYOUT, height=310,
                    xaxis=dict(title="Date / Time", gridcolor="#e0f2ea", zeroline=False),
                    yaxis=dict(title="Cumulative Count", gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})

            with ch4:
                st.markdown('<div class="chart-label">Risk Breakdown by Profession</div>', unsafe_allow_html=True)
                prof_risk = (
                    df_filtered.groupby(['profession', 'predicted_risk'])
                    .size().reset_index(name='count')
                )
                fig_prof = px.bar(
                    prof_risk, x='profession', y='count', color='predicted_risk',
                    color_discrete_map=RISK_COLORS, barmode='stack',
                    labels={'profession': 'Profession', 'count': 'Count', 'predicted_risk': 'Risk'},
                    text='count',
                )
                fig_prof.update_traces(
                    textposition='inside', textfont=dict(size=11, color="#ffffff"),
                    hovertemplate="<b>%{x}</b><br>Risk: %{fullData.name}<br>Count: %{y}<extra></extra>"
                )
                fig_prof.update_layout(
                    **PLOT_LAYOUT, height=310,
                    xaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(title="Users", gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_prof, use_container_width=True, config={"displayModeBar": False})

            st.markdown('</div>', unsafe_allow_html=True)

            # ── 5. Chart Row 3: Avg metrics heatmap + BMI distribution ──
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔬 Health Metrics Deep-Dive</div>', unsafe_allow_html=True)

            ch5, ch6 = st.columns(2, gap="large")

            with ch5:
                st.markdown('<div class="chart-label">Average Health Metrics by Risk Level</div>', unsafe_allow_html=True)
                metrics_avg = df_filtered.groupby('predicted_risk').agg(
                    BMI=('bmi', 'mean'),
                    Exercise=('exercise', 'mean'),
                    Sleep=('sleep', 'mean'),
                    Sugar=('sugar_intake', 'mean'),
                    Alcohol=('alcohol', 'mean'),
                ).round(2).reset_index()

                metrics_cols = ['BMI', 'Exercise', 'Sleep', 'Sugar', 'Alcohol']
                heat_vals = metrics_avg[metrics_cols].values
                col_min = heat_vals.min(axis=0)
                col_max = heat_vals.max(axis=0)
                denom = (col_max - col_min)
                denom[denom == 0] = 1
                norm_vals = (heat_vals - col_min) / denom

                fig_avg_heat = go.Figure(data=go.Heatmap(
                    z=norm_vals,
                    x=metrics_cols,
                    y=metrics_avg['predicted_risk'].tolist(),
                    colorscale=[[0, "#e8f5f0"], [0.5, "#1a9e78"], [1, "#0d6e56"]],
                    text=heat_vals.round(1),
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    hovertemplate="Risk: %{y}<br>Metric: %{x}<br>Avg: %{text}<extra></extra>",
                    showscale=True,
                    colorbar=dict(thickness=12, tickfont=dict(size=9),
                                  title=dict(text="Rel.", side="right"))
                ))
                fig_avg_heat.update_layout(
                    **PLOT_LAYOUT, height=310,
                    xaxis=dict(title=""),
                    yaxis=dict(title="Risk Level"),
                )
                st.plotly_chart(fig_avg_heat, use_container_width=True, config={"displayModeBar": False})

            with ch6:
                st.markdown('<div class="chart-label">BMI Distribution by Risk Level</div>', unsafe_allow_html=True)
                fig_bmi_box = go.Figure()
                # Binary: only Low and High box plots
                for risk_label, colour in [("Low", "#1a9e78"), ("High", "#e05252")]:
                    grp = df_filtered[df_filtered['predicted_risk'] == risk_label]['bmi']
                    if not grp.empty:
                        fig_bmi_box.add_trace(go.Box(
                            y=grp, name=risk_label,
                            marker_color=colour,
                            boxmean='sd',
                            hovertemplate=f"<b>{risk_label}</b><br>BMI: %{{y:.1f}}<extra></extra>"
                        ))
                fig_bmi_box.update_layout(
                    **PLOT_LAYOUT, height=310, showlegend=False,
                    xaxis=dict(title="Risk Level", gridcolor="rgba(0,0,0,0)"),
                    yaxis=dict(title="BMI", gridcolor="#e0f2ea", zeroline=False),
                )
                st.plotly_chart(fig_bmi_box, use_container_width=True, config={"displayModeBar": False})

            st.markdown('</div>', unsafe_allow_html=True)

            # ── 6. Full Data Table ──────────────────────────
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🗃️ All Stored Predictions</div>', unsafe_allow_html=True)

            show_cols = ['id', 'timestamp', 'age', 'bmi', 'exercise', 'sleep',
                         'sugar_intake', 'alcohol', 'smoking', 'married',
                         'profession', 'predicted_risk']
            st.dataframe(
                df_filtered[show_cols].sort_values('id', ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=340,
                hide_index=True,
                column_config={
                    "id":             st.column_config.NumberColumn("ID"),
                    "timestamp":      st.column_config.TextColumn("Timestamp"),
                    "predicted_risk": st.column_config.TextColumn("Risk Level"),
                    "bmi":   st.column_config.NumberColumn("BMI",     format="%.1f"),
                    "age":   st.column_config.NumberColumn("Age"),
                    "exercise": st.column_config.ProgressColumn("Exercise", min_value=0, max_value=7, format="%d d/wk"),
                    "sleep":    st.column_config.ProgressColumn("Sleep",    min_value=0, max_value=12, format="%d h"),
                    "sugar_intake": st.column_config.NumberColumn("Sugar"),
                    "alcohol":      st.column_config.NumberColumn("Alcohol"),
                    "smoking":      st.column_config.NumberColumn("Smoking"),
                    "married":      st.column_config.NumberColumn("Married"),
                    "profession":   st.column_config.TextColumn("Profession"),
                }
            )

            csv_data = df_filtered[show_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️  Download as CSV",
                data=csv_data,
                file_name="health_predictions.csv",
                mime="text/csv",
            )

            st.markdown('</div>', unsafe_allow_html=True)