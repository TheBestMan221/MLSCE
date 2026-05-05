"""
app.py  –  F1 Race Prediction & Strategy Recommendation System
Run:  streamlit run app.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 ML Prediction System",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS  ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e8e8;
    font-family: 'Rajdhani', sans-serif;
}
h1, h2, h3 { font-family: 'Orbitron', monospace; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #15001a 100%);
    border-right: 2px solid #e10600;
}
section[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

/* metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #12121e, #1a0a0a);
    border: 1px solid #e10600;
    border-radius: 8px;
    padding: 12px;
}
div[data-testid="metric-container"] label { color: #ffd700 !important; font-family: 'Rajdhani'; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Orbitron'; font-size: 1.4rem !important; }

/* buttons */
div.stButton > button {
    background: linear-gradient(90deg, #e10600, #ff4444);
    color: white;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 2rem;
    transition: all 0.2s;
}
div.stButton > button:hover { background: linear-gradient(90deg, #ff4444, #e10600); transform: scale(1.02); }

/* select boxes */
div[data-baseweb="select"] > div { background-color: #12121e !important; border-color: #e10600 !important; color: #e8e8e8 !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #0f0f1a; border-bottom: 2px solid #e10600; }
.stTabs [data-baseweb="tab"] { font-family: 'Orbitron'; color: #888; }
.stTabs [aria-selected="true"] { color: #e10600 !important; border-bottom: 2px solid #e10600; }

/* divider */
hr { border-color: #e10600; }

/* header banner */
.f1-header {
    background: linear-gradient(90deg, #0a0a0f 0%, #1a0505 50%, #0a0a0f 100%);
    border-top: 3px solid #e10600;
    border-bottom: 3px solid #e10600;
    padding: 1rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.f1-header h1 { font-size: 2rem; color: #fff; margin: 0; letter-spacing: 3px; }
.f1-header p { color: #ffd700; margin: 0; font-size: 1rem; }

.card {
    background: linear-gradient(135deg, #12121e 0%, #0f0a14 100%);
    border: 1px solid #2a2a3a;
    border-left: 4px solid #e10600;
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.card h4 { color: #ffd700; font-family: 'Orbitron'; font-size: 0.85rem; margin-bottom: 0.5rem; }
.card p { color: #ccc; margin: 0; }

.badge-red { background:#e10600; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-family:'Orbitron'; }
.badge-gold { background:#ffd700; color:#000; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-family:'Orbitron'; }
.badge-green { background:#00c851; color:#000; padding:2px 10px; border-radius:12px; font-size:0.8rem; }

.strategy-box {
    background: #0f0f1a;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── load helpers (cached) ────────────────────────────────────────────────────
MODELS_TRAINED = os.path.exists(os.path.join("models", "meta.json"))

@st.cache_data
def get_meta():
    if not MODELS_TRAINED:
        return None
    from predictor import load_meta
    return load_meta()

@st.cache_data
def get_metrics():
    if not MODELS_TRAINED:
        return None
    from predictor import load_metrics
    return load_metrics()

@st.cache_data
def get_fi():
    if not MODELS_TRAINED:
        return None
    from predictor import load_feature_importance
    return load_feature_importance()

@st.cache_data
def get_processed():
    path = os.path.join("data", "processed", "master.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ── plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(10,10,15,0)",
    plot_bgcolor="rgba(10,10,15,0)",
    font=dict(color="#e8e8e8", family="Rajdhani"),
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#333"),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#333"),
    margin=dict(l=30, r=30, t=40, b=30),
)

# ── sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <div style="font-family:'Orbitron';font-size:1.2rem;color:#e10600;font-weight:900;">🏎️ F1 ML</div>
        <div style="font-family:'Rajdhani';font-size:0.8rem;color:#888;">PREDICTION SYSTEM</div>
        <hr style="border-color:#e10600;margin:10px 0;">
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        [
            "🏠 Home Dashboard",
            "🏁 Race Prediction",
            "🥇 Pole Prediction",
            "🎲 Race Simulation",
            "🛞 Strategy",
            "👤 Driver Analysis",
            "🏭 Team Analysis",
            "📊 Model Performance",
        ],
        label_visibility="visible",
    )

    if not MODELS_TRAINED:
        st.error("⚠️ Models not trained!\nRun:\n```\npython src/train_models.py\n```")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="f1-header">
        <h1>🏎️ F1 RACE PREDICTION SYSTEM</h1>
        <p>Machine Learning · Strategy Recommendation · Monte Carlo Simulation</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    df = get_processed()
    meta = get_meta()

    with col1:
        races_count = df["raceId"].nunique() if df is not None else "–"
        st.metric("Total Races", races_count)
    with col2:
        drv_count = len(meta["drivers"]) if meta else "–"
        st.metric("Drivers", drv_count)
    with col3:
        team_count = len(meta["constructors"]) if meta else "–"
        st.metric("Teams", team_count)
    with col4:
        circ_count = len(meta["circuits"]) if meta else "–"
        st.metric("Circuits", circ_count)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🤖 ML Models Deployed")
        models_info = [
            ("Logistic Regression", "Classification", "#e10600"),
            ("Decision Tree", "Classification", "#e10600"),
            ("Random Forest", "Classification", "#e10600"),
            ("SVM", "Classification", "#e10600"),
            ("Naive Bayes", "Classification", "#e10600"),
            ("Linear Regression", "Regression", "#ffd700"),
            ("Ridge Regression", "Regression", "#ffd700"),
            ("Lasso Regression", "Regression", "#ffd700"),
            ("K-Means", "Clustering", "#00c8ff"),
        ]
        for name, mtype, color in models_info:
            st.markdown(
                f'<div class="card"><h4>{name}</h4><p><span style="color:{color}">▶ {mtype}</span></p></div>',
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("### 📋 Project Workflow")
        steps = [
            ("1", "Load Raw CSVs", "results, races, drivers, qualifying, pit_stops, lap_times"),
            ("2", "Merge & Filter", "Year ≥ 2000 · 8 tables merged"),
            ("3", "Clean & Impute", "Handle missing values · Encode categoricals"),
            ("4", "Feature Engineering", "Rolling avgs · pit counts · qual positions"),
            ("5", "Train/Test Split", "80% train · 20% test · Standardization"),
            ("6", "Train Models", "9 ML models trained & saved"),
            ("7", "Evaluate", "Accuracy · F1 · MAE · R² per model"),
            ("8", "Predict & Simulate", "Web UI for interactive prediction"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="card">
                <h4><span style="color:#e10600">STEP {num}</span> · {title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        if df is not None and MODELS_TRAINED:
            metrics = get_metrics()
            if metrics:
                rf_acc = metrics.get("Random Forest", {}).get("accuracy", 0)
                rf_f1 = metrics.get("Random Forest", {}).get("f1", 0)
                ridge_r2 = metrics.get("Ridge Regression", {}).get("r2", 0)
                st.markdown("### 🏆 Best Model Highlights")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RF Accuracy", f"{rf_acc:.1%}")
                mc2.metric("RF F1 Score", f"{rf_f1:.4f}")
                mc3.metric("Ridge R²", f"{ridge_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RACE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏁 Race Prediction":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">🏁 Race Prediction</h2>', unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Please train models first: `python src/train_models.py`")
        st.stop()

    from predictor import build_input_vector, predict_top10, predict_position, predict_all_models

    meta = get_meta()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Configure Race Entry")
        driver = st.selectbox("Driver", meta["drivers"])
        team = st.selectbox("Team / Constructor", meta["constructors"])
        circuit = st.selectbox("Circuit", meta["circuits"])
        grid = st.slider("Grid Position", 1, 20, 5)
        qual = st.slider("Qualifying Position", 1, 20, 5)
        year = st.selectbox("Year", sorted(meta["years"], reverse=True))
        driver_avg = st.slider("Driver Avg Finish (recent)", 1.0, 20.0, 8.0, step=0.5)
        team_avg = st.slider("Team Avg Finish (recent)", 1.0, 20.0, 8.0, step=0.5)
        pit_cnt = st.slider("Expected Pit Stops", 1, 4, 2)
        avg_lap = st.number_input("Avg Lap Time (ms)", 60000, 130000, 90000, step=1000)

        predict_btn = st.button("🏎️ PREDICT")

    with col2:
        if predict_btn:
            driver_ref = meta["driverRef_map"].get(driver, driver.lower())
            constructor_ref = meta["constructorRef_map"].get(team, team.lower())
            circuit_ref = meta["circuitRef_map"].get(circuit, circuit.lower())

            x = build_input_vector(
                grid=grid,
                qual_position=qual,
                year=year,
                driver_ref=driver_ref,
                constructor_ref=constructor_ref,
                circuit_ref=circuit_ref,
                driver_avg_finish=driver_avg,
                team_avg_finish=team_avg,
                pit_stop_count=float(pit_cnt),
                avg_lap_ms=float(avg_lap),
            )

            top10 = predict_top10(x, "Random_Forest")
            pos = predict_position(x, "Ridge_Regression")
            all_m = predict_all_models(x)

            prob = top10["probability"]
            pred_pos = pos["predicted_position"]

            # headline metrics
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Top 10 Probability", f"{prob:.1%}")
            hc2.metric("Predicted Position", f"P{int(round(pred_pos))}")
            hc3.metric("Grid → Predicted", f"P{grid} → P{int(round(pred_pos))}")

            # gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Top 10 Finish Probability (%)", "font": {"family": "Orbitron", "color": "#ffd700"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#888"},
                    "bar": {"color": "#e10600"},
                    "bgcolor": "#1a1a2e",
                    "steps": [
                        {"range": [0, 40], "color": "#2a0000"},
                        {"range": [40, 70], "color": "#1a1a00"},
                        {"range": [70, 100], "color": "#001a00"},
                    ],
                    "threshold": {"line": {"color": "#ffd700", "width": 3}, "value": 50},
                },
                number={"suffix": "%", "font": {"color": "#ffffff", "family": "Orbitron"}},
            ))
            fig_gauge.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # all classifiers comparison
            st.markdown("#### 📊 All Classifier Predictions (Top 10 Probability)")
            cls_data = all_m["classification"]
            cls_names = [n.replace("_", " ") for n in cls_data]
            cls_probs = [v.get("top10_prob", 0) for v in cls_data.values()]
            colors = ["#e10600" if p >= 0.5 else "#555" for p in cls_probs]

            fig_cls = go.Figure(go.Bar(
                x=cls_names, y=[p * 100 for p in cls_probs],
                marker_color=colors, text=[f"{p:.1%}" for p in cls_probs],
                textposition="outside",
            ))
            fig_cls.update_layout(**PLOTLY_LAYOUT, yaxis_title="Probability (%)", height=300)
            st.plotly_chart(fig_cls, use_container_width=True)

            # regressors
            st.markdown("#### 📈 All Regressor Predictions (Finishing Position)")
            reg_data = all_m["regression"]
            reg_names = [n.replace("_", " ") for n in reg_data]
            reg_pos = [v.get("predicted_position", 10) for v in reg_data.values()]

            fig_reg = go.Figure(go.Bar(
                x=reg_names, y=reg_pos,
                marker_color="#ffd700",
                text=[f"P{p:.1f}" for p in reg_pos],
                textposition="outside",
            ))
            fig_reg.update_layout(**PLOTLY_LAYOUT, yaxis_title="Predicted Position", yaxis_autorange="reversed", height=280)
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("👈 Configure race entry on the left and click **PREDICT**")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: POLE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🥇 Pole Prediction":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">🥇 Pole Position Prediction</h2>', unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Please train models first.")
        st.stop()

    from predictor import build_input_vector, load_meta
    import joblib

    meta = get_meta()

    st.markdown("Pole probability is estimated using the Random Forest classifier with qualifying-centric inputs.")

    col1, col2 = st.columns([1, 2])
    with col1:
        driver = st.selectbox("Driver", meta["drivers"], key="pole_driver")
        team = st.selectbox("Team", meta["constructors"], key="pole_team")
        circuit = st.selectbox("Circuit", meta["circuits"], key="pole_circuit")
        year = st.selectbox("Year", sorted(meta["years"], reverse=True), key="pole_year")
        driver_avg = st.slider("Driver Avg Finish", 1.0, 20.0, 5.0, key="pole_avg")
        pole_btn = st.button("🥇 PREDICT POLE PROBABILITY")

    with col2:
        if pole_btn:
            driver_ref = meta["driverRef_map"].get(driver, driver.lower())
            constructor_ref = meta["constructorRef_map"].get(team, team.lower())
            circuit_ref = meta["circuitRef_map"].get(circuit, circuit.lower())

            # For pole: grid=1, qual=1 as target scenario; vary qual 1–5 to show sensitivity
            results = []
            model = joblib.load(os.path.join("models", "Random_Forest.pkl"))
            sc = joblib.load(os.path.join("models", "scaler.pkl"))

            for q in range(1, 6):
                x = build_input_vector(1, q, year, driver_ref, constructor_ref, circuit_ref,
                                       driver_avg_finish=driver_avg)
                x_s = sc.transform(x)
                p = float(model.predict_proba(x_s)[0][1])
                results.append({"Qual Position": q, "Top10 Probability": round(p, 4)})

            df_pole = pd.DataFrame(results)

            # pole prob ≈ top10 prob when qual=1 boosted
            pole_prob = float(df_pole[df_pole["Qual Position"] == 1]["Top10 Probability"])

            pc1, pc2 = st.columns(2)
            pc1.metric("Pole/Front Row Probability", f"{pole_prob:.1%}")
            pc2.metric("Driver", driver)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pole["Qual Position"], y=df_pole["Top10 Probability"] * 100,
                mode="lines+markers",
                line=dict(color="#e10600", width=3),
                marker=dict(size=10, color="#ffd700"),
                name="Top10 Prob",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Probability vs Qualifying Position",
                xaxis_title="Qualifying Position",
                yaxis_title="Top 10 Probability (%)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_pole, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RACE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎲 Race Simulation":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">🎲 Monte Carlo Race Simulation</h2>', unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Please train models first.")
        st.stop()

    from predictor import build_input_vector, load_meta
    from simulator import run_simulation
    import joblib

    meta = get_meta()
    model = joblib.load(os.path.join("models", "Random_Forest.pkl"))
    sc = joblib.load(os.path.join("models", "scaler.pkl"))

    st.markdown("Select up to 20 drivers, choose a circuit, and run the simulation.")

    col1, col2 = st.columns([1, 2])
    with col1:
        circuit = st.selectbox("Circuit", meta["circuits"], key="sim_circ")
        year = st.selectbox("Year", sorted(meta["years"], reverse=True), key="sim_year")
        n_sims = st.select_slider("Simulations", [100, 250, 500, 1000], value=500)
        selected_drivers = st.multiselect(
            "Select Drivers (2–20)",
            meta["drivers"],
            default=meta["drivers"][:8],
        )
        sim_btn = st.button("🎲 RUN SIMULATION")

    with col2:
        if sim_btn:
            if len(selected_drivers) < 2:
                st.error("Select at least 2 drivers.")
            else:
                circuit_ref = meta["circuitRef_map"].get(circuit, "bahrain")
                probs = []
                for i, drv in enumerate(selected_drivers):
                    drv_ref = meta["driverRef_map"].get(drv, drv.lower())
                    x = build_input_vector(i + 1, i + 1, year, drv_ref, "ferrari", circuit_ref)
                    x_s = sc.transform(x)
                    p = float(model.predict_proba(x_s)[0][1])
                    probs.append(p)

                sim = run_simulation(selected_drivers, probs, n_sims=n_sims)
                results = sim["results"]
                df_sim = pd.DataFrame(results)

                st.success(f"🏆 Predicted Winner: **{sim['winner']}**")

                sc1, sc2, sc3 = st.columns(3)
                winner = results[0]
                sc1.metric("Win Probability", f"{winner['win_prob']:.1%}")
                sc2.metric("Podium Probability", f"{winner['podium_prob']:.1%}")
                sc3.metric("Simulations Run", f"{n_sims:,}")

                # win prob bar chart
                fig_win = go.Figure(go.Bar(
                    x=df_sim["driver"],
                    y=df_sim["win_prob"] * 100,
                    marker_color=["#e10600" if i == 0 else "#ffd700" if i < 3 else "#555"
                                  for i in range(len(df_sim))],
                    text=[f"{v:.1%}" for v in df_sim["win_prob"]],
                    textposition="outside",
                ))
                fig_win.update_layout(**PLOTLY_LAYOUT, title="Win Probability (%)", height=350,
                                      xaxis_tickangle=-30, yaxis_title="Win %")
                st.plotly_chart(fig_win, use_container_width=True)

                # podium vs top10 comparison
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(name="Podium %", x=df_sim["driver"],
                                          y=df_sim["podium_prob"] * 100, marker_color="#e10600"))
                fig_comp.add_trace(go.Bar(name="Top 10 %", x=df_sim["driver"],
                                          y=df_sim["top10_prob"] * 100, marker_color="#ffd700"))
                fig_comp.update_layout(**PLOTLY_LAYOUT, barmode="group", height=320,
                                       xaxis_tickangle=-30, title="Podium vs Top 10 Probability")
                st.plotly_chart(fig_comp, use_container_width=True)

                # full table
                st.markdown("#### Simulated Finishing Order")
                df_display = df_sim[["sim_position", "driver", "win_prob", "podium_prob", "top10_prob", "avg_finish"]].copy()
                df_display.columns = ["Position", "Driver", "Win %", "Podium %", "Top10 %", "Avg Finish"]
                df_display["Win %"] = df_display["Win %"].apply(lambda x: f"{x:.1%}")
                df_display["Podium %"] = df_display["Podium %"].apply(lambda x: f"{x:.1%}")
                df_display["Top10 %"] = df_display["Top10 %"].apply(lambda x: f"{x:.1%}")
                st.dataframe(df_display, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛞 Strategy":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">🛞 Race Strategy Recommendation</h2>', unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Please train models first.")
        st.stop()

    from strategy import recommend
    meta = get_meta()

    col1, col2 = st.columns([1, 2])
    with col1:
        circuit = st.selectbox("Circuit", meta["circuits"], key="strat_circ")
        grid = st.slider("Grid Position", 1, 20, 5, key="strat_grid")
        avg_lap = st.number_input("Avg Lap Time (ms)", 60000, 130000, 90000, step=1000, key="strat_lap")
        strat_btn = st.button("🛞 RECOMMEND STRATEGY")

    with col2:
        if strat_btn:
            circuit_ref = meta["circuitRef_map"].get(circuit, "bahrain")
            rec = recommend(circuit_ref, grid, float(avg_lap))

            st.markdown(f"""
            <div class="strategy-box">
                <h3 style="font-family:Orbitron;color:#e10600;">📋 Strategy for {circuit}</h3>
                <hr style="border-color:#333;">
                <table style="width:100%;font-family:Rajdhani;font-size:1.1rem;">
                    <tr>
                        <td style="color:#888;padding:6px 0;">Recommended Pit Stops</td>
                        <td style="color:#ffd700;font-weight:700;font-size:1.4rem;">{rec['recommended_pit_stops']}</td>
                    </tr>
                    <tr>
                        <td style="color:#888;padding:6px 0;">Tyre Strategy</td>
                        <td style="color:#e10600;font-weight:600;">{rec['tyre_strategy']}</td>
                    </tr>
                    <tr>
                        <td style="color:#888;padding:6px 0;">Pit Windows</td>
                        <td style="color:#fff;">{'  |  '.join(rec['pit_windows'])}</td>
                    </tr>
                    <tr>
                        <td style="color:#888;padding:6px 0;">Historical Avg Stops</td>
                        <td style="color:#aaa;">{rec['avg_historical_stops']}</td>
                    </tr>
                    <tr>
                        <td style="color:#888;padding:6px 0;">Pace Category</td>
                        <td style="color:#aaa;">{rec['pace_category']}</td>
                    </tr>
                </table>
                <hr style="border-color:#333;">
                <p style="color:#ccc;font-style:italic;">💡 {rec['circuit_note']}</p>
            </div>
            """, unsafe_allow_html=True)

            # tyre compound visual
            tyres = rec["tyre_strategy"].split(" → ")
            tyre_colors = {
                "Soft": "#e10600", "Medium": "#ffd700", "Hard": "#ffffff",
                "Intermediate": "#00c851", "Wet": "#0077ff",
            }
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(len(tyres) * 2 - 1)
            for i, tyre in enumerate(tyres):
                color = tyre_colors.get(tyre, "#888")
                cols[i * 2].markdown(
                    f'<div style="text-align:center;padding:20px 10px;background:{color};'
                    f'color:{"#000" if tyre != "Soft" and tyre != "Hard" else "#fff" if tyre == "Soft" else "#000"};'
                    f'border-radius:50%;font-family:Orbitron;font-size:0.75rem;font-weight:700;">{tyre}</div>',
                    unsafe_allow_html=True,
                )
                if i < len(tyres) - 1:
                    cols[i * 2 + 1].markdown(
                        '<div style="text-align:center;font-size:1.5rem;padding-top:15px;color:#888;">→</div>',
                        unsafe_allow_html=True,
                    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DRIVER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Driver Analysis":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">👤 Driver Analysis</h2>', unsafe_allow_html=True)

    df = get_processed()
    if df is None:
        st.warning("Processed data not found. Run training first.")
        st.stop()

    meta = get_meta()
    driver = st.selectbox("Select Driver", meta["drivers"])

    driver_ref = meta["driverRef_map"].get(driver, driver.lower())
    ddf = df[df["driverRef"] == driver_ref].copy()

    if ddf.empty:
        st.warning("No data found for this driver.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Races", len(ddf))
    col2.metric("Avg Finish", f"{ddf['positionOrder'].mean():.1f}")
    col3.metric("Wins", int((ddf["positionOrder"] == 1).sum()))
    col4.metric("Podiums", int((ddf["positionOrder"] <= 3).sum()))

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Season Performance", "🏟️ Circuit Performance", "🏎️ Raw Data"])

    with tab1:
        season = ddf.groupby("year").agg(
            avg_finish=("positionOrder", "mean"),
            races=("raceId", "count"),
            wins=("positionOrder", lambda x: (x == 1).sum()),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=season["year"], y=season["avg_finish"],
                                  mode="lines+markers", name="Avg Finish",
                                  line=dict(color="#e10600", width=2)), secondary_y=False)
        fig.add_trace(go.Bar(x=season["year"], y=season["wins"],
                              name="Wins", marker_color="#ffd700", opacity=0.6), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, title=f"{driver} – Season Performance",
                          height=380, xaxis_title="Year")
        fig.update_yaxes(title_text="Avg Finish Position", secondary_y=False, autorange="reversed")
        fig.update_yaxes(title_text="Wins", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        circ = ddf.groupby("circuit_name").agg(
            avg_finish=("positionOrder", "mean"),
            races=("raceId", "count"),
        ).reset_index().sort_values("avg_finish")
        fig2 = px.bar(circ.head(15), x="circuit_name", y="avg_finish",
                       color="avg_finish", color_continuous_scale=["#00c851", "#ffd700", "#e10600"],
                       title="Avg Finishing Position per Circuit (lower = better)")
        fig2.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_tickangle=-30, yaxis_autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(
            ddf[["year", "race_name", "grid", "qual_position", "positionOrder", "points", "pit_stop_count"]].sort_values(["year", "race_name"]),
            use_container_width=True, hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Team Analysis":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">🏭 Constructor / Team Analysis</h2>', unsafe_allow_html=True)

    df = get_processed()
    if df is None:
        st.warning("Processed data not found. Run training first.")
        st.stop()

    meta = get_meta()
    team = st.selectbox("Select Team", meta["constructors"])
    con_ref = meta["constructorRef_map"].get(team, team.lower())
    tdf = df[df["constructorRef"] == con_ref].copy()

    if tdf.empty:
        st.warning("No data found for this team.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", len(tdf))
    col2.metric("Avg Finish", f"{tdf['positionOrder'].mean():.1f}")
    col3.metric("Wins", int((tdf["positionOrder"] == 1).sum()))
    col4.metric("Podiums", int((tdf["positionOrder"] <= 3).sum()))

    st.markdown("---")

    # Season trend
    season = tdf.groupby("year").agg(
        avg_finish=("positionOrder", "mean"),
        total_points=("points", "sum"),
        wins=("positionOrder", lambda x: (x == 1).sum()),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=season["year"], y=season["avg_finish"],
                              mode="lines+markers", name="Avg Finish",
                              line=dict(color="#e10600", width=2)), secondary_y=False)
    fig.add_trace(go.Bar(x=season["year"], y=season["total_points"],
                          name="Points", marker_color="#ffd700", opacity=0.5), secondary_y=True)
    fig.update_layout(**PLOTLY_LAYOUT, title=f"{team} – Season Trend", height=380, xaxis_title="Year")
    fig.update_yaxes(title_text="Avg Finish", secondary_y=False, autorange="reversed")
    fig.update_yaxes(title_text="Total Points", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Drivers used
    st.markdown("#### Drivers Used")
    drv_stats = tdf.groupby("driver_name").agg(
        races=("raceId", "count"),
        avg_finish=("positionOrder", "mean"),
        wins=("positionOrder", lambda x: (x == 1).sum()),
    ).reset_index().sort_values("avg_finish")
    st.dataframe(drv_stats.rename(columns={"driver_name": "Driver", "races": "Races",
                                             "avg_finish": "Avg Finish", "wins": "Wins"}),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<h2 style="font-family:Orbitron;color:#e10600;">📊 Model Performance Dashboard</h2>', unsafe_allow_html=True)

    if not MODELS_TRAINED:
        st.warning("Please train models first.")
        st.stop()

    metrics = get_metrics()
    fi = get_fi()

    tab1, tab2, tab3 = st.tabs(["🎯 Classification", "📈 Regression", "🌐 Feature Importance"])

    with tab1:
        cls_models = {k: v for k, v in metrics.items() if v.get("type") == "classification"}
        names = list(cls_models.keys())
        accs = [v["accuracy"] for v in cls_models.values()]
        precs = [v["precision"] for v in cls_models.values()]
        recalls = [v["recall"] for v in cls_models.values()]
        f1s = [v["f1"] for v in cls_models.values()]

        fig = go.Figure()
        for metric, vals, color in [
            ("Accuracy", accs, "#e10600"),
            ("Precision", precs, "#ffd700"),
            ("Recall", recalls, "#00c8ff"),
            ("F1", f1s, "#00c851"),
        ]:
            fig.add_trace(go.Bar(name=metric, x=names, y=vals, marker_color=color))
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Classification Metrics",
                          yaxis_title="Score", height=400, xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

        # best model highlight
        best_name = names[f1s.index(max(f1s))]
        st.success(f"🏆 Best F1 Score: **{best_name}** — F1={max(f1s):.4f}")

        # confusion matrix for best classifier
        best_cm = cls_models[best_name]["confusion_matrix"]
        fig_cm = px.imshow(
            best_cm,
            text_auto=True,
            color_continuous_scale=["#0a0a1a", "#e10600"],
            title=f"Confusion Matrix – {best_name}",
            labels=dict(x="Predicted", y="Actual"),
            x=["Not Top10", "Top10"],
            y=["Not Top10", "Top10"],
        )
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        # table
        cls_df = pd.DataFrame([
            {"Model": k, "Accuracy": v["accuracy"], "Precision": v["precision"],
             "Recall": v["recall"], "F1": v["f1"]}
            for k, v in cls_models.items()
        ])
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

    with tab2:
        reg_models = {k: v for k, v in metrics.items() if v.get("type") == "regression"}
        reg_df = pd.DataFrame([
            {"Model": k, "MAE": v["mae"], "MSE": v["mse"], "R²": v["r2"]}
            for k, v in reg_models.items()
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

        fig_r = go.Figure()
        fig_r.add_trace(go.Bar(x=list(reg_models), y=[v["r2"] for v in reg_models.values()],
                                marker_color="#00c851", name="R²"))
        fig_r.add_trace(go.Bar(x=list(reg_models), y=[v["mae"] for v in reg_models.values()],
                                marker_color="#e10600", name="MAE"))
        fig_r.update_layout(**PLOTLY_LAYOUT, barmode="group", title="Regression Metrics", height=380)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab3:
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        fi_names = [f.replace("_", " ") for f, _ in fi_sorted]
        fi_vals = [v for _, v in fi_sorted]

        fig_fi = go.Figure(go.Bar(
            x=fi_vals, y=fi_names, orientation="h",
            marker_color=["#e10600" if v == max(fi_vals) else "#ffd700" if v > 0.1 else "#555"
                          for v in fi_vals],
        ))
        fig_fi.update_layout(
            **PLOTLY_LAYOUT,
            title="Feature Importances — Random Forest",
            xaxis_title="Importance",
            height=420,
            yaxis=dict(autorange="reversed", gridcolor="#1e1e2e"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        km = metrics.get("K-Means", {})
        if km:
            st.info(f"🔵 K-Means Clustering: k={km['k']}  |  Inertia={km['inertia']}")
