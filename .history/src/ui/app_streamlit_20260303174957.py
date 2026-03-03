import streamlit as st
import plotly.graph_objects as go
import time

from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Stress Intelligence System",
    layout="wide"
)

# ===============================
# GLOBAL STYLING
# ===============================
st.markdown("""
<style>
body {
    background-color: #0a0f1c;
}

.hero-title {
    font-size:60px;
    font-weight:900;
    color:#00f0ff;
    margin-bottom:0px;
}

.subtitle {
    font-size:20px;
    color:#9aa4b2;
    margin-top:-10px;
    margin-bottom:40px;
}

.card {
    background: #111827;
    padding:30px;
    border-radius:16px;
    margin-bottom:30px;
    box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
}

.section-title {
    font-size:28px;
    font-weight:700;
    margin-bottom:20px;
}

.explain-text {
    font-size:16px;
    color:#cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# HERO HEADER
# ===============================
st.markdown('<div class="hero-title">AI Stress Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid ML + LLM Cognitive Analysis System</div>', unsafe_allow_html=True)

# ===============================
# INPUT SECTION
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">Routine Data</div>', unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
    sleep_quality = st.slider("Sleep Quality", 1, 5, 3)
    workload_hours = st.slider("Workload Hours", 0.0, 12.0, 6.0)
    assignment_pressure = st.slider("Assignment Pressure", 1, 5, 3)
    physical_activity = st.slider("Physical Activity (mins)", 0, 120, 30)
    screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 5.0)
    social_interaction = st.slider("Social Interaction (hours)", 0.0, 6.0, 2.0)
    caffeine_intake = st.slider("Caffeine Intake", 0, 6, 1)

with col2:
    st.markdown('<div class="section-title">Facial AI Scan</div>', unsafe_allow_html=True)
    run_scan = st.button("Start AI Face Scan")

st.markdown('</div>', unsafe_allow_html=True)


# ===============================
# RUN SYSTEM
# ===============================
if run_scan:

    with st.spinner("Scanning face..."):
        live_data = capture_session(5)
        avg_face = aggregate_features(live_data)

    routine = {
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "workload_hours": workload_hours,
        "assignment_pressure": assignment_pressure,
        "physical_activity": physical_activity,
        "screen_time": screen_time,
        "social_interaction": social_interaction,
        "caffeine_intake": caffeine_intake
    }

    final_input = {**routine, **avg_face}
    result = predict_stress(final_input)

    # ===============================
    # STRESS GAUGE
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Stress Level</div>', unsafe_allow_html=True)

    stress_map = {"LOW": 25, "MODERATE": 60, "HIGH": 90}
    gauge_value = stress_map[result["stress_level"]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': result["stress_level"]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if gauge_value > 70 else "orange" if gauge_value > 40 else "green"},
            'steps': [
                {'range': [0, 40], 'color': "#14532d"},
                {'range': [40, 70], 'color': "#78350f"},
                {'range': [70, 100], 'color': "#7f1d1d"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="explain-text">
    The predicted stress level is based on behavioral routine patterns and
    facial signal metrics analyzed by a Random Forest classification model.
    Feature influence was computed using SHAP explainability to determine
    how each factor contributed to this prediction.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # INFLUENCE ANALYSIS (UPGRADED)
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Influence Breakdown</div>', unsafe_allow_html=True)

    for feature, value in result["top_factors"]:

        impact_percent = abs(value) * 100
        direction = "Increased Stress" if value > 0 else "Reduced Stress"
        color = "#ef4444" if value > 0 else "#22c55e"

        st.markdown(f"""
        <div style="margin-bottom:15px;">
            <strong style="color:{color}; font-size:18px;">{feature}</strong><br>
            <span style="color:{color};">
            Impact: {impact_percent:.2f}% — {direction}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(impact_percent / 100, 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # FACIAL SUMMARY
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Facial Analysis Summary</div>', unsafe_allow_html=True)

    st.write(f"Eye Ratio: {avg_face['eye_ratio']:.2f}")
    st.write(f"Neutral Emotion Probability: {avg_face['neutral']:.2f}")
    st.write(f"Happy Emotion Probability: {avg_face['happy']:.2f}")
    st.write(f"Sad Emotion Probability: {avg_face['sad']:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # AI REPORT (WITH LOADING IMPROVEMENT)
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Cognitive Report</div>', unsafe_allow_html=True)

    placeholder = st.empty()
    placeholder.info("Generating cognitive analysis...")

    time.sleep(0.5)

    placeholder.success(result["ai_report"])

    st.markdown('</div>', unsafe_allow_html=True)