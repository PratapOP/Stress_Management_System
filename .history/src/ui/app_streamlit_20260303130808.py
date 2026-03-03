import streamlit as st
import plotly.graph_objects as go

from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress


st.set_page_config(
    page_title="AI Stress Intelligence",
    layout="wide"
)

st.markdown("""
<style>
.big-title {
    font-size:42px;
    font-weight:800;
    color:#00f0ff;
}
.section-title {
    font-size:24px;
    font-weight:600;
    margin-top:20px;
}
.report-box {
    background-color:#111;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">AI Stress Intelligence System</p>',
            unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">Routine Data</p>',
                unsafe_allow_html=True)

    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
    sleep_quality = st.slider("Sleep Quality", 1, 5, 3)
    workload_hours = st.slider("Workload Hours", 0.0, 12.0, 6.0)
    assignment_pressure = st.slider("Assignment Pressure", 1, 5, 3)
    physical_activity = st.slider("Physical Activity (mins)", 0, 120, 30)
    screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 5.0)
    social_interaction = st.slider("Social Interaction (hours)", 0.0, 6.0, 2.0)
    caffeine_intake = st.slider("Caffeine Intake", 0, 6, 1)

with col2:
    st.markdown('<p class="section-title">Facial AI Scan</p>',
                unsafe_allow_html=True)

    run_scan = st.button("Start AI Face Scan")

if run_scan:

    with st.spinner("Scanning face with AI system..."):
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

    st.divider()

    # ---------- STRESS GAUGE ----------
    st.markdown('<p class="section-title">Stress Level</p>',
                unsafe_allow_html=True)

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
                {'range': [0, 40], 'color': "#0f5132"},
                {'range': [40, 70], 'color': "#664d03"},
                {'range': [70, 100], 'color': "#842029"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------- SHAP FACTORS ----------
    st.markdown('<p class="section-title">AI Influence Analysis</p>',
                unsafe_allow_html=True)

    for feature, value in result["top_factors"]:
        color = "green" if value < 0 else "red"
        direction = "Reducing Stress" if value < 0 else "Increasing Stress"

        st.markdown(
            f"<span style='color:{color}; font-weight:bold;'>"
            f"{feature} ({direction})"
            f"</span>",
            unsafe_allow_html=True
        )

        st.progress(min(abs(value), 1.0))

    # ---------- FACIAL SUMMARY ----------
    st.markdown('<p class="section-title">Facial Analysis Summary</p>',
                unsafe_allow_html=True)

    st.write(f"Eye Ratio: {avg_face['eye_ratio']:.2f}")
    st.write(f"Neutral Emotion: {avg_face['neutral']:.2f}")
    st.write(f"Happy Emotion: {avg_face['happy']:.2f}")
    st.write(f"Sad Emotion: {avg_face['sad']:.2f}")

    # ---------- AI REPORT ----------
    st.markdown('<p class="section-title">AI Cognitive Report</p>',
                unsafe_allow_html=True)

    st.markdown(f"<div class='report-box'>{result['ai_report']}</div>",
                unsafe_allow_html=True)

    # ---------- RECOMMENDATIONS ----------
    st.markdown('<p class="section-title">Recommendations</p>',
                unsafe_allow_html=True)

    for advice in result["advice"]:
        st.write("•", advice)