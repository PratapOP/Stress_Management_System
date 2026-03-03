import streamlit as st
import plotly.graph_objects as go

from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress
from src.utils.pdf_report import generate_pdf


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="AI Stress Intelligence System",
    layout="wide"
)

st.markdown("""
<style>
.hero-title {
    font-size:70px;
    font-weight:900;
    color:#00f0ff;
    margin-bottom:10px;
}
.subtitle {
    font-size:22px;
    color:#9aa4b2;
    margin-bottom:40px;
}
.card {
    background: #111827;
    padding:30px;
    border-radius:16px;
    margin-bottom:30px;
}
.section-title {
    font-size:28px;
    font-weight:700;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# HEADER
# ===============================
st.markdown('<div class="hero-title">AI Stress Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalized Hybrid ML + LLM Cognitive Analysis</div>', unsafe_allow_html=True)

# ===============================
# USER INFO
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">User Information</div>', unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    name = st.text_input("Full Name")

with colB:
    age = st.number_input("Age", min_value=10, max_value=100, step=1)

st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# ROUTINE INPUT
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Routine Data</div>', unsafe_allow_html=True)

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
sleep_quality = st.slider("Sleep Quality", 1, 5, 3)
workload_hours = st.slider("Workload Hours", 0.0, 12.0, 6.0)
assignment_pressure = st.slider("Assignment Pressure", 1, 5, 3)
physical_activity = st.slider("Physical Activity (mins)", 0, 120, 30)
screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 5.0)
social_interaction = st.slider("Social Interaction (hours)", 0.0, 6.0, 2.0)
caffeine_intake = st.slider("Caffeine Intake", 0, 6, 1)

st.markdown('</div>', unsafe_allow_html=True)

run_scan = st.button("Start AI Face Scan & Generate Report")

# ===============================
# EXECUTION
# ===============================
if run_scan:

    if not name:
        st.error("Please enter your name before generating the report.")
        st.stop()

    with st.spinner("Capturing facial signals..."):
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

    result = predict_stress(final_input, name, age)

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
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===============================
    # AI REPORT
    # ===============================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Cognitive Report</div>', unsafe_allow_html=True)

    st.success(result["ai_report"])

    # ===============================
    # PDF DOWNLOAD
    # ===============================
    pdf_buffer = generate_pdf(
        name,
        age,
        result["stress_level"],
        result["ai_report"]
    )

    st.download_button(
        label="Download Personalized AI Stress Report (PDF)",
        data=pdf_buffer,
        file_name=f"{name}_Stress_Report.pdf",
        mime="application/pdf"
    )

    st.markdown('</div>', unsafe_allow_html=True)