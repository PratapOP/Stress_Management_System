import streamlit as st
import plotly.graph_objects as go

from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress
from src.utils.pdf_report import generate_pdf


st.set_page_config(page_title="AI Stress Intelligence", layout="wide")

st.markdown("<h1 style='font-size:60px;'>AI Stress Intelligence System</h1>", unsafe_allow_html=True)

# ================= USER INFO =================
st.markdown("### User Information")
colA, colB = st.columns(2)

with colA:
    name = st.text_input("Enter Your Name")

with colB:
    age = st.number_input("Enter Your Age", min_value=10, max_value=100, step=1)

st.divider()

# ================= ROUTINE INPUT =================
st.markdown("### Routine Data")

sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
sleep_quality = st.slider("Sleep Quality", 1, 5, 3)
workload_hours = st.slider("Workload Hours", 0.0, 12.0, 6.0)
assignment_pressure = st.slider("Assignment Pressure", 1, 5, 3)
physical_activity = st.slider("Physical Activity (mins)", 0, 120, 30)
screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 5.0)
social_interaction = st.slider("Social Interaction (hours)", 0.0, 6.0, 2.0)
caffeine_intake = st.slider("Caffeine Intake", 0, 6, 1)

st.divider()

run_scan = st.button("Start AI Face Scan & Analysis")

if run_scan and name:

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

    # ===== Stress Gauge =====
    st.markdown("## Stress Level")

    stress_map = {"LOW": 25, "MODERATE": 60, "HIGH": 90}
    gauge_value = stress_map[result["stress_level"]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': result["stress_level"]},
        gauge={'axis': {'range': [0, 100]}}
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ===== AI Report =====
    st.markdown("## AI Cognitive Report")
    st.success(result["ai_report"])

    # ===== PDF Download =====
    pdf_buffer = generate_pdf(
        name,
        age,
        result["stress_level"],
        result["ai_report"]
    )

    st.download_button(
        label="Download AI Stress Report (PDF)",
        data=pdf_buffer,
        file_name=f"{name}_Stress_Report.pdf",
        mime="application/pdf"
    )