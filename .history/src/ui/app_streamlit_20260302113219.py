import streamlit as st

from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI Stress Management System",
    layout="wide"
)

# ---------- CUSTOM STYLE ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size:40px;
    font-weight:bold;
    color:#00e5ff;
}
.sub-title {
    color:#bbbbbb;
}
.stButton>button {
    background-color:#00e5ff;
    color:black;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="big-title">AI Stress Management Dashboard</p>',
            unsafe_allow_html=True)

st.markdown('<p class="sub-title">Live Multimodal Stress Detection System</p>',
            unsafe_allow_html=True)

st.divider()

# ---------- INPUT LAYOUT ----------
col1, col2 = st.columns(2)

with col1:

    st.subheader("🧠 Routine Input")

    sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 6.0)
    sleep_quality = st.slider("Sleep Quality", 1, 5, 3)
    workload_hours = st.slider("Workload Hours", 0.0, 12.0, 6.0)
    assignment_pressure = st.slider("Assignment Pressure", 1, 5, 3)
    physical_activity = st.slider("Physical Activity (mins)", 0, 120, 30)
    screen_time = st.slider("Screen Time (hours)", 0.0, 12.0, 5.0)
    social_interaction = st.slider("Social Interaction (hours)", 0.0, 6.0, 2.0)
    caffeine_intake = st.slider("Caffeine Intake", 0, 6, 1)

with col2:

    st.subheader("📷 Live Face Capture")

    st.info("Press button to capture 5-second live session.")

    run_capture = st.button("Start Live Analysis")

# ---------- RUN SYSTEM ----------
if run_capture:

    with st.spinner("Capturing live facial data..."):
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

    # ---------- RESULT ----------
    st.subheader("⚡ Stress Prediction")

    stress = result["stress_level"]

    if stress == "LOW":
        st.success(f"Stress Level: {stress}")
    elif stress == "MODERATE":
        st.warning(f"Stress Level: {stress}")
    else:
        st.error(f"Stress Level: {stress}")

    # ---------- TOP FACTORS ----------
    st.subheader("🔥 Top 5 Affecting Factors")

    for f, v in result["top_factors"]:
        st.progress(float(v))
        st.write(f"**{f}**")

    # ---------- ADVICE ----------
    st.subheader("💡 AI Recommendations")

    for advice in result["advice"]:
        st.write("•", advice)