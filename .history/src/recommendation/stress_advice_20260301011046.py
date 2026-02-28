def get_stress_advice(features, stress_level):

    advice = []

    if features["sleep_hours"] < 6:
        advice.append("Increase sleep duration to improve recovery.")

    if features["workload_hours"] > 8:
        advice.append("Break workload into smaller focused sessions.")

    if features["physical_activity"] < 20:
        advice.append("Add light physical activity or walking daily.")

    if features["screen_time"] > 8:
        advice.append("Reduce continuous screen exposure with short breaks.")

    if features["sad"] > 0.3:
        advice.append("Take short relaxation or breathing breaks during work.")

    if stress_level == 2:
        advice.append("Consider talking to a mentor or counselor if stress persists.")

    if len(advice) == 0:
        advice.append("Maintain current lifestyle balance.")

    return advice