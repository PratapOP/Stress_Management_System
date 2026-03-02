from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features
from src.models.predict import predict_stress


def get_routine_input():

    print("\nEnter Daily Routine Data:")

    return {
        "sleep_hours": float(input("Sleep hours: ")),
        "sleep_quality": int(input("Sleep quality (1-5): ")),
        "workload_hours": float(input("Workload hours: ")),
        "assignment_pressure": int(input("Assignment pressure (1-5): ")),
        "physical_activity": float(input("Physical activity (minutes): ")),
        "screen_time": float(input("Screen time (hours): ")),
        "social_interaction": float(input("Social interaction (hours): ")),
        "caffeine_intake": int(input("Caffeine intake (cups): "))
    }


def main():

    print("===== LIVE STRESS MANAGEMENT SYSTEM =====")

    # ---------- ROUTINE ----------
    routine = get_routine_input()

    # ---------- LIVE FACE ----------
    live_data = capture_session(5)

    avg_face = aggregate_features(live_data)

    # ---------- MERGE FEATURES ----------
    final_input = {**routine, **avg_face}

    # ---------- PREDICT ----------
    result = predict_stress(final_input)

    print("\n===== FINAL RESULT =====")
    print("Stress Level:", result["stress_level"])

    print("\nTop 5 Affecting Factors:")
    for f, v in result["top_factors"]:
        print(f"- {f}")

    print("\nAdvice:")
    for a in result["advice"]:
        print("-", a)


if __name__ == "__main__":
    main()