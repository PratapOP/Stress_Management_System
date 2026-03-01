from src.live.webcam_capture import capture_session
from src.live.session_aggregator import aggregate_features

data = capture_session(5)

avg = aggregate_features(data)

print("\nAVERAGED FEATURES:")
print(avg)