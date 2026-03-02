import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

mp_face_mesh = mp.solutions.face_mesh


# ---------- helper ----------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_ratio(landmarks, eye_points):
    p = [landmarks[i] for i in eye_points]
    vertical = distance(p[1], p[5]) + distance(p[2], p[4])
    horizontal = distance(p[0], p[3])
    return vertical / (2.0 * horizontal)


def mouth_ratio(landmarks):
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]

    vertical = distance(top, bottom)
    horizontal = distance(left, right)

    return vertical / horizontal


# ---------- MAIN CAPTURE ----------
def capture_session(duration_sec=5):

    cap = cv2.VideoCapture(0)
    
    features_list = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:

        print("Capturing live session...")

        start = cv2.getTickCount()

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:

                landmarks = results.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape

                coords = [(int(l.x*w), int(l.y*h)) for l in landmarks]

                # eye ratios
                left_eye_idx = [33,160,158,133,153,144]
                right_eye_idx = [362,385,387,263,373,380]

                left_ear = eye_ratio(coords, left_eye_idx)
                right_ear = eye_ratio(coords, right_eye_idx)

                ear = (left_ear + right_ear) / 2
                mar = mouth_ratio(coords)

                # emotion detection
                try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                emo = result[0]["emotion"]

                except:
                    continue

                        features_list.append({
                            "eye_ratio": ear,
                            "mouth_ratio": mar,
                            "angry": emo["angry"],
                            "fear": emo["fear"],
                            "sad": emo["sad"],
                            "neutral": emo["neutral"],
                            "happy": emo["happy"],
                            "surprise": emo["surprise"]
                        })

            cv2.imshow("Live Capture (Press Q to stop)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
            if elapsed > duration_sec:
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Session completed.")
    return features_list