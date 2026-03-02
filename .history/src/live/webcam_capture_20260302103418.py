import cv2
import numpy as np


def capture_session(duration_sec=5):

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    features_list = []

    print("Capturing live session...")

    start = cv2.getTickCount()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            # ---------- SIMPLE PROXY FEATURES ----------
            eye_ratio = 0.25
            if len(eyes) > 0:
                eye_sizes = [eh/ew for (_,_,ew,eh) in eyes]
                eye_ratio = float(np.mean(eye_sizes))

            mouth_ratio = float(h / w) * 0.3

            # ---------- SIMULATED EMOTIONS ----------
            # (stable demo approximation)
            neutral = 0.4
            happy = max(0.05, 0.4 - eye_ratio)
            sad = max(0.05, 0.3 + (0.3 - eye_ratio))

            angry = 0.1
            fear = 0.1
            surprise = 0.1

            total = angry + fear + sad + neutral + happy + surprise

            features_list.append({
                "eye_ratio": eye_ratio,
                "mouth_ratio": mouth_ratio,
                "angry": angry/total,
                "fear": fear/total,
                "sad": sad/total,
                "neutral": neutral/total,
                "happy": happy/total,
                "surprise": surprise/total
            })

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

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