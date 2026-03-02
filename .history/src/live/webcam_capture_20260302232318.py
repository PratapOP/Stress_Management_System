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

    print("AI Face Analysis Started...")

    start = cv2.getTickCount()

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
        progress = min(elapsed / duration_sec, 1.0)

        status_text = "Searching Face..."

        for (x, y, w, h) in faces:

            status_text = "Face Detected"

            # ---------- FACE BOX ----------
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            eye_ratio = 0.25

            # ---------- EYE DETECTION ----------
            for (ex, ey, ew, eh) in eyes:

                cv2.rectangle(
                    frame,
                    (x+ex, y+ey),
                    (x+ex+ew, y+ey+eh),
                    (255, 0, 0),
                    2
                )

            if len(eyes) >= 2:
                status_text = "Eyes Detected"

                eye_centers = []
                for (ex, ey, ew, eh) in eyes[:2]:
                    cx = x + ex + ew//2
                    cy = y + ey + eh//2
                    eye_centers.append((cx, cy))

                # draw line between eyes
                cv2.line(frame, eye_centers[0], eye_centers[1], (0, 255, 255), 2)

                eye_distance = np.linalg.norm(
                    np.array(eye_centers[0]) -
                    np.array(eye_centers[1])
                )

                eye_ratio = float(eye_distance / w)

                cv2.putText(
                    frame,
                    f"Eye Distance: {int(eye_distance)} px",
                    (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,255),
                    2
                )

            # ---------- FACE STRUCTURE ----------
            face_ratio = h / w

            cv2.putText(
                frame,
                f"Face Ratio: {face_ratio:.2f}",
                (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            # ---------- SIMPLE EMOTION PROXY ----------
            neutral = 0.5
            sad = max(0.05, 0.4 - eye_ratio)
            happy = max(0.05, eye_ratio)

            angry = 0.1
            fear = 0.1
            surprise = 0.1

            total = angry + fear + sad + neutral + happy + surprise

            features_list.append({
                "eye_ratio": eye_ratio,
                "mouth_ratio": face_ratio * 0.3,
                "angry": angry/total,
                "fear": fear/total,
                "sad": sad/total,
                "neutral": neutral/total,
                "happy": happy/total,
                "surprise": surprise/total
            })

            break  # only first face

        # ---------- UI OVERLAY ----------
        cv2.putText(
            frame,
            "AI FACE ANALYSIS ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Capture Progress: {int(progress*100)}%",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ---------- FINISH MESSAGE ----------
        if progress >= 1.0:
            cv2.putText(
                frame,
                "DATA COLLECTED - PRESS Q TO CONTINUE",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("AI Live Face Scanner", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Session completed.")
    return features_list