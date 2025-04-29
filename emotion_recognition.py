import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)  # Increased confidence

# Define emotion-to-color mapping
emotion_colors = {
    'angry': (0, 0, 255),        # Red
    'disgust': (128, 0, 128),    # Purple
    'fear': (0, 255, 255),       # Yellow
    'happy': (0, 255, 0),        # Green
    'neutral': (255, 255, 255),  # White
    'sad': (255, 0, 0),          # Blue
    'surprise': (255, 165, 0)    # Orange
}

# Function to process frames
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB (required by MediaPipe & DeepFace)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        # Default emotion
        dominant_emotion = "neutral"
        color = emotion_colors['neutral']

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )

                # Ensure bounding box is within frame limits
                x, y, w, h = max(x, 0), max(y, 0), min(w, iw - x), min(h, ih - y)

                # Crop and preprocess face for DeepFace
                face = frame[y:y+h, x:x+w]

                if face.size > 0:
                    # Convert face to grayscale for better feature extraction
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(gray_face, (48, 48))  # Standardizing size
                    face_resized = np.stack((face_resized,) * 3, axis=-1)  # Convert to 3-channel

                    try:
                        # Analyze face emotion with DeepFace
                        result = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False)
                        dominant_emotion = result[0]['dominant_emotion']

                        # Set bounding box color based on emotion
                        color = emotion_colors.get(dominant_emotion, (255, 255, 255))
                    except Exception:
                        dominant_emotion = "neutral"

                # Draw rectangle around detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode the frame and yield as response
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('emotion_recognition.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
