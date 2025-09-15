import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Load trained model
model = tf.keras.models.load_model("models/vr_cnn_lstm.h5")

# Emotion labels (update based on your dataset order)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Sequence buffer for LSTM (last 10 frames)
sequence_length = 10
frame_sequence = deque(maxlen=sequence_length)

# Open webcam
cap = cv2.VideoCapture(0)
print("🎥 Starting real-time facial expression detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)) / 255.0
        roi_gray = roi_gray.reshape(48, 48, 1)
        frame_sequence.append(roi_gray)

        # Run prediction when sequence is full
        if len(frame_sequence) == sequence_length:
            input_data = np.array(frame_sequence).reshape(1, sequence_length, 48, 48, 1)
            prediction = model.predict(input_data, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("VR Engagement - Real-time Facial Expression", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
