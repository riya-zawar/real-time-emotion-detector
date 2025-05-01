import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mini_xception_best.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_colors = {
    'Angry': (0, 0, 255),       # Red
    'Disgust': (0, 255, 0),     # Green
    'Fear': (128, 0, 128),      # Purple
    'Happy': (0, 255, 255),     # Yellow
    'Neutral': (200, 200, 200), # Gray
    'Sad': (255, 0, 0),         # Blue
    'Surprise': (255, 255, 0)   # Cyan
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        prediction = model.predict(face_input, verbose=0)[0]
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]
        confidence = int(prediction[emotion_index] * 100)

        label_text = f"{emotion} ({confidence}%)"
        box_color = emotion_colors.get(emotion, (255, 255, 255))  # default white if missing

        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), box_color, -1)
        cv2.putText(frame, label_text, (x + 5, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('Real-Time Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
