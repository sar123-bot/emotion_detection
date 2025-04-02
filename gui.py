import cv2
import numpy as np
from keras.models import model_from_json

# Load Model
with open("emotiondetector.json", "r") as json_file:
    model = model_from_json(json_file.read())

model.load_weights("emotiondetector.h5")

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion Labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess image for the model
def extract_features(image):
    feature = np.expand_dims(image, axis=-1)  # Add channel dimension
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    return feature / 255.0  # Normalize pixel values

# Start Webcam Capture
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        print("Error: Unable to access webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        img = extract_features(roi_gray)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Draw rectangle and display emotion text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
