import temp
import cv2
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from deepface import DeepFace

validation_data = [
    {'image_path': 'validation_data\\happy1.jpg', 'emotion': 'happy'}, 
    {'image_path': 'validation_data\\happy2.jpg', 'emotion': 'happy'},
    {'image_path': 'validation_data\\sad1.jpg', 'emotion': 'sad'},
    {'image_path': 'validation_data\\sad2.jpg', 'emotion': 'sad'},
    {'image_path': 'validation_data\\angry1.jpg', 'emotion': 'angry'},
    {'image_path': 'validation_data\\angry2.jpg', 'emotion': 'angry'},
    {'image_path': 'validation_data\\neutral1.jpg', 'emotion': 'neutral'},
    {'image_path': 'validation_data\\neutral2.jpg', 'emotion': 'neutral'},
    {'image_path': 'validation_data\\surprise1.jpg', 'emotion': 'surprise'},
    {'image_path': 'validation_data\\surprise2.jpg', 'emotion': 'surprise'},
    {'image_path': 'validation_data\\disgust1.jpg', 'emotion': 'disgust'},
    ]


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    return image

def detect_emotions(image_path, output_image_path, detector, emotion_detector):
    image = preprocess_image(image_path)
    faces = detector.detect_faces(image)
    emotions = emotion_detector.analyze_emotions(faces, image)
    return emotions

def analyze_emotions(faces, image, emotion_detector):
    emotions = []
    for face in faces:
        x, y, w, h = face['box']
        x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
        face_image = image[y:y + h, x:x + w]
        try:
            emotion_predictions = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
            if emotion_predictions:
                emotion = emotion_predictions[0]['dominant_emotion']
                emotions.append(emotion)
            else:
                emotions.append("No emotion detected")
        except ValueError:
            emotions.append("Face not detected")
    return emotions


detector = temp.MTCNN()
emotion_detector = temp.EmotionDetection()

true_emotions = []
predicted_emotions = []

n_splits = 2 # Adjust as needed

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_index, test_index in skf.split(validation_data, [item['emotion'] for item in validation_data]):
    train_set = [validation_data[i] for i in train_index]
    test_set = [validation_data[i] for i in test_index]

    for data in test_set:
        image_path = data['image_path']
        true_emotion = data['emotion']

        filename = os.path.basename(image_path)
        output_image_path = os.path.join('output', filename)

        detected_emotion = detect_emotions(image_path, output_image_path, detector, emotion_detector)

        true_emotions.append(true_emotion)
        predicted_emotions.append(detected_emotion[0])

accuracy = accuracy_score(true_emotions, predicted_emotions)

print(f"Accuracy: {accuracy * 100:.2f}%")
