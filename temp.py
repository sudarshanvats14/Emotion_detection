import cv2
import os
import glob
from mtcnn import MTCNN
from deepface import DeepFace

class EmotionDetection:

    def __init__(self):
        self.detector = MTCNN()

    def detect_emotions(self, image_path, output_image_path):
        image = cv2.imread(image_path)
        faces = self.detector.detect_faces(image)
        emotions = self.analyze_emotions(faces, image)
        self.draw_emotion_labels(image, faces, emotions)
        self.save_output_image(output_image_path, image)

    def analyze_emotions(self, faces, image):
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

    def draw_emotion_labels(self, image, faces, emotions):
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            emotion_label = emotions[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def save_output_image(self, output_image_path, image):
        cv2.imwrite(output_image_path, image)
        print(f"Output image saved as {output_image_path}")

if __name__ == "__main__":
    data_folder = 'data'
    output_folder = 'output'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    emotion_detector = EmotionDetection()

    # Use glob to get a list of image files in the 'data' folder
    image_files = glob.glob(os.path.join(data_folder, '*.jpg')) + glob.glob(os.path.join(data_folder, '*.jpeg')) + glob.glob(os.path.join(data_folder, '*.png'))

    for image_path in image_files:
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_folder, filename)

        emotion_detector.detect_emotions(image_path, output_image_path)
