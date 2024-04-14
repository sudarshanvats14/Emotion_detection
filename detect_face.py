import cv2
from mtcnn import MTCNN
from pytube import YouTube
import os

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, frame, frame_number, output_folder):
        faces = self.detector.detect_faces(frame)

        for i, face in enumerate(faces):
            x, y, width, height = face['box']
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Save the frame with bounding boxes
            frame_with_boxes = frame.copy()
            output_path = os.path.join(output_folder, f"frame_{frame_number:04d}_face_{i:02d}.jpg")
            cv2.imwrite(output_path, frame_with_boxes)

def process_video_from_youtube(youtube_url, output_folder):
    detector = FaceDetector()

    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension='mp4').first()
    video_path = stream.download(output_path='.')

    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detector.detect_faces(frame, frame_number, output_folder)
        frame_number += 1

    cap.release()

if __name__ == '__main__':
    youtube_url = 'https://www.youtube.com/shorts/09S-aDj1RTw'
    output_folder = 'output'  # Output folder for processed frames

    os.makedirs(output_folder, exist_ok=True)
    process_video_from_youtube(youtube_url, output_folder)
