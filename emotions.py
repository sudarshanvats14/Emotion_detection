import cv2
import os
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from pytube import YouTube

def download_video_frames(video_url, output_dir):
    yt = YouTube(video_url)
    stream = yt.streams.filter(adaptive=True, file_extension="mp4").first()
    video_stream = stream.download(output_path=output_dir)
    return video_stream

detector = MTCNN()

def process_video(video_path, output_dir):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        faces = detector.detect_faces(frame)

        # Analyze emotions for each detected face
        emotions = []
        for face in faces:
            x, y, w, h = face['box']
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            face_image = frame[y:y + h, x:x + w]

            # Perform emotion analysis
            try:
                emotion_predictions = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                emotion = emotion_predictions[0]['dominant_emotion']
                emotions.append(emotion)
            except ValueError:
                emotions.append("Face not detected")

        # Draw bounding boxes with emotions
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            emotion_label = emotions[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame with detected faces and emotions
        output_frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(output_frame_path, frame)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# YouTube video URL
video_url = "https://www.youtube.com/watch?v=fU39-XSVE8U"

# Output directory for frames
output_directory = "output"

# Download the video
video_path = download_video_frames(video_url, output_directory)

# Process the downloaded video frames
process_video(video_path, output_directory)