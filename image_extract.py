import os
import cv2

def frame_extract(video_path, output_dir='./frames'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)                    # 1 second interval

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'")