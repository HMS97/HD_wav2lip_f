import torch, face_detection
import scipy, cv2, os, sys, argparse, audio
import numpy as np
from tqdm import tqdm

face = '../videos/main/fps_corrected_video/video-3-0-4a.mp4'
video_stream = cv2.VideoCapture(face)
fps = video_stream.get(cv2.CAP_PROP_FPS)
crop = [0, -1, 0, -1]
print('Reading video frames...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                        flip_input=False, device='cpu')
full_frames = []
while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break


    y1, y2, x1, x2 = crop
    if x2 == -1: x2 = frame.shape[1]
    if y2 == -1: y2 = frame.shape[0]

    frame = frame[y1:y2, x1:x2]

    full_frames.append(frame)

predictions = []
batch_size = 1
for i in tqdm(range(0, len(full_frames), batch_size)):
    # print(images[i:i + batch_size][0].shape, len(images[i:i + batch_size]))
    predictions.extend(detector.get_detections_for_batch(np.array(full_frames[i:i + batch_size])))
print(len(full_frames))