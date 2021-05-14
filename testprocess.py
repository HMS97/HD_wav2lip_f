import cv2
import numpy as np
vfile = '/home/huimingsun/Desktop/wav2lip/videos/main/videos/video-0-0-1a.mp4'
video_stream = cv2.VideoCapture(vfile)
# video_stream = skvideo.io.VideoCapture(vfile)
frames = []
while 1:
    # still_reading, frame = video_stream.read()
    still_reading, frame = video_stream.read()
    # print(frame.shape)

    # print(still_reading)
    if not still_reading:
        video_stream.release()
        break
    frame = cv2.resize(frame,(256,256))
    frames.append(frame)
print(frames[0].shape)
print(len(frames))
print(np.asarray(frames).shape)