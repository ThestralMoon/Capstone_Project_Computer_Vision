import os
import glob
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

file_path = os.path.join(os.getcwd(), "VIRAT_Ground_Dataset/annotations/VIRAT_S_000001.viratdata.objects.txt")
video_path = os.path.join(os.getcwd(), "VIRAT_Ground_Dataset/videos_original/VIRAT_S_000001.mp4")

df = pd.read_csv(file_path, sep=' ',
                 names=['object_id', 'object_duration', 'current_frame', 'bbox_lefttop_x', 'bbox_lefttop_y',
                        'bbox_width', 'bbox_height', 'object_type'])

ind = 0
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_FRAMES, 3455)

success, frame = cap.read()

if success:
    cv2.imwrite("test_frame.png", frame)

cap.release()
cv2.destroyAllWindows()

# img = cv2.imread("test_frame.png")
#
# x, y, w, h = 1, 663, 76, 132
#
# cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
# cv2.imshow("Test Frame With Bbox", img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(df.head())
print(df.shape)

x_centers = (df['bbox_lefttop_x'] + df['bbox_width']) / 2
y_centers = (df['bbox_lefttop_y'] + df['bbox_height']) / 2

x_norm = x_centers / 1920
y_norm = y_centers / 1080
w_norm = df['bbox_width'] / 1920
h_norm = df['bbox_height'] / 1080

# print(x_norm)
# print(y_norm)
# print(w_norm)
# print(h_norm)

print(df['current_frame'].sort_values(ascending=True))

grouped = df.groupby('current_frame')
i = 0

for name, group in grouped:
    print(name)
    # print(group)
    if i == 3:
        break

    i = i + 1

data_path = os.path.join(os.getcwd(), "VIRAT Ground Dataset/annotations")
videos_path = os.path.join(os.getcwd(), "VIRAT Ground Dataset/videos_original")


def extract_frames():
    obj_files = glob.glob(data_path + "\\*.viratdata.objects.txt")
    for i in range(len(obj_files)):
        df = pd.read_csv(obj_files[i], header=None, sep=' ',
                         names=['object_id', 'object_duration', 'current_frame', 'bbox_lefttop_x', 'bbox_lefttop_y',
                                'bbox_width', 'bbox_height', 'object_type'])
        print(df.head(3))
        print(df.shape)
        print(df['bbox_height'])
        if i == 0:
            break

# extract_frames()
