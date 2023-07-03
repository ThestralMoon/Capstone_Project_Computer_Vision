import cv2
import os

video_files_path = 'C:\\Users\\shahe\\PycharmProjects\\Capstone_Project_Computer_Vision\\VIRAT\\Public Dataset\\VIRAT Video Dataset Release 2.0\\VIRAT Ground Dataset\\videos_original'
video_frames_path = 'C:\\Users\\shahe\\PycharmProjects\\Capstone_Project_Computer_Vision\\video_frames'

def extract_frames(video, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video)
    ind = 0
    while cap.isOpened():
        ret, mat = cap.read()
        if ret:
            if ind % 5 == 0:
                frame_name = output_path + str(ind) + '.png'
                print("Creating " + frame_name)
                cv2.imwrite(frame_name, mat)

            ind = ind + 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return

video_files = os.listdir(video_files_path)
for vid in video_files:
    vid_basename = os.path.splitext(vid)[0]
    in_file = video_files_path + '/' + vid
    out_path = video_frames_path + '/' + vid_basename + "_frame"
    extract_frames(in_file, out_path)
