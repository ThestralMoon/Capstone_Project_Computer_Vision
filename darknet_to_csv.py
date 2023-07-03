# All annotations are in yolo darknet format and must
# be parsed and stored in a csv file
import os
import cv2
import pandas as pd
import glob

import config


def retrieve_class_from_id(class_id):
    f = open(config.CLASSES_FILE)
    classes_list = list(f.readlines())
    f.close()

    classes_list = [text.strip() for text in classes_list]

    classes_map = dict(enumerate(classes_list))

    return classes_map.get(class_id)


def parse_annotations(path):
    images = glob.glob(path + "\\*.png")
    annotations = glob.glob(path + "\\*.txt")

    index = 0
    rows = []

    for image in images:
        img = cv2.imread(image)
        dh, dw = img.shape[:2]

        f = open(annotations[index])
        data = f.readlines()
        f.close()

        index = index + 1

        for dt in data:

            class_id, x, y, w, h = map(float, dt.split(' '))

            class_name = str(retrieve_class_from_id(class_id))

            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1

            row = (os.path.basename(image), class_name, l, t, r, b, dw, dh)
            rows.append(row)

    col_labels = ['Image', 'Class', 'xMin', 'yMin', 'xMax', 'yMax', 'Width', 'Height']
    df = pd.DataFrame(rows, columns=col_labels)

    return df


folders = [config.TRAIN_PATH, config.TEST_PATH, config.VALIDATION_PATH]

print("Converting darknet to csv...")

# for folder in folders:
#     annotations_df = parse_annotations(folder)
#     out_path = os.path.join(os.getcwd(), 'annotated_frames\\' + os.path.basename(folder))
#     annotations_df.to_csv((out_path + '_annotations.csv'), index=None)
#     print('Created ' + out_path + '_annotations.csv')

annotations_df = parse_annotations(config.VALIDATION_PATH)
out_path = os.path.join(os.getcwd(), 'annotated_frames\\' + os.path.basename(config.VALIDATION_PATH))
annotations_df.to_csv((out_path + '_annotations.csv'), index=None)
print('Created ' + out_path + '_annotations.csv')
