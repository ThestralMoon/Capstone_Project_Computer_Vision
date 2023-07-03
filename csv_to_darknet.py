import os
import cv2
import pandas as pd
import glob

import config

type_path = {'train': config.TRAIN_PATH, 'test': config.TEST_PATH, 'val': config.VALIDATION_PATH}


def class_to_id(class_label):
    f = open(config.CLASSES_FILE)
    classes_list = list(f.readlines())
    f.close()

    classes_list = [text.strip() for text in classes_list]

    return classes_list.index(class_label)


# data_type (train/test/val)
def csv_to_yolo(path, data_type):
    df = pd.read_csv(path)
    grouped = df[df.get('Image').str.contains('augmented')].groupby('Image')
    for name, group in grouped:
        base_name = name.strip('.png')
        out_path = os.path.join(type_path.get(data_type), base_name + '.txt')

        mod_group = group.reset_index().drop(columns=['index', 'Image'])
        mod_group = mod_group.assign(Class=mod_group.get('Class').apply(class_to_id)).rename(columns={'Class': 'class_id'})

        dw = 1./mod_group.get('Width')
        dh = 1./mod_group.get('Height')

        class_ids = mod_group.get('class_id')
        x_centers = (((mod_group.get('xMin') + mod_group.get('xMax')) / 2) / mod_group.get('Width')).round(6)
        y_centers = (((mod_group.get('yMin') + mod_group.get('yMax')) / 2) / mod_group.get('Height')).round(6)
        widths = ((mod_group.get('xMax') - mod_group.get('xMin')) * dw).round(6)
        heights = ((mod_group.get('yMax') - mod_group.get('yMin')) * dh).round(6)

        data = {
                'class': class_ids,
                'x_center': x_centers,
                'y_center': y_centers,
                'width': widths,
                'height': heights
        }

        yolo_df = pd.DataFrame(data=data)
        yolo_df.to_csv(out_path, sep=' ', header=None, index=None)


train_data_csv = os.path.join(config.FRAMES_PATH, "combined_validation_data.csv")
csv_to_yolo(train_data_csv, 'val')
