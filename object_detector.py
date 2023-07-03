import csv
import glob
import os

import keras.backend

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import gc

import cv2
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

from tqdm import tqdm

import config

# phys_devices = tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(phys_devices))
# tf.config.experimental.set_memory_growth(phys_devices[0], True)


##########################################################################

# raw_dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(), 'train.tfrecord'))


def read_tfrecord(tfexample):
    features = {
        'image/name': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }

    parsed_features = tf.io.parse_single_example(tfexample, features)
    return parsed_features


# dataset = raw_dataset.map(read_tfrecord)
# label = tf.stack(dataset['image/object/class/label'],
#     dataset['image/object/bbox/xmin'],
#     dataset['image/object/bbox/ymin'],
#     dataset['image/object/bbox/xmax'],
#     dataset['image/object/bbox/ymax'],
#                  axis=0
# )
#
# img = tf.io.decode_png(dataset['image/encoded'])
# img_batch, labels_batch = tf.train.shuffle_batch


##########################################################################


def prepare_and_load_data(csv_path, images_path):
    rows = pd.read_csv(csv_path)
    rows = rows.dropna()

    images_list = []
    class_list = []
    bboxes_list = []

    for ind, row in rows.iterrows():
        (img_name, class_label, xMin, yMin, xMax, yMax, width, height) = row

        x1 = float(xMin) / float(width)
        y1 = float(yMin) / float(height)
        x2 = float(xMax) / float(width)
        y2 = float(yMax) / float(height)

        img_path = os.path.join(images_path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(640, 360))
        img_arr = tf.keras.preprocessing.image.img_to_array(img)

        images_list.append(img_arr)
        class_list.append(class_labels.get(class_label))
        bboxes_list.append((x1, y1, x2, y2))

    images_list = np.array(images_list, dtype='float32') / 255.0
    class_list = np.array(class_list)
    bboxes_list = np.array(bboxes_list, dtype='float32')

    return images_list, class_list, bboxes_list


class_labels = {'car': 0, 'person': 1, 'box': 2}

TRAIN_CSV = os.path.join(config.FRAMES_PATH, 'combined_train_data.csv')
TEST_CSV = os.path.join(config.FRAMES_PATH, 'combined_test_data.csv')

train_images, train_classes, train_bboxes = prepare_and_load_data(TRAIN_CSV, config.TRAIN_PATH)
test_images, test_classes, test_bboxes = prepare_and_load_data(TEST_CSV, config.TEST_PATH)

def reset_keras():
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()
    sess = keras.backend.get_session()

    print(gc.collect())  # if it does something you should see a number as output

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    keras.backend.set_session(tf.compat.v1.Session(config=config))


def create_cnn():
    inputs = tf.keras.Input(shape=(640, 360, 3))

    conv2D = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(inputs)
    max_pool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2D)
    conv2D = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(max_pool2D)
    max_pool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2D)
    conv2D = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(max_pool2D)
    max_pool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2D)
    conv2D = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(max_pool2D)
    max_pool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2D)
    conv2D = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.1))(max_pool2D)
    max_pool2D = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2D)

    flatten = tf.keras.layers.Flatten()(max_pool2D)

    # Bounding Box Branch
    box = tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(flatten)
    box = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(box)
    box = tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(box)
    box = tf.keras.layers.Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(box)
    box = tf.keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(box)
    box = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(box)

    # Classification Branch
    classifier = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(flatten)
    classifier = tf.keras.layers.Dropout(0.15)(classifier)
    classifier = tf.keras.layers.Dense(3, activation='softmax', name='classes')(classifier)

    model = tf.keras.Model(inputs=inputs, outputs=(box, classifier))
    return model


model = create_cnn()

losses = {'bbox': tf.losses.MSE, 'classes': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}

train_targets = {'bbox': train_bboxes, 'classes': train_classes}
test_targets = {'bbox': test_bboxes, 'classes': test_classes}

optimizer = tf.keras.optimizers.Adam(learning_rate=config.INIT_LEARN_RATE, epsilon=0.1, clipnorm=0.5)
model.compile(loss=losses, optimizer=optimizer, metrics=['mean_absolute_error', 'accuracy'])

# with open('metrics/summaries/model_summary_ver1.txt', 'w') as f:
#     model.summary(print_fn=lambda x: f.write(x + '\n'))
#
# tf.keras.utils.plot_model(model, to_file='metrics/graphs/model_ver_one.png')

tboard_path = os.path.join(os.getcwd(), 'metrics')

tboard = tf.keras.callbacks.TensorBoard(
    log_dir=tboard_path,
    histogram_freq=0,
    write_graph=True,
    write_images=True
)

hist_path = os.path.join(os.getcwd(), 'metrics/', '')


with tf.device('CPU'):
    history = model.fit(x=train_images, y=train_targets, validation_data=(test_images, test_targets),
                    batch_size=config.BATCH_SIZE, callbacks=[tboard], epochs=config.NUM_EPOCHS,
                    verbose=1, shuffle=True)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(hist_path + "model_hist_ver1", index=None)

