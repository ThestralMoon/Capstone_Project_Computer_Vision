import os.path
import re
import shutil

import imageio
import imgaug as ia
import pandas as pd
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import config


def bbs_obj_to_df(bbs_object):
    #     convert BoundingBoxesOnImage object into array
    bbs_array = bbs_object.to_xyxy_array()
    #     convert array into a DataFrame ['xmin', 'ymin', 'xmax', 'ymax'] columns
    df_bbs = pd.DataFrame(bbs_array, columns=['xMin', 'yMin', 'xMax', 'yMax'])
    return df_bbs


aug = iaa.SomeOf(2, [
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Affine(rotate=(-60, 60)),
    iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
    iaa.Fliplr(1),
    iaa.Multiply((0.5, 1.5)),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.AdditiveGaussianNoise(scale=(0.03 * 255, 0.05 * 255))
])


def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    # create data frame which we're going to populate with augmented image info
    aug_bbs_xy = pd.DataFrame(columns=
                              ['Image', 'Width', 'Height', 'Class', 'xMin', 'yMin', 'xMax', 'yMax']
                              )
    grouped = df.groupby('Image')

    for filename in df['Image'].unique():
        #   get separate data frame grouped by file name
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        #   read the image
        image = imageio.v2.imread(images_path + '/' + filename)
        #   get bounding boxes coordinates and write into array
        bb_array = group_df.drop(['Image', 'Width', 'Height', 'Class'], axis=1).values
        #   pass the array of bounding boxes coordinates to the imgaug library
        bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
        #   apply augmentation on image and on the bounding boxes
        image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()

        #   don't perform any actions with the image if there are no bounding boxes left in it
        if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
            pass

        #   otherwise continue
        else:
            #   write augmented image to a file
            imageio.imwrite(aug_images_path + image_prefix + filename, image_aug)
            #   create a data frame with augmented values of image width and height
            info_df = group_df.drop(['xMin', 'yMin', 'xMax', 'yMax'], axis=1)
            for index, _ in info_df.iterrows():
                info_df.at[index, 'Width'] = image_aug.shape[1]
                info_df.at[index, 'Height'] = image_aug.shape[0]
            #   rename filenames by adding the predifined prefix
            info_df['Image'] = info_df['Image'].apply(lambda x: image_prefix + x)
            #   create a data frame with augmented bounding boxes coordinates using the function we created earlier
            bbs_df = bbs_obj_to_df(bbs_aug)
            #   concat all new augmented info into new data frame
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            #   append rows to aug_bbs_xy data frame
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

            # return dataframe with updated images and bounding boxes annotations
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


val_df_path = os.path.join(config.FRAMES_PATH, 'validation_frames_annotations.csv')
aug_path = os.path.join(config.FRAMES_PATH, 'aug_validation', "")
val_df = pd.read_csv(val_df_path)
aug_validation_df = image_aug(val_df, config.VALIDATION_PATH, aug_path,
                         'augmented_', aug)

all_labels_df = pd.concat([val_df, aug_validation_df])
all_labels_df.to_csv('combined_validation_data.csv', index=False)

for file in os.listdir(aug_path):
    shutil.copy(aug_path + file, config.VALIDATION_PATH + '/' + file)
