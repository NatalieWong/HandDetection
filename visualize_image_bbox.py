import pandas as pd
import os
import cv2
import shutil
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

"""WARNING!
    If you run this script after `split_train_test_csv_images.py`,
    please move the csv file in hand_train/ or hand_test/ directory
    one-level up to the first level of the dataset directory.

    For mac users, please check if `.DS_Store` file exists
    in hand_train/ or hand_test/ directories.
"""

# TODO: modify path
imgdf = pd.read_csv("hand_my_dataset/hand_train_labels.csv")

# TODO: modify path
for filename in os.listdir("hand_my_dataset/hand_train"):

    i = imgdf.index[imgdf['filename'] == filename]

    xmin = imgdf.loc[i, "xmin"].values[0]
    ymin = imgdf.loc[i, "ymin"].values[0]
    xmax = imgdf.loc[i, "xmax"].values[0]
    ymax = imgdf.loc[i, "ymax"].values[0]

    print filename, (xmin, ymin, xmax, ymax)

    # visualize hand bounding box
    # img = mpimg.imread('hand_images/'+filename)

    # TODO: modify path
    img = cv2.imread('hand_my_dataset/hand_train/'+filename)
    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    cv2.imshow('Verifying annotation of '+filename, img)
    cv2.waitKey(3000)
    cv2.destroyWindow('Verifying annotation of '+filename)
