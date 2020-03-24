import pandas as pd
import os
import cv2
import shutil
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

imgdf = pd.read_csv("hand_my_dataset/hand_test/hand_test_labels.csv")

for filename in os.listdir("hand_my_dataset/hand_test"):

    i = imgdf.index[imgdf['filename'] == filename]

    xmin = imgdf.loc[i, "xmin"].values[0]
    ymin = imgdf.loc[i, "ymin"].values[0]
    xmax = imgdf.loc[i, "xmax"].values[0]
    ymax = imgdf.loc[i, "ymax"].values[0]

    print filename, (xmin, ymin, xmax, ymax)

    # visualize hand bounding box
    # img = mpimg.imread('hand_images/'+filename)
    img = cv2.imread('hand_my_dataset/hand_test/'+filename)
    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    cv2.imshow('Verifying annotation of '+filename, img)
    cv2.waitKey(2000)
    cv2.destroyWindow('Verifying annotation of '+filename)
