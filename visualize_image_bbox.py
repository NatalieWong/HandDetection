import pandas as pd
import os
import cv2

imgdf = pd.read_csv("hand_train.csv")

for filename in os.listdir("image"):
    print filename

    i = imgdf.index[imgdf['filename'] == filename]
    xmin = imgdf.loc[i, "xmin"]
    ymin = imgdf.loc[i, "ymin"]
    xmax = imgdf.loc[i, "xmax"]
    ymax = imgdf.loc[i, "ymax"]
    print xmin, ymin, xmax, ymax

    # visualize hand bounding box
    img = cv2.imread('image/'+filename)
    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    cv2.imshow('Verifying annotation of '+filename, img)
    cv2.waitKey(3000)
    cv2.destroyWindow('Verifying annotation of '+filename)