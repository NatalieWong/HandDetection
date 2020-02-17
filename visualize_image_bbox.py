import pandas as pd
import os
import cv2

imgdf = pd.read_csv("hand_train.csv")

for filename in os.listdir("image"):
    # print filename
    rownum = int(filename[filename.rfind('_')+1 : filename.rfind('.')]) - 1

    # xmin = imgdf.loc[filename, "xmin"]
    # ymin = imgdf.loc[filename, "ymin"]
    # xmax = imgdf.loc[filename, "xmax"]
    # ymax = imgdf.loc[filename, "ymax"]
    xmin = imgdf.loc[rownum, "xmin"]
    ymin = imgdf.loc[rownum, "ymin"]
    xmax = imgdf.loc[rownum, "xmax"]
    ymax = imgdf.loc[rownum, "ymax"]
    print rownum+1, xmin, ymin, xmax, ymax

    # visualize hand bounding box
    img = cv2.imread('image/'+filename)
    cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
    cv2.imshow('Verifying annotation of '+filename, img)
    cv2.waitKey(4000)
    cv2.destroyWindow('Verifying annotation of '+filename)