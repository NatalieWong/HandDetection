"""for resizing the images to the desired size after the execution of HandDetection.py"""

import pandas as pd
import os
import cv2
import shutil
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

imgdf = pd.read_csv("hand_labels-size_err.csv")

for filename in os.listdir("hand_images-size_err"):

    i = imgdf.index[imgdf['filename'] == filename]

    w = imgdf.loc[i, 'width'].values[0]
    h = imgdf.loc[i, 'height'].values[0]

    if (not w == 1280) and (not h == 720):
        # print filename

        # resize the image
        ratio_w = 1280 / w
        ratio_h = 720 / h

        img = cv2.imread('hand_images-size_err/'+filename)
        resized_frame = cv2.resize(img, (0,0), fx=ratio_w, fy=ratio_h)

        # print "resized size:", (resized_frame.shape[1], resized_frame.shape[0])

        # change csv cell values
        imgdf.loc[i, 'width'] = resized_frame.shape[1]
        imgdf.loc[i, 'height'] = resized_frame.shape[0]

        xmin = imgdf.loc[i, "xmin"].values[0]
        ymin = imgdf.loc[i, "ymin"].values[0]
        xmax = imgdf.loc[i, "xmax"].values[0]
        ymax = imgdf.loc[i, "ymax"].values[0]

        # print "original: ", (xmin, ymin, xmax, ymax)

        imgdf.loc[i, "xmin"] = xmin * ratio_w
        imgdf.loc[i, "ymin"] = ymin * ratio_h
        imgdf.loc[i, "xmax"] = xmax * ratio_w
        imgdf.loc[i, "ymax"] = ymax * ratio_h

        xmin = imgdf.loc[i, "xmin"].values[0]
        ymin = imgdf.loc[i, "ymin"].values[0]
        xmax = imgdf.loc[i, "xmax"].values[0]
        ymax = imgdf.loc[i, "ymax"].values[0]

        # print "resized: ", (xmin, ymin, xmax, ymax)

        # save the resized frame
        cv2.imwrite(os.path.join('hand_images', filename), resized_frame)
        # visualize bounding box
        # img = cv2.imread(os.path.join('hand_images', filename))
        # cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
        # cv2.imshow('Verifying annotation of '+filename, img)
        # cv2.waitKey(2000)
        # cv2.destroyWindow('Verifying annotation of '+img)
    else:
        # copy the image file
        shutil.copyfile(os.path.join('hand_images-size_err', filename), os.path.join('hand_images', filename))
        

imgdf.to_csv('hand_labels.csv', index = False, header = False)