import pandas as pd
import numpy as np
import csv
import os
import shutil

"""Structure of files and directories

    hand_images/ --> holding all the images to be split into training/testing images

    hand_labels.csv --> holding all the information of the images and coordinates of the hand bounding boxes in the images

    hand_my_dataset/ --> to be generated in this program
        |
        |--- hand_test/
        |   |
        |    --- hand_test_labels.csv
        |   |
        |    --- all the images for testing
        |
         --- hand_train/
            |
             --- hand_train_labels.csv
            |
             --- all the images for training
"""

# TODO: change the name of the directories when necessary
curr_image_dir = "hand_images"

dataset_dir = "hand_my_dataset/"
train_image_dir = "hand_train"
test_image_dir = "hand_test"
train_csv_filename = "hand_train_labels.csv"
test_csv_filename = "hand_test_labels.csv"


# split train & test csv
# TODO: change the name of the csv file when necessary
imgdf = pd.read_csv("hand_labels.csv")
x = imgdf.iloc[:,:].values
y = np.ones(imgdf.shape[0]) # dummy values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# create directories, write rows to train & test csv
if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)


if not os.path.exists(os.path.join(dataset_dir, train_image_dir)):
    os.mkdir(os.path.join(dataset_dir, train_image_dir))

with open(os.path.join(dataset_dir + train_image_dir, train_csv_filename), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    for i in range(len(x_train)):
        writer.writerow(x_train[i])

if not os.path.exists(os.path.join(dataset_dir, test_image_dir)):
    os.mkdir(os.path.join(dataset_dir, test_image_dir))

with open(os.path.join(dataset_dir + test_image_dir, test_csv_filename), 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
    for i in range(len(x_test)):
        writer.writerow(x_test[i])



# move images to train/test directory
img_train_df = pd.read_csv(os.path.join(dataset_dir + train_image_dir, train_csv_filename))
filename_train_list = img_train_df['filename'].tolist()
for filename in filename_train_list:
    shutil.copyfile(os.path.join(curr_image_dir, filename), os.path.join(dataset_dir + train_image_dir, filename))


img_test_df = pd.read_csv(os.path.join(dataset_dir + test_image_dir, test_csv_filename))
filename_test_list = img_test_df['filename'].tolist()
for filename in filename_test_list:
    shutil.copyfile(os.path.join(curr_image_dir, filename), os.path.join(dataset_dir + test_image_dir, filename))