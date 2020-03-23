import pandas as pd
import numpy as np
import csv
import os
import shutil

curr_image_dir = "image/"
train_image_dir = "hand_train/"
test_image_dir = "hand_test/"

imgdf = pd.read_csv("hand_label.csv")

# split train & test csv
x = imgdf.iloc[:,:].values
y = np.ones(imgdf.shape[0])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# write rows to train & test csv
if not os.path.exists(train_image_dir):
    os.mkdir(train_image_dir)

with open('hand_train/hand_train.csv', 'w') as file:
    writer = csv.writer(file)
    for i in range(len(x_train)):
        writer.writerow(x_train[i])

if not os.path.exists(test_image_dir):
    os.mkdir(test_image_dir)

with open('hand_test/hand_test.csv', 'w') as file:
    writer = csv.writer(file)
    for i in range(len(x_test)):
        writer.writerow(x_test[i])



# move images to train/test folder
img_train_df = pd.read_csv("hand_train/hand_train.csv")
filename_train_list = img_train_df['filename'].tolist()
for filename in filename_train_list:
    shutil.copyfile(curr_image_dir+filename, train_image_dir+filename)


img_test_df = pd.read_csv("hand_test/hand_test.csv")
filename_test_list = img_test_df['filename'].tolist()
for filename in filename_test_list:
    shutil.copyfile(curr_image_dir+filename, test_image_dir+filename)