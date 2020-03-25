"""
Usage: To generate a train/test TFRecord file for TensorFlow hand detection model training

Attention, run the program twice
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

"""Structure of files and directories

    hand_my_dataset/
        |
        |--- hand_test/
        |   |
        |    --- hand_test_labels.csv
        |   |
        |    --- all the images for testing
        |
        | --- hand_train/
        |   |
        |    --- hand_train_labels.csv
        |   |
        |    --- all the images for training
        |
         --- hand_eval.record --> to be generated in this program
        |
         --- hand_train.record --> to be generated in this program
"""

# TODO: change the name of the directories for generating a train/test TFRecord file
dataset_dir = "hand_my_dataset"
image_dir = "hand_train"
csv_filename = "hand_train_labels.csv"
tfrecord_filename = "hand_train.record"


# TODO: replace this with label map when necessary
def class_text_to_int(row_label):
    if row_label == 'hand':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.io.TFRecordWriter(os.path.join(dataset_dir, tfrecord_filename))
    path = os.path.join(dataset_dir, image_dir)
    examples = pd.read_csv(os.path.join(path, csv_filename))
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created TFRecord file:', os.path.join(dataset_dir, tfrecord_filename))


if __name__ == '__main__':
    tf.compat.v1.app.run()