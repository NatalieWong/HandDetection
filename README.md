# Auto CSV Generator for TensorFlow Object Detection model training

This program is specialized for generating a csv file which can be used for training a TensorFlow hand detection model.

Run `python HandDetection.py` to execute the program.

This program detects hand in frames from a video source using OpenCV. If a hand is detected in a frame, the frame can be saved in JPEG format under the `image` directory and the coordinates of the bounding box for the hand in the frame will be recorded. After the detection of hand in the video frames finished, a csv file `hand_label.csv` will be generated.

When the operations for frame extraction and writing `hand_label.csv` file is completed, run `split_train_test_csv_images.py` to split the dataset and the csv file into 2 separated directories. `hand_train` is used for training and `hand_test` is used for testing during the hand detection model training process.

## Important Notice
This program has been successfully run using python2.7.

Remember to `pip install cv2 numpy csv pandas scikit-learn` to get all the dependencies ready before the execution of the program.

Currently only one label - `hand` is supported. See the line `rowdata = [filename, self.frameWidth, self.frameHeight, 'hand', xmin, ymin, xmax, ymax]`

At present, as there are a number of hard-coded lines, please change the following lines in `HandDetection.py` to customize the program.

- `?.mp4` in the line `hand_detector = HandDetector('resources/?.mp4')` to your video file in MP4 format
- `self.cntImg` to an integer that you want to start from for the generation of the saved image filename.
- `self.csvholder = [['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]` is used for generating the csv header at the first time you run the program. After you have run the program once, use `self.csvholder = []` instead.
- `with open('hand_label.csv', 'w') as file:` will create a csv file for the first time you run the program. If you want to append data to this file onwards, use `with open('hand_label.csv','a') as file:` instead.

An `image` directory will be automatically created for you to hold all the JPEG images which are going to be saved during the execution of the program. You may create another directory with a different name for the same purpose by changing the variable `self.image_dir`.

After a hand is detected in the frame, you can save the frame by activating any one of the debug window and pressing the `s` hotkey twice. You can also press the `x` hotkey once to discard the frame. Otherwise, after 8 seconds, the frame will be discarded automatically. You may change the time period for making a decision by modifying the number in ms i.e. 8000 in the lines `if cv2.waitKey(8000) & 0xFF == ord('x'):` and `elif cv2.waitKey(8000) & 0xFF == ord('s'):`.

In `split_train_test_csv_images.py`, you may change the names of the training and testing directories by modifying the variables `train_image_dir` and `test_image_dir`.

## Optional Operation
You may call `visualize_image_bbox.py` seperately to visualize the bounding box for each image saved under the `image` directory.