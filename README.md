# Auto CSV Generator for TensorFlow Object Detection model training

This program is specialized for generating a csv file which can be used for training a TensorFlow hand detection model.

This program detects hand using OpenCV in frames which are extracted from a video source. If a hand is detected in a frame, the frame will be saved in JPEG format under the image folder and the coordinates of the bounding box for the hand in the frame will be recorded. After the detection of hand in the video frames finished, a csv file will be generated.

## Important Notice
This program has been successfully run using python2.7.

Remember to `pip install cv2 numpy csv pandas` to get all the dependencies ready before the execution of the program.

Please create a folder called `image` by yourself to hold all the JPEG images which are going to be saved during the execution of the program.

Currently only one label - hand is supported. See the line `rowdata = [filename, self.frameWidth, self.frameHeight, 'hand', xmin, ymin, xmax, ymax]`

At present, as there are a number of hard-coded lines, please change the following lines in `HandDetection.py` to suit your purpose.

- `hand_detector = HandDetector('resources/???.mp4')` to your video file in MP4 format
- `self.cntImg = 1` to an integer that you want to start from for the generation of the saved image filename.
- `self.csvholder = [['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]` is used for generating the csv header at the first time you run the program. After you have run the program once, use `self.csvholder = []` instead.
- `with open('hand_train.csv', 'w') as file:` will create a csv file for the first time you run the program. If you want to append data to this file onwards, use `with open('hand_train.csv','a') as file:` instead.

You may discard a frame within 2 seconds after a hand is detected in the frame by pressing the `x` hotkey after you have activated any one of the debug window while the program is running. You may change the time period for decision making by modifying the number in the line `if cv2.waitKey(2000) & 0xFF == ord('x'):`.

You may call `visualize_image_bbox.py` seperately to visualize the bounding box for each image saved under the image folder.