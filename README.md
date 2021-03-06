# Auto Hand Dataset Generator for TensorFlow Object Detection Model training

This program is specialized for automatically detecting **ONE** hand per frame in a MP4 video using OpenCV API.

During the execution of the program, you have to choose whether you would like to save the frame or not by pressing one of the hotkeys. When the program terminates, a csv file will be generated. Run other consecutive programs to generate other csv files and TFRecord files which can be used for training and testing a TensorFlow hand detection model.

## Environment
This program has been successfully run using python2.7.

## Dependencies
Remember to `pip install cv2 numpy csv pandas scikit-learn` to get all the dependencies ready before the execution of the program.

## Label for TensorFlow Object Detection model
Currently only one label - `hand` is supported. See the line `rowdata = [filename, final_w, final_h, 'hand', xmin, ymin, xmax, ymax]`

## Structure of files and directories created after the execution of the programs
```
    hand_images/ --> holding all the images to be split into training/testing images

    hand_labels.csv --> holding all the information of the images and coordinates of the hand bounding boxes in the images

    hand_my_dataset/
        |
        |--- hand_test/
        |   L hand_test_labels.csv
        |   L all the images for testing
        |
        |--- hand_train/
        |   L hand_train_labels.csv
        |   L all the images for training
        |
        --- hand_eval.record
        |
        --- hand_train.record
```

## Essential Operations

### Run `python HandDetection.py`, the main program
This program detects hand in frames from a video source using OpenCV. If a hand is detected in a frame, the frame can be saved in JPEG format under the `image` directory and the coordinates of the bounding box for the hand in the frame will be recorded. After the detection of hand in the video frames finished, a csv file `hand_label.csv` will be generated.

Please change `?.mp4` in the line `hand_detector = HandDetector('resources/?.mp4')` to your video file in MP4 format. You are advised to put all the video sources under the resources directory.

After a hand is detected in the frame, you can save the frame by activating any one of the debug window and pressing the `s` hotkey twice. You can also press the `x` hotkey once to discard the frame. Otherwise, after 8 seconds, the frame will be discarded automatically. You may change the time period for making a decision by modifying the number in ms i.e. 8000 in the lines `if cv2.waitKey(8000) & 0xFF == ord('x'):` and `elif cv2.waitKey(8000) & 0xFF == ord('s'):`.

Before the frame is saved, it will be automatically resized to a size of 1280 * 720. You may change your desired frame width and height by changing the number of the variables `self.desiredFrameWidth` and `self.desiredFrameHeight`.

#### Customize the program
You may change the variable `self.image_dir` to create another directory with a different name for holding all the JPEG images which are going to be saved during the execution of the program. The default name of the directory is `image`.

You may change the variable `self.csvFilename` to create another csv file for recording all the information about the image and the coordinates of the bounding box indicating the position of a hand in the image. The default name of the csv file is `hand_label.csv`.

### Run `python split_train_test_csv_images.py` afterwards
This program splits the images in the `image` directory into 2 separated directories and corresponding image data in the `hand_label.csv` csv file into 2 separated csv files. By default a dataset directory called `hand_my_dataset` will be created. Under this dataset directory, there are two subdirectories. `hand_train` is intended to be used for training while `hand_test` is intended to be used for testing during the hand detection model training process.

#### Customize the program
You may change the names of the images, training, testing directories and the csv files by modifying the variables located at the beginning of the program.

### Run `python generate_tfrecord.py` at last
This program generates a TFRecord file either based on the train or test csv file, i.e. `hand_train_labels.csv` or `hand_test_labels.csv`.

Run the program twice to generate 2 TFRecord files, one for training and another one for testing. Change the name of the files and directories by modifying values of the variables located at the beginning of the program.


## Optional Operation
Run `visualize_image_bbox.py` seperately to visualize the bounding box for each image saved under the `image`, `hand_train` and `hand_test` directories.


## Evaluations
### Limitations
At present, only **ONE** hand can be detected per frame.

The bounding box indicating the position of the hand (cropped area) cannot perfectly fit in the size of the hand i.e. the bounding box is bigger than the size of the hand and includes other objects surrounding a hand.

If a hand appears to be too big in the frame, it can hardly be detected or only a few fingers can be detected. Same problem happends if the background of the video or the objects surrounding the hands has a very similar color to skin.

### Possible Solutions to enhance accuracy for hand detection in video frames using OpenCV
A hand which appears to be smaller in size in each frame can be detected with a higher accuracy i.e. the bounding box can fit in the size of the hand.

Avoid choosing a video in which hands appeared to be too big as well as the color of the background and the objects surrounding the hands has a similar color to skin.

## Troubleshooting
For Mac users, be aware of the `.DS_Store` file which might appear in the dataset directory i.e. `hand_my_dataset` as well as `hand_train` and `hand_test` directories. Please run `ls -la` in the terminal to check if the `.DS_Store` file exists in the mentioned directories. If any, delete the file **before generating the TFRecord files**.
