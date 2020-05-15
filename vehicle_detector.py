# vehicle_detector.py

# https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models

# https://gpuopen.com/rocm-tensorflow-1-8-release/
#  py -m pip install Cython contextlib2 pillow lxml jupyter matplotlib
# http://www.mingw.org/wiki/Getting_Started

import os
import six.moves.urllib as urllib
import sys
import tarfile
import time
from collections import defaultdict


# Installed
import cv2
import tensorflow as tf
import numpy as np

# Project
from grabscreen import grab_screen
import keys as k
from getkeys import key_check

# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

keys = k.Keys({})


# ## Model preparation
# What model to download.
MODEL_NAME = "ssd_mobilenet_v1_coco_11_06_2017"
MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + "/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("data", "mscoco_label_map.pbtxt")

NUM_CLASSES = 90

# ## Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if "frozen_inference_graph.pb" in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name="")


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

# Pause options
paused = False
pause_control = True
debug = False

last_ran = time.time()


def determine_movement(mid_x, mid_y, width=1280, height=705, x_sens=5, y_sens=5):
    global debug
    global last_ran
    global pause_control

    x_move = mid_x - 0.5
    y_move = mid_y - 0.5
    how_much_to_move_x = x_move / 0.5
    how_much_to_move_y = y_move / 0.5

    # Move mouse to point
    loc_x = how_much_to_move_x * width
    loc_y = how_much_to_move_y * height

    loc_x = int(loc_x / x_sens)
    loc_y = int(loc_y / y_sens)

    print(f"x: {loc_x}, y: {loc_y}")

    if debug:
        print(time.time())
        print(last_ran - time.time())

    time_dif = time.time() - last_ran

    if time_dif > 0.01:
        last_ran = time.time()

        if not pause_control:
            keys.directMouse(loc_x, loc_y)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        # for i in list(range(4))[::-1]:
        # print(i + 1)
        # time.sleep(1)

        while True:

            # screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
            screen = cv2.resize(grab_screen(region=(0, 40, 1280, 745)), (800, 450))
            image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name("detection_scores:0")
            classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded},
            )
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            if not paused:

                # Store found vehicals
                vehicle_dict = {}
                # FIND OBJECTS
                for i, b in enumerate(boxes[0]):
                    # https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt

                    if (
                        # person
                        classes[0][i]
                        == 1
                        # car
                        # classes[0][i] == 3
                        # bus
                        # or classes[0][i] == 6
                        # truck
                        # or classes[0][i] == 8
                    ):

                        if scores[0][i] > 0.5:
                            # print("Found: More than 50% confidant")
                            # More than 50% confidant
                            mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
                            mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
                            apx_distance = round(
                                (1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4, 1
                            )

                            # Add found vehicals
                            vehicle_dict[apx_distance] = [mid_x, mid_y, scores[0][i]]

                            # label found objects
                            cv2.putText(
                                image_np,
                                "{}".format(apx_distance),
                                (int(mid_x * 800), int(mid_y * 450)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            # if apx_distance <= 0.5:
                            #     if mid_x > 0.3 and mid_x < 0.7:
                            #         cv2.putText(
                            #             image_np,
                            #             "ENEMY!!!",
                            #             (int(mid_x * 800), int(mid_y * 450)),
                            #             cv2.FONT_HERSHEY_SIMPLEX,
                            #             1.0,
                            #             (0, 0, 255),
                            #             3,
                            #         )

                    # center on it
                    if len(vehicle_dict) > 0:
                        # Center Target
                        closest = sorted(vehicle_dict.keys())[0]
                        vehicle_choice = vehicle_dict[closest]
                        determine_movement(
                            mid_x=vehicle_choice[0], mid_y=vehicle_choice[1]
                        )

                        if not pause_control:
                            # ADS
                            keys.directMouse(buttons=keys.mouse_rb_press)
                            # Fire
                            keys.directMouse(buttons=keys.mouse_lb_press)

                    else:
                        if not pause_control:
                            # Stop ADS
                            keys.directMouse(buttons=keys.mouse_rb_release)
                            # Stop Fire
                            keys.directMouse(buttons=keys.mouse_lb_release)

            # SHOW OUTPUT IN WINDOW
            cv2.imshow("window", image_np)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

            # check for manual input
            input_keys = key_check()

            # t pauses game and can get annoying.
            if "T" in input_keys:
                if paused:
                    paused = False
                else:
                    paused = True

                print("PAUSE: T", paused)

                time.sleep(0.5)
                keys.directMouse(buttons=keys.mouse_rb_release)
                keys.directMouse(buttons=keys.mouse_lb_release)

            # t pauses game and can get annoying.
            if "Y" in input_keys:
                if pause_control:
                    pause_control = False
                else:
                    pause_control = True

                print("PAUSE: Y", pause_control)

                time.sleep(0.5)
                keys.directMouse(buttons=keys.mouse_rb_release)
                keys.directMouse(buttons=keys.mouse_lb_release)
