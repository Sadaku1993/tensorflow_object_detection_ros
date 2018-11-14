#coding:utf-8

import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Mamually Install
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# Load a Tensorflow mode into memory
def load_frozenmodel():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def load_labelmap():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index

class ObjectDetection(object):
    def __init__(self, detection_graph, category_index):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.imageCallback, queue_size=10)
        self.image_pub = rospy.Publisher("/object_detection/image", Image, queue_size=10)
        self.detection_graph = detection_graph
        self.category_index  = category_index

    def imageCallback(self, image_msg):
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            print (e)

    def main(self):
        rospy.init_node("object_detection_ros")
        rate = rospy.Rate(30)
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while not rospy.is_shutdown():
                    image_np = self.cv_image
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                    ]:
                      tensor_name = key + ':0'
                      if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                    if 'detection_masks' in tensor_dict:
                      # The following processing is only for single image
                      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                          detection_masks, detection_boxes, image_np.shape[0], image_np.shape[1])
                      detection_masks_reframed = tf.cast(
                          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                      # Follow the convention by adding back the batch dimension
                      tensor_dict['detection_masks'] = tf.expand_dims(
                          detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    # Run inference
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                      output_dict['detection_masks'] = output_dict['detection_masks'][0]
                    
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        self.category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                           
                    image_height = self.cv_image.shape[0]
                    image_width  = self.cv_image.shape[1]
                    resize_image = cv2.resize(image_np, (image_width, image_height))
                    pub_image = CvBridge().cv2_to_imgmsg(resize_image, "bgr8")
                    self.image_pub.publish(pub_image)

def main():
    # Load 
    category = load_labelmap()
    graph = load_frozenmodel()
    # Detection
    detection = ObjectDetection(graph, category)
    detection.main()

if __name__ == '__main__':
    print("start")
    main()
