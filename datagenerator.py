# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import os
import time

from tensorflow.contrib.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import occlude_black_rectangle as obc


IMAGENET_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32) # BGR

trainingPath = {"bbox":"Annotations/CLS-LOC/train/", "image":"Data/CLS-LOC/train/"}
validationPath = {"bbox":"Annotations/CLS-LOC/val/", "image":"Data/CLS-LOC/val/"}
trainingGroundTruth = "devkit/train/shuffled_training_ground_truth_bboxOnly.txt"

current_dir = os.getcwd() #This has to be the root folder of ILSVRC

# Path to the directory of the images and bounding boxes
bbox_dir_train = os.path.join(current_dir, trainingPath["bbox"])
img_dir_train = os.path.join(current_dir, trainingPath["image"])

bbox_dir_val = os.path.join(current_dir, validationPath["bbox"])
img_dir_val = os.path.join(current_dir, validationPath["image"])

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000, occlusionRatio=0):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes
        self.occlusionRatio = occlusionRatio
	self.mode = mode
        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.bbox_paths = convert_to_tensor(self.bbox_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.bbox_paths, self.labels))
        print("Dataset creation...")

        # distinguish between train/infer. when calling the parsing functions
        if mode == 'training':

            data = data.map(
                lambda img_path, bbox_path, label: tuple(tf.py_func(
                    self._occlude, [img_path, bbox_path, label], [tf.uint8, bbox_path.dtype, label.dtype])), num_parallel_calls=8)
            data = data.map(self._parse_function_train, num_parallel_calls=8)
            data.prefetch(100*batch_size)


        elif mode == 'inference':
            data = data.map(
                lambda img_path, bbox_path, label: tuple(tf.py_func(
                    self._occlude, [img_path, bbox_path, label], [tf.uint8, bbox_path.dtype, label.dtype])), num_parallel_calls=8)
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
            data.prefetch(100*batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        data = data.batch(batch_size)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.bbox_paths = []
        self.labels = []
        if self.mode == 'training':
            img_dir = img_dir_train
            bbox_dir = bbox_dir_train
        elif self.mode == 'inference':
            img_dir = img_dir_val
            bbox_dir = bbox_dir_val
        else:
            raise ValueError("Invalid mode '%s'." % (mode))
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(img_dir + items[0])
                self.bbox_paths.append(bbox_dir + items[0][:-4] + "xml")
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _occlude(self, filename, bbox, label):
        # Occlude the image
        img = obc.cropAndOccludeCenter(filename,bbox,self.occlusionRatio)
        return img, bbox, label

    def _parse_function_train(self, img, bbox, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # load and preprocess the image
        img = tf.cast(img, tf.float32)
        # img_string = tf.read_file(filename)
        # img_decoded = tf.image.decode_jpeg(img_string, channels=3)
        # Call image manipulation func here
        # img_resized = tf.image.resize_images(img_decoded, [227, 227])
        """
        Dataaugmentation comes here.
        """
        img_centered = tf.subtract(img, IMAGENET_MEAN)

        # RGB -> BGR
        # img_bgr = img_centered[:, :, ::-1]

        return img_centered, one_hot

    def _parse_function_inference(self, img, bbox, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        # load and preprocess the image
        img = tf.cast(img, tf.float32)
        # img_decoded = tf.convert_to_tensor(img)
        # img_string = tf.read_file(filename)
        # img_decoded = tf.image.decode_png(img_string, channels=3)
        # img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img, IMAGENET_MEAN)

        # RGB -> BGR
        # img_bgr = img_centered[:, :, ::-1]

        return img_centered, one_hot
