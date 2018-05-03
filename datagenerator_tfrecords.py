# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import os
import time

Dataset = tf.data.Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import occlude_black_rectangle as obc


IMAGENET_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32) # BGR




class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, mode, batch_size, num_classes, shuffle=True,
                 buffer_size=1000, occlusionRatio=0):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            occlusionRatio: The percentage occlusion used for the occlude function
        """
        self.iterator = iterator

    def read_tfrecords(self):
        filenames = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(...)  # Parse the record into tensors.
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.batch(32)
        iterator = dataset.make_initializable_iterator()

        # You can feed the initializer with the appropriate filenames for the current
        # phase of execution, e.g. training vs. validation.

        # Initialize `iterator` with training data.
        training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

        # Initialize `iterator` with validation data.
        validation_filenames = ["/var/data/validation1.tfrecord", ...]
        sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
