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
            mode: Either 'training' or 'inference'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            occlusionRatio: The percentage occlusion used for the occlude function
        """
        self.num_classes = num_classes
        self.occlusionRatio = occlusionRatio
        self.filenames = []
        if mode == 'training':
            self.filenames = tf.gfile.Glob('tfrecords/train/*')
        elif mode == 'inference':
            self.filenames = tf.gfile.Glob('tfrecords/val/*')
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_tfrecords,num_parallel_calls=40)  # Parse the record into tensors.
        dataset = dataset.map(self.cropAndOccludeCenter,num_parallel_calls=40)  # Parse the record into tensors.
        dataset = dataset.repeat()  # Repeat the input indefinitely.
        dataset = dataset.batch(batch_size)

        self.iterator = dataset.make_one_shot_iterator()

    def _parse_tfrecords(self,example_proto):
        '''
        @params:
            example_proto: raw example proto serialized data
        @return:
            parsed_features: dictionary of features to tensor
        '''
        keys_to_features = {
            'image/height': tf.FixedLenFeature((),tf.int64),
            'image/width': tf.FixedLenFeature((),tf.int64),
            'image/channels': tf.FixedLenFeature((),tf.int64),
            'image/class/label': tf.FixedLenFeature((),tf.int64),
            'image/class/synset': tf.FixedLenFeature((),tf.string),
            'image/class/text': tf.FixedLenFeature((),tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/filename': tf.FixedLenFeature((),tf.string),
            'image/encoded': tf.FixedLenFeature((),tf.string)
            }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features

    def cropAndOccludeCenter(self, parsed_features):
        '''
        Crop an image to the bounding box and occlude on the center of the image by @occlusion.
        @params:
            parsed_feature: dict of features:
        @var:
            image_data: JPEG image data in String
            shape: shape of the image
            bbox: List of bounding boxes tuple (xmin,ymin,xmax,ymax)
            occlusion: occlusionRatio
        @return:

        '''
        image_data = parsed_features['image/encoded']
        shape = (parsed_features['image/height'],
                 parsed_features['image/width'],
                 parsed_features['image/channels'])
        bbox = (tf.sparse_tensor_to_dense(parsed_features['image/object/bbox/xmin']),
                 tf.sparse_tensor_to_dense(parsed_features['image/object/bbox/xmax']),
                 tf.sparse_tensor_to_dense(parsed_features['image/object/bbox/ymin']),
                 tf.sparse_tensor_to_dense(parsed_features['image/object/bbox/ymax']))

        height = shape[0]
        width = shape[1]
        tf.assert_equal(shape[2],tf.constant([3],shape[2].dtype),message="Channels not equal 3")
        # tf.assert_equal(tf.size(bbox),tf.constant([4],tf.int32),message="Bbox size is not 4")
        xmin_scaled = bbox[0][0]
        xmax_scaled = bbox[1][0]
        ymin_scaled = bbox[2][0]
        ymax_scaled = bbox[3][0]

        offset_height = tf.cast(ymin_scaled*tf.to_float(height),tf.int32)
        offset_width = tf.cast(xmin_scaled*tf.to_float(width),tf.int32)
        target_height = tf.cast((ymax_scaled - ymin_scaled)*tf.to_float(height),tf.int32)
        target_width = tf.cast((xmax_scaled - xmin_scaled)*tf.to_float(width),tf.int32)

        imageCropped = tf.image.decode_and_crop_jpeg(image_data,[offset_height,offset_width,target_height,target_width])
        imageResized = tf.image.resize_images(imageCropped, [227,227],tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        side = tf.sqrt(self.occlusionRatio)
        offset_height2 = tf.cast(((1.0-side)/2)*227,tf.int32)
        target_height2 = tf.cast((side)*227,tf.int32)

        imageOccluded = self.occlude(imageResized,offset_height2,offset_height2,target_height2,target_height2)
        # RGB -> BGR
        bgrImageOccluded = imageOccluded[:,:,::-1]
        image = tf.subtract(tf.cast(bgrImageOccluded,tf.float32),IMAGENET_MEAN)
        data = {}
        # data['image_data'] = image
        label = tf.cast(parsed_features['image/class/label'],tf.int32)
        # data['synset'] = parsed_features['image/class/synset']
        # data['text'] = parsed_features['image/class/text']
        # data['filename'] = parsed_features['image/filename']
        one_hot = tf.one_hot(label,self.num_classes)
        return image,label

    def occlude(self, image, offset_height, offset_width, target_height, target_width):
    # tf.assert_equal (image.shape,[227,227,3])
        occludedRegion = tf.zeros([target_height,target_width,3],tf.int32)
        topPad = tf.ones([offset_height,target_width,3],tf.int32)
        occludedRegion = tf.concat([topPad,occludedRegion],0)
        leftPad = tf.ones([target_height + offset_height,offset_width,3],tf.int32)
        occludedRegion = tf.concat([leftPad,occludedRegion],1)
        bottomPad = tf.ones([227-offset_height-target_height,offset_width + target_width,3],tf.int32)
        occludedRegion = tf.concat([occludedRegion,bottomPad],0)
        rightPad = tf.ones([227,227-offset_width-target_width,3],tf.int32)
        occludedRegion  = tf.concat([occludedRegion,rightPad],1)
        occludedImage = tf.multiply(image, tf.cast(occludedRegion,tf.uint8))
        return occludedImage
