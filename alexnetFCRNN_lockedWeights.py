"""This is an TensorFLow implementation of AlexNet by Alex Krizhevsky at all.

Paper:
(http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Explanation can be found in my blog post:
https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

This script enables finetuning AlexNet on any given Dataset with any number of
classes. The structure of this script is strongly inspired by the fast.ai
Deep Learning class by Jeremy Howard and Rachel Thomas, especially their vgg16
finetuning script:
Link:
- https://github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py


The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/

@author: Frederik Kratzert (contact: f.kratzert(at)gmail.com)
"""

import tensorflow as tf
import numpy as np


class AlexNet(object):
    """Implementation of the AlexNet."""

    def __init__(self, x, keep_prob, num_classes, train_layer,batch_size=128
                ,weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            train_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.train_layer = train_layer
        self.batch_size = batch_size

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        """Create the network graph."""
        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = lrn(conv1, 2, 1e-04, 0.75, name='norm1')
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = lrn(conv2, 2, 1e-04, 0.75, name='norm2')
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # print(conv5.shape)
        flattened = tf.reshape(pool5, [-1,5,6*6*256])


        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.KEEP_PROB)

        print(conv1.shape)
        print(pool1.shape)
        print(conv2.shape)
        print(pool2.shape)
        print(conv3.shape)
        print(conv4.shape)
        print(conv5.shape)
        print(pool5.shape)
        print(self.X.shape)
        print(flattened.shape)
	print(dropout7.shape)
        lstm1 = tf.contrib.rnn.LSTMCell(128,name="lstm")
        # hidden_state1 = tf.zeros([self.batch_size, 5])
        # current_state1 = tf.zeros([self.batch_size, 5])
        state1 = lstm1.zero_state(self.batch_size,tf.float32)


        for step in range(5):
            output1,state1 = lstm1(dropout7[:,step],state1)
        print(output1.shape)

        # 8th Layer: FC and return unscaled activations
        self.fc8 = fc(output1, 128, self.NUM_CLASSES, lastLayer=True, name='fc8')

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if layer should be trained from scratch
            # If they are in the skip layer, by right skip the loading of weights, but in this case we
            # want to load the weights, but with a boolean flag that tells the network that this node is trainable
            if op_name in self.train_layer:
                with tf.variable_scope(op_name, reuse=tf.AUTO_REUSE):
                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def conv(xFive, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    reluList = []
    for x in range(5):
        # Get number of input channels
        input_channels = int(xFive[:,x].get_shape()[-1])

        # Create lambda function for the convolution. 'name' is used to label the operation for easier retrieval or it is just useless i think.
        convolve = lambda i, k, name : tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding,
                                             name=name)

        # Name scope is used to retrieve same weights and biases for the same convolutional layer. Hence reusing the weights and biases when the existing name (layer) already exists.
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels/groups,
                                                        num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(xFive[:,x], weights,"rcl")

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=xFive[:,x])
            weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
            output_groups = [convolve(i, k,"rcl" + str(count)) for count,(i, k) in enumerate(zip(input_groups, weight_groups))]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)
        reluList.append(relu)
    return tf.stack(reluList,axis=1)


def fc(x, num_in, num_out, name, isRelu=True, lastLayer=False):
    """Create a fully connected layer."""
    reluList = []
    if lastLayer: # If this is the last layer (before softmax), we don't need to do it 5 times because this is after the RNN.
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:

        # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
	    return act
    for step in range(5):

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scope:

        # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x[:,step], weights, biases, name=scope.name)

        if isRelu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
	    reluList.append(relu)
        else:
	    reluList.append(act)
    return tf.stack(reluList,axis=1)


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    out = []
    for i in range(5):
        out.append(tf.nn.max_pool(x[:,i], ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name))
    return tf.stack(out,axis=1)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    out = []
    for i in range(5):
        out.append(tf.nn.local_response_normalization(x[:,i], depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name))

    return tf.stack(out,axis=1)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

# def RNN(X,lstm1):
#     for i in range(5):
#         conv1 = conv(X[i], 11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         norm1 = lrn(conv1, 2, 1e-04, 0.75, name='norm1')
#         pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
#         output1,state1 = lstm1(pool1,state1)
#     return output1,state1
