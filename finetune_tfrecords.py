"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from alexnet import AlexNet
from datagenerator_tfrecords import ImageDataGenerator
from datetime import datetime
Iterator = tf.data.Iterator
import time

"""
Configuration Part.
"""

# Path to the textfiles for the trainings and validation set
train_file = 'devkit/train/shuffled_training_ground_truth_bboxOnly.txt'
val_file = 'devkit/validation/validation_ground_truth.txt'

# Learning params
learning_rate = 0.001
num_epochs = 30
batch_size = 128
occlusion_ratio = float(sys.argv[1])

# Network params
dropout_rate = 0.5
num_classes = 1000
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']

# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "tmp/finetune_alexnet/tensorboard"
checkpoint_path = "tmp/finetune_alexnet/checkpoints"

"""
Main Part of the finetuning Script.
"""

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True,
                                 occlusionRatio=occlusion_ratio)
    val_data = ImageDataGenerator(mode='inference',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True,
                                 occlusionRatio=occlusion_ratio)
    tr_iterator = tr_data.iterator
    val_iterator = val_data.iterator
    tr_next_batch = tr_iterator.get_next()
    val_next_batch = val_iterator.get_next()

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3],name="image_input")
y = tf.placeholder(tf.float32, [batch_size, num_classes],name="label_input")
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)


# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    label = tf.argmax(y,-1)
    label2 = tf.cast(tf.expand_dims(label,1),tf.int32)
    _top5prob,top5index = tf.nn.top_k(score,5,name = "top5index")
    top5acc = tf.reduce_sum(tf.cast(tf.equal(label2,top5index),tf.int32),-1)
    top5accMean = tf.reduce_mean(tf.cast(top5acc,tf.float32))
    top1acc = tf.equal(tf.argmax(score, -1), label)
    top1accMean = tf.reduce_mean(tf.cast(top1acc, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('Top1 Accuracy', top1accMean)
tf.summary.scalar('Top5 Accuracy', top5accMean)
tf.summary.image('Pre-processed image', x,max_outputs=2)
# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver_top1 = tf.train.Saver(max_to_keep=3)
saver_top5 = tf.train.Saver(max_to_keep=3)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = int(np.floor(544545/batch_size))
val_batches_per_epoch = int(np.floor(50000/ batch_size))

# Start Tensorflow session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4
with tf.Session(config=config) as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    bestCheckpoints = [[],[]]
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        # Initialize iterator with the training dataset

        for step in range(train_batches_per_epoch):

            # get next batch of data
            #start_time = time.time()
            img_batch, label_batch = sess.run(tr_next_batch)
            #end_time1 = time.time()
            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})
            #end_time2 = time.time()
            #print("Batching: {}".format(end_time1-start_time))
            #print("Training Op: {}".format(end_time2-end_time1))

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))
        top1val_acc = 0.
        top5val_acc = 0.
        for _ in range(val_batches_per_epoch):

            img_batch, label_batch = sess.run(val_next_batch)
            top1,top5 = sess.run([top1accMean,top5accMean], feed_dict={x: img_batch,
                                                y: label_batch,
                                                keep_prob: 1.})
            top1val_acc += top1
            top5val_acc += top5
        top1val_acc /= val_batches_per_epoch
        top5val_acc /= val_batches_per_epoch
        print("{} Top 1 Validation Accuracy = {:.4f}".format(datetime.now(),
                                                       top1val_acc))
        print("{} Top 5 Validation Accuracy = {:.4f}".format(datetime.now(),top5val_acc))

        if (len(bestCheckpoints[0]) < 3):
          bestCheckpoints[0].append(top1val_acc)
          bestCheckpoints[1].append(top5val_acc)
          bestCheckpoints[0].sort()
          bestCheckpoints[1].sort()
	  print("{} Saving checkpoint of model...".format(datetime.now()))
          # save checkpoint of the model
          checkpoint_name = os.path.join(checkpoint_path,
                                         str(occlusion_ratio) + '_cropCenter_model_epoch'+str(epoch+1)+'top1.ckpt')
          save_path = saver_top1.save(sess, checkpoint_name)

          print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                         checkpoint_name))
	  print("{} Saving checkpoint of model...".format(datetime.now()))
          # save checkpoint of the model
          checkpoint_name = os.path.join(checkpoint_path,
                                         str(occlusion_ratio) + '_cropCenter_model_epoch'+str(epoch+1)+'top5.ckpt')
          save_path = saver_top5.save(sess, checkpoint_name)

          print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                         checkpoint_name))
	if (bestCheckpoints[0][0] < top1val_acc):
          bestCheckpoints[0][0] = top1val_acc
          bestCheckpoints[0].sort()
          print("{} Saving checkpoint of model...".format(datetime.now()))
          # save checkpoint of the model
          checkpoint_name = os.path.join(checkpoint_path,
                                         str(occlusion_ratio) + '_cropCenter_model_epoch'+str(epoch+1)+'top1.ckpt')
          save_path = saver_top1.save(sess, checkpoint_name)

          print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                         checkpoint_name))
          sys.stdout.flush()
        if bestCheckpoints[1][0] < top5val_acc:
          bestCheckpoints[1][0] = top1val_acc
          bestCheckpoints[1].sort()
          print("{} Saving checkpoint of model...".format(datetime.now()))
          # save checkpoint of the model
          checkpoint_name = os.path.join(checkpoint_path,
                                         str(occlusion_ratio) + '_cropCenter_model_epoch'+str(epoch+1)+'top5.ckpt')
          save_path = saver_top5.save(sess, checkpoint_name)

          print("{} Model checkpoint saved at {}".format(datetime.now(),
                                                         checkpoint_name))
          sys.stdout.flush()

