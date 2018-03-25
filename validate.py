import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import time

occlusion_ratio = sys.argv[1]
epoch = sys.argv[2]

with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False,
                                  occlusionRatio=0.1)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
    next_batch = iterator.get_next()
validation_init_op = iterator.make_initializer(val_data.data)

checkpoint_path = "tmp/finetune_alexnet/checkpoints"
checkpoint_name = os.path.join(checkpoint_path, str(occlusion_ratio) + 'model_epoch'+str(epoch)+'.ckpt')

saver = tf.train.Saver()





with tf.Session() as sess:
  saver.restore(sess, checkpoint_name)
  print("{} Start validation".format(datetime.now()))
  sess.run(validation_init_op)
  test_acc = 0.
  test_count = 0
  for _ in range(val_batches_per_epoch):
    img_batch, label_batch = sess.run(next_batch)
    acc = sess.run(accuracy, feed_dict={x: img_batch,
                                        y: label_batch,
                                        keep_prob: 1.})
    test_acc += acc
    test_count += 1
  test_acc /= test_count
  print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))
