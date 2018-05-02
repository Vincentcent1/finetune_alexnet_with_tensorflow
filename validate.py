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
val_file = 'devkit/validation/validation_ground_truth.txt'

batch_size = 128
num_classes = 1000

with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
                                  shuffle=False,
                                  occlusionRatio=occlusion_ratio)

    # create an reinitializable iterator given the dataset structure
    iterator = Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
    next_batch = iterator.get_next()
validation_init_op = iterator.make_initializer(val_data.data)
#x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
#y = tf.placeholder(tf.float32, [batch_size, num_classes])
#keep_prob = tf.placeholder(tf.float32)

checkpoint_path = "tmp/finetune_alexnet/checkpoints"

checkpoint_name = os.path.join(checkpoint_path, str(occlusion_ratio) + 'model_epoch'+str(epoch)+'.ckpt')
meta_name = os.path.join(checkpoint_path, str(occlusion_ratio) + 'model_epoch' + str(epoch) + '.ckpt.meta')

print(checkpoint_name)
print(meta_name)


val_batches_per_epoch = int(np.floor(val_data.data_size / batch_size))
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
  saver = tf.train.import_meta_graph(meta_name)
  saver.restore(sess, checkpoint_name)
  graph = tf.get_default_graph()
  all_ops = graph.get_operations()
  for op in all_ops:
    print(str(op.name))
  # allVars = tf.global_variables()
  #for var in allVars:
   # print(var)
  sess.run(tf.initialize_all_variables())
  accuracy = graph.get_tensor_by_name("accuracy/Mean:0")
  y = graph.get_tensor_by_name("Placeholder_1:0")
  keep_prob = graph.get_tensor_by_name("Placeholder_2:0")
  x = graph.get_tensor_by_name("Placeholder:0")
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
