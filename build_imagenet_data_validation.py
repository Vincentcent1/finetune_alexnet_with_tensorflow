from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys

import numpy as np
import six
import tensorflow as tf

tf.app.flags.DEFINE_string('labels_file',
                                                     'devkit/wnetClass/wnetId_orderedBy_classId.txt',
                                                     'Labels file')

tf.app.flags.DEFINE_string('imagenet_metadata_file',
                                                     'devkit/wnetClass/wnetId_to_className_mapping.txt',
                                                     'ImageNet metadata file')
tf.app.flags.DEFINE_string('validation_directory', 'Data/CLS-LOC/val',
                                                     'Validation data directory')
tf.app.flags.DEFINE_string('bbox_file', 'devkit/validation/validation_bbox.txt','Validation bounding box file list')
tf.app.flags.DEFINE_string('bbox_dir', 'Annotations/CLS-LOC/val','Validation bounding box directory')

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config)

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                                                    feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                                                    feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                                                     feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    """Process a single image file.
    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, JPEG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Clean the dirty data.
    if _is_png(filename):
        # 1 image is a PNG.
        print('Converting PNG to JPEG for %s' % filename)
        image_data = coder.png_to_jpeg(image_data)
    elif _is_cmyk(filename):
        # 22 JPEG images are in CMYK colorspace.
        print('Converting CMYK to RGB for %s' % filename)
        image_data = coder.cmyk_to_rgb(image_data)

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _convert_to_example(filename, image_buffer, label, synset, human, bbox, height, width):
    """Build an Example proto for an example.
    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, JPEG encoding of RGB image
        label: integer, identifier for the ground truth for the network
        synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
        human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
        bbox: list of bounding boxes; each box is a list of integers
            specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
            the same label as the image label.
        height: integer, image height in pixels
        width: integer, image width in pixels
    Returns:
        Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bbox:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
        # pylint: enable=expression-not-assigned

    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/colorspace': _bytes_feature(colorspace),
            'image/channels': _int64_feature(channels),
            'image/class/label': _int64_feature(label),
            'image/class/synset': _bytes_feature(synset),
            'image/class/text': _bytes_feature(human),
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/bbox/label': _int64_feature([label] * len(xmin)),
            'image/format': _bytes_feature(image_format),
            'image/filename': _bytes_feature(os.path.basename(filename)),
            'image/encoded': _bytes_feature(image_buffer)}))
    return example

def _find_human_readable_labels(synsets, synset_to_human):
    """Build a list of human-readable labels.
    Args:
        synsets: list of strings; each string is a unique WordNet ID.
        synset_to_human: dict of synset to human labels, e.g.,
            'n02119022' --> 'red fox, Vulpes vulpes'
    Returns:
        List of human-readable strings corresponding to each synset.
    """
    humans = []
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    return humans

def _build_synset_lookup(imagenet_metadata_file):
    """Build lookup for synset to human-readable label.
    Args:
        imagenet_metadata_file: string, path to file containing mapping from
            synset to human-readable label.
            Assumes each line of the file looks like:
                n02119247    black fox
                n02119359    silver fox
                n02119477    red fox, Vulpes fulva
            where each line corresponds to a unique mapping. Note that each line is
            formatted as <synset>\t<human readable label>.
    Returns:
        Dictionary of synset to human labels, such as:
            'n02119022' --> 'red fox, Vulpes vulpes'
    """
    lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if l:
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human
    return synset_to_human

def ProcessXMLAnnotation(xml_file):
  """Process a single XML file containing a bounding box."""
  # pylint: disable=broad-except
  try:
    tree = ET.parse(xml_file)
  except Exception:
    print('Failed to parse: ' + xml_file, file=sys.stderr)
    return None
  # pylint: enable=broad-except
  root = tree.getroot()

  num_boxes = FindNumberBoundingBoxes(root)
  boxes = []

  for index in range(num_boxes):
    box = BoundingBox()
    # Grab the 'index' annotation.
    box.xmin = GetInt('xmin', root, index)
    box.ymin = GetInt('ymin', root, index)
    box.xmax = GetInt('xmax', root, index)
    box.ymax = GetInt('ymax', root, index)

    box.width = GetInt('width', root)
    box.height = GetInt('height', root)
    box.filename = GetItem('filename', root) + '.JPEG'
    box.label = GetItem('name', root)

    xmin = float(box.xmin) / float(box.width)
    xmax = float(box.xmax) / float(box.width)
    ymin = float(box.ymin) / float(box.height)
    ymax = float(box.ymax) / float(box.height)

    # Some images contain bounding box annotations that
    # extend outside of the supplied image. See, e.g.
    # n03127925/n03127925_147.xml
    # Additionally, for some bounding boxes, the min > max
    # or the box is entirely outside of the image.
    min_x = min(xmin, xmax)
    max_x = max(xmin, xmax)
    box.xmin_scaled = min(max(min_x, 0.0), 1.0)
    box.xmax_scaled = min(max(max_x, 0.0), 1.0)

    min_y = min(ymin, ymax)
    max_y = max(ymin, ymax)
    box.ymin_scaled = min(max(min_y, 0.0), 1.0)
    box.ymax_scaled = min(max(max_y, 0.0), 1.0)

    boxes.append(box)

def FindNumberBoundingBoxes(root):
  index = 0
  while True:
    if GetInt('xmin', root, index) == -1:
      break
    index += 1
  return index

if __name__ == '__main__':
    labels = []
    filenames = []

    challenge_synsets = ['0'] # wnetId Ordered by label
    challenge_synsets.extend([l.strip() for l in tf.gfile.FastGFile(FLAGS.labels_file, 'r').readlines()])

    synset_to_human = _build_synset_lookup(FLAGS.imagenet_metadata_file)
    humans = _find_human_readable_labels(challenge_synsets, synset_to_human)

    for l in tf.gfile.FastGFile(FLAGS.ground_truth, 'r').readlines():
        a,b = l.split(' ')
        filenames.append(os.path.join(FLAGS.validation_directory,a))
        labels.append(int(b)+1)

    bbox_path = [os.path.join(FLAGS.bbox_dir,(l.strip().split(' ')[1])) for l in tf.gfile.FastGFile(FLAGS.bbox_file, 'r').readlines()]:

    num_of_images_per_batch = (len(filenames)/128)

    for i in range(len(filenames)):
        if not i % num_of_images_per_batch:
            writer = tf.python_io.TFRecordWriter("validation%d" % (i/num_of_images_per_batch))
        filename = filenames[i]
        label = labels[i]
        synset = challenge_synsets[label]
        bbox = ProcessXMLAnnotation(bbox_path[i])
        assert len(bbox) > 0 , ("Error... Empty bbox found for %s" % (filename))
        human = humans[label]
        coder = ImageCoder()
        image_buffer, height, width = _process_image(filename,coder)
        example = _convert_to_example(filename, image_buffer,label, synset,human,bbox,height,width)
        writer.write(example.SerializeToString())

