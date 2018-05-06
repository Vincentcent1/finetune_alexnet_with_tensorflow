



import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def cropAndOccludeCenter(image_data, bbox, occlusion):
    '''
    Crop an image to the bounding box and occlude on the center of the image by @occlusion.
    @params:
        image_data: JPEG image data in String
        bbox: List of bounding boxes tuple (xmin,ymin,xmax,ymax)
        occlusion: occlusionRatio
    '''
    image = tf.image.decode_jpeg(image_data, channels=3)
    assert len(bbox) == 4, "Bbox tuple has size not equal to 4"
    xmin_scaled = bbox[0]
    ymin_scaled = bbox[1]
    xmax_scaled = bbox[2]
    ymax_scaled = bbox[3]

    shape = tf.shape(image).eval()
    height = shape[0]
    width = shape[1]
    assert shape[2] == 3, "More than 3 channels detected"

    offset_height = int(ymin_scaled*height)
    offset_width = int(xmin_scaled*width)
    target_height = int((ymax_scaled - ymin_scaled)*height)
    target_width = int((xmax_scaled - xmin_scaled)*width)

    imageCropped = tf.image.crop_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    imageResized = tf.image.resize_images(imageCropped, [227,227],tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    side = tf.sqrt(occlusion).eval()
    offset_height2 = int(((1-side)/2)*227)
    target_height2 = int((side)*227)

    imageOccluded = occlude(imageResized,offset_height2,offset_height2,target_height2,target_height2)

    return imageOccluded




def occlude(image, offset_height, offset_width, target_height, target_width):
    tf.assert_equal (image.shape,[227,227,3])
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





# %%timeit
with tf.Session() as sess:
    with sess.as_default():
        imageCropped= cropAndOccludeCenter(image_data,[xmin,ymin,xmax,ymax], 0.5)
        cropped = sess.run(imageCropped)
        plt.imshow(cropped)




with tf.Session() as sess:
    image = sess.run(tf.image.decode_jpeg(image_data, channels=3))
    print(image.shape)

