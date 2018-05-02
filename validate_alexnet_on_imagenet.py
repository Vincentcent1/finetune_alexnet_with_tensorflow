#some basic imports and setups
import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import occlude_black_rectangle as obc

'''
ILSVRC directory structure, root = ILSVRC/

Training data:
Bounding boxes: Annotations/CLS-LOC/train/<wnetid>/<wnetid>_<imageid>.xml
Image:          Data/CLS-LOC/train/<wnetid>/<wnetid>_<imageid>.JPEG

Validation data:
Bounding boxes: Annotations/CLS-LOC/val/ILSVRC2012_val_000<5-digit-imagenumber>.xml (e.g )
Image:          Data/CLS-LOC/val/ILSVRC2012_val_000<5-digit-imagenumber>.JPEG

Test data:
Bounding boxes: None
Image:          Data/CLS-LOC/test/ILSVRC2012_test_00<6-digit-imagenumber>.JPEG
'''
trainingPath = {"bbox":"Annotations/CLS-LOC/train/", "image":"Data/CLS-LOC/train/"}
validationPath = {"bbox":"Annotations/CLS-LOC/val/", "image":"Data/CLS-LOC/val/"}
testPath = {"image":"Data/CLS-LOC/test/"}
validationGroundTruth = "devkit/validation_ground_truth.txt"



#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)


current_dir = os.getcwd() #This has to be the root folder of ILSVRC
bbox_dir = os.path.join(current_dir, validationPath["bbox"])
img_dir = os.path.join(current_dir, validationPath["image"])
validation_path = os.path.join(current_dir, validationGroundTruth)
# get_ipython().magic(u'matplotlib inline')
validations = []
with open(validation_path, 'r') as f:
    for line in f:
        validations.append(int(line.split()[1]))
# In[2]:


#get list of all images
xml_files = [os.path.join(bbox_dir, f) for f in os.listdir(bbox_dir) if f[-4:].lower() == '.xml']
xml_files = np.sort(xml_files)
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f[-5:].upper() == '.JPEG']
img_files = np.sort(img_files)
# img_files = [os.path.join(image_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpeg')]

#load all images
imgs = []
occlusionPercentageList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# print("Classifying validation data with {:.0f}% occlusion".format(occlusionPercentage))



# for f in img_files:
#     imgs.append(cv2.imread(f))

#plot images
# fig = plt.figure(figsize=(15,6))
# for i, img in enumerate(imgs):
#     fig.add_subplot(1,3,i+1) # 1-indexed
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')


# First we will create placeholder for the dropout rate and the inputs and create an AlexNet object. Then we will link the activations from the last layer to the variable `score` and define an op to calculate the softmax values.

# In[3]:


from alexnet import AlexNet
from caffe_classes import class_names

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000, [])

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax
softmax = tf.nn.softmax(score)


# Now we will start a TensorFlow session and load pretrained weights into the layer weights. Then we will loop over all images and calculate the class probability for each image and plot the image again, together with the predicted class and the corresponding class probability.

# In[4]:



with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Create figure handle
    # fig2 = plt.figure(figsize=(15,6))

    # Loop over all images
    top1 = 0
    count = 0
    top5 = 0
    for occlusionRatio in occlusionPercentageList:
        print("Validation with {}% occlusion".format(occlusionRatio*100))
        for i,[imgPath,bboxPath] in enumerate(zip(img_files,xml_files)):
            imgs.append(obc.occlude(imgPath,bboxPath,occlusionRatio))
            i+=1
            if i %1000 == 0:
                # print("{} images occluded".format(i))
                for image in imgs:
                    # print(img_files[count])

                    # Convert image to float32 and resize to (227x227)
                    img = cv2.resize(image.astype(np.float32), (227,227))
                    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    # plt.show()

                    # Subtract the ImageNet mean
                    img -= imagenet_mean

                    # Reshape as needed to feed into model
                    img = img.reshape((1,227,227,3))
-
                    # Run the session and calculate the class probability
                    probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})

                    # Get the class name of the class with the highest probability
                    # class_name = class_names[np.argmax(probs)]
                    predictions = np.array(range(1000))[np.argpartition(probs[0],(-1,-2,-3,-4,-5))][-5:] # Sort the top 5 highest and get the top 5 highest predictions
                    # print(predictions)
                    # print(probs[0,np.argmax(probs)])
                    # print(validations[count])
                    # print predictions
                    if (predictions[-1] == validations[count]):
                        top1+= 1
                        top5+= 1
                    else:
                        for prediction in predictions:
                            if (prediction == validations[count]):
                                top5+= 1

                    # Plot image with class name and prob in the title
                    # fig2.add_subplot(1,3,i+1)
                    # plt.title("Class: " + class_name + ", probability: %.4f" %probs[0,np.argmax(probs)])
                    # plt.axis('off')
                    count+= 1
                    # if (count % 100 == 0):
                    #     accuracy = float(top1)/count
                    #     print("Validation accuracy: {}".format(accuracy))
                    #     print("{} correct prediction out of {}".format(top1, count))
                imgs = []
        top1accuracy = float(top1)/count
        top5accuracy = float(top5)/count
        print("top1 {} {} {} {}".format(occlusionRatio, top1, count, top1accuracy))
        print("top5 {} {} {} {}".format(occlusionRatio, top5, count, top5accuracy))
        top1 = 0
        count = 0
        top5 = 0

