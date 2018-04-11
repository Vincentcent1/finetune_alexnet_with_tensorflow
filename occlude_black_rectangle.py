import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import time
# first argument is the image path
# second argument is the bounding box xml file path
# third argument is the occlusion percentage


if __name__ == "__main__":
	boundBoxPath = sys.argv[1]
	imageFolder = sys.argv[2]
	occlusionPercentage = sys.argv[3]
	write = sys.argv[4]
	isBoundBox = sys.argv[5]
	filename = boundBoxPath.split('/')[-1].split('.')[0] #Get the filename 
	# imgPath = '/media/Vincent/Deep Learning/Images/testPanda/' + filename
	imgPath = imageFolder + filename + ".JPEG"
	print(sys.argv)
	print(imgPath, boundBoxPath, occlusionPercentage)
	if isBoundBox == "true":
		occludedImg = occlude(imgPath,boundBoxPath, occlusionPercentage)
	elif isBoundBox == "random":
		occludedImg = occludeOriginalRandom(imgPath, occlusionPercentage)
	elif isBoundBox == "false":
		occludedImg = occludeOriginalCenter(imgPath, occlusionPercentage)
	elif isBoundBox == "gaussiancenter":
		occludedImg = gaussianCenter(imgPath, occlusionPercentage)	
	elif isBoundBox == "gaussianrandom":
		occludedImg = gaussianRandom(imgPath, occlusionPercentage)
	else:
		print(isBoundBox + " is not a valid parameter for isBoundBox")

	if write == "w":
		# write the new image to a file
		cv2.imwrite(filename + '_Occluded.JPEG',occludedImg)
	elif write == "r":
		# display the result
		cv2.imshow('occluded',occludedImg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()	

def parseXML(boundBoxPath):
	'''
	parse bounding box from imagenet
	@params: path to the bounding box xml file 
	@return: lists of rectangular boundary in the format of (xmin,ymin,xmax,ymax)
	'''
	boundaries = []
	tree = ET.parse(boundBoxPath)
	root = tree.getroot()
	bndboxes = root.iter('bndbox') #location of the bounding box element in XML
	for bndbox in bndboxes:
		xmin = bndbox[0].text
		ymin = bndbox[1].text
		xmax = bndbox[2].text
		ymax = bndbox[3].text
		boundaries.append((xmin,ymin,xmax,ymax))
	
	# print('xmin:{}, ymin:{}, xmax:{}, ymax:{}'.format(xmin,ymin,xmax,ymax))
	return boundaries

def occlude(imgPath, boundBoxPath, occlusionPercentage):
	'''
	occlude the image in the path and write an updated image to the 
	@params: 
		imgPath: path to the image to be occluded
		boundBoxPath: path to the bounding box xml file
		occlusionPercentage: percentage of occlusion wanted, in float (e.g. 0.5)
	@return: Image object to be written of size [227,227,3]
	'''
	img = cv2.imread(imgPath) # Bottleneck
	boundaries = parseXML(boundBoxPath)
	for boundary in boundaries:
		xmin,ymin,xmax,ymax = tuple(map(int, boundary)) #Get the boundary
		# img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),1)
		# Get the percentage offset for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
		# We do 1 - to get the remaining % for the width
		# Calculate percentage of how much gap should be left on each side of the boundary
		offset = (1 - np.sqrt(float(occlusionPercentage)))/2 
		width = xmax - xmin
		xoffset = (offset)*width
		xmin = int(xmin + xoffset)
		xmax = int(xmax - xoffset)
		height = ymax - ymin
		yoffset = (offset)*height
		ymin = int(ymin + yoffset)
		ymax = int(ymax - yoffset)
		img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),-1)
	img = cv2.resize(img, (227,227))
	return img

def cropAndOccludeCenter(imgPath, boundBoxPath, occlusionPercentage):
	'''
	occlude the image in the path and write an updated image to the 
	@params: 
		imgPath: path to the image to be occluded
		boundBoxPath: path to the bounding box xml file
		occlusionPercentage: percentage of occlusion wanted, in float (e.g. 0.5)
	@return: Image object to be written of size [227,227,3]
	'''
	img = cv2.imread(imgPath) # Bottleneck
	boundaries = parseXML(boundBoxPath)
	xmin,ymin,xmax,ymax = tuple(map(int, boundaries[0])) #Get the boundary
	img = img[ymin:ymax, xmin:xmax]
	xmin,ymin,xmax,ymax = 0,0,img.shape[1],img.shape[0] #Get the image boundary
	# Get the percentage offset for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
	# We do 1 - to get the remaining % for the width
	# Calculate percentage of how much gap should be left on each side of the boundary (left-right,bottom-top)
	offset = (1 - np.sqrt(float(occlusionPercentage)))/2
	width = xmax - xmin
	xoffset = (offset)*width # real value of the xoffset on left and right of the black box (% remaining * width)
	xmin = int(xmin + xoffset)
	xmax = int(xmax - xoffset)
	height = ymax - ymin
	yoffset = (offset)*height #real value of the yoffset on bottom and top of the black box (% remaining * height)
	ymin = int(ymin + yoffset)
	ymax = int(ymax - yoffset)
	img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),-1) # Generate black box on the image
	img = cv2.resize(img, (227,227), cv2.INTER_AREA)
	return img

def occludeOriginalCenter(imgPath, occlusionPercentage):
	'''
	occlude the image in the path with a black box at the center and return the image.
	@params:
		imgPath:path to the image to be occluded
		occlusionPercentage: percentage of occlusion wanted, in float  
	@return: Image object to be written of size [227,227,3]
	'''
	img = cv2.imread(imgPath)
	shape = img.shape
	xmin,ymin,xmax,ymax = 0,0,shape[1],shape[0] #Get the image boundary
	# Get the percentage offset for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
	# We do 1 - to get the remaining % for the width
	# Calculate percentage of how much gap should be left on each side of the boundary (left-right,bottom-top)
	offset = (1 - np.sqrt(float(occlusionPercentage)))/2
	width = xmax - xmin
	xoffset = (offset)*width # real value of the xoffset on left and right of the black box (% remaining * width)
	xmin = int(xmin + xoffset)
	xmax = int(xmax - xoffset)
	height = ymax - ymin
	yoffset = (offset)*height #real value of the yoffset on bottom and top of the black box (% remaining * height)
	ymin = int(ymin + yoffset)
	ymax = int(ymax - yoffset)
	img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),-1) # Generate black box on the image
	return img

def occludeOriginalRandom(imgPath, occlusionPercentage):
	'''
	occlude the image in the path with a black box at random position and return the image.
	@params:
		imgPath:path to the image to be occluded
		occlusionPercentage: percentage of occlusion wanted, in float
	@return: Image object to be written of size [227,227,3]
	'''	
	img = cv2.imread(imgPath)
	shape = img.shape
	xmin,ymin,xmax,ymax = 0,0,shape[1],shape[0]
	# Get the percentage occluded for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
	occludedSides = np.sqrt(float(occlusionPercentage))
	# We do 1 - to get the remaining % for the width
	# Calculate percentage of how much gap should be left in total (e.g 0.6 width occlusion means we have 0.4 * width to play with)
	offset = 1 - occludedSides
	# Randomize for each axis between 0 and 1
	randomX = float(np.random.random(1))
	randomY = float(np.random.random(1))
	width = xmax - xmin
	# to get the starting x offset, we multiply randomX to the % gap remaining
	xoffset = randomX*offset*width
	xmin = int(xmin + xoffset) # Start of black box
	# to get the end of the box, we just add the length of width occluded (percentage occluded * width of the full image)
	xmax = int(xmin + occludedSides * width)
	height = ymax - ymin
	# to get the starting y offset, we multiply randomY to the % gap remaining	
	yoffset = randomY*offset*height
	ymin = int(ymin + yoffset) # Start of black box
	# to get the end of the box, we just add the length of height occluded (percentage occluded * width of the full image)
	ymax = int(ymin + occludedSides * height)
	img = cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,0),-1) #Generate black box on the image
	return img

def gaussianCenter(imgPath, occlusionPercentage):
	'''
	occlude the image in the path with gaussian noise at the center and return the image.
	@params:
		imgPath:path to the image to be occluded
		occlusionPercentage: percentage of occlusion wanted, in float
	@return: Image object to be written of size [227,227,3]
	'''		
	img = cv2.imread(imgPath)
	shape = img.shape
	xmin,ymin,xmax,ymax = 0,0,shape[1],shape[0] #Get the image boundary
	# Get the percentage offset for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
	# We do 1 - to get the remaining % for the width
	# Calculate percentage of how much gap should be left on each side of the boundary (left-right,bottom-top)
	offset = (1 - np.sqrt(float(occlusionPercentage)))/2
	width = xmax - xmin
	xoffset = (offset)*width # real value of the xoffset on left and right of the black box (% remaining * width)
	xmin = int(xmin + xoffset)
	xmax = int(xmax - xoffset)
	height = ymax - ymin
	yoffset = (offset)*height #real value of the yoffset on bottom and top of the black box (% remaining * height)
	ymin = int(ymin + yoffset)
	ymax = int(ymax - yoffset)
	# Adding gaussian noise to the defined occlusion boundary
	rectangle = img[ymin:ymax, xmin:xmax]
	noise = np.zeros(rectangle.shape, np.uint8)
	cv2.randn(noise,0,(100,100,100)) # Standard deviation of 100
	img[ymin:ymax, xmin:xmax] = cv2.add(rectangle,noise)
	return img	

def gaussianRandom(imgPath, occlusionPercentage):
	'''
	occlude the image in the path with gaussian noise at random position and return the image.
	@params:
		imgPath:path to the image to be occluded
		occlusionPercentage: percentage of occlusion wanted, in float
	@return: Image object to be written of size [227,227,3]
	'''		
	img = cv2.imread(imgPath)
	shape = img.shape
	xmin,ymin,xmax,ymax = 0,0,shape[1],shape[0]
	# Get the percentage occluded for the width and height (e.g 0.36 occlusion means 0.6 * width and 0.6 * height is occluded)
	occludedSides = np.sqrt(float(occlusionPercentage))
	# We do 1 - to get the remaining % for the width
	# Calculate percentage of how much gap should be left in total (e.g 0.6 width occlusion means we have 0.4 * width to play with)
	offset = 1 - occludedSides
	# Randomize for each axis between 0 and 1
	randomX = float(np.random.random(1))
	randomY = float(np.random.random(1))
	width = xmax - xmin
	# to get the starting x offset, we multiply randomX to the % gap remaining
	xoffset = randomX*offset*width
	xmin = int(xmin + xoffset) # Start of black box
	# to get the end of the box, we just add the length of width occluded (percentage occluded * width of the full image)
	xmax = int(xmin + occludedSides * width)
	height = ymax - ymin
	# to get the starting y offset, we multiply randomY to the % gap remaining	
	yoffset = randomY*offset*height
	ymin = int(ymin + yoffset) # Start of black box
	# to get the end of the box, we just add the length of height occluded (percentage occluded * width of the full image)
	ymax = int(ymin + occludedSides * height)
	# Adding gaussian noise to the defined occlusion boundary	
	rectangle = img[ymin:ymax, xmin:xmax]
	noise = np.zeros(rectangle.shape, np.uint8)
	cv2.randn(noise,0,(100,100,100)) # Standard deviation of 100
	img[ymin:ymax, xmin:xmax] = cv2.add(rectangle,noise)
	return img	




