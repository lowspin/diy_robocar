#!/usr/bin/env python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imagefunctions 

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	###############################################
	# Presentation Overlays before transform
	###############################################
#	points0 = np.array([[0,290],[200,100],[440,100],[640,290]])
#	cv2.polylines(image, [points0],True,(0,255,255),2)
	
	###############################################
	# Extract pixels for detection
	###############################################
	img_bin = imagefunctions.extractpixels(image)
	###############################################

	###############################################
	# Perspective Transform
	###############################################
#	M, Minv, warpedimg, warpedbin = imagefunctions.perspectiveTransform(image,img_bin)
	###############################################

	###############################################
	# Presentation Overlays 
	###############################################
#	out_img = np.dstack((img_bin,img_bin,img_bin))*255
	#cv2.rectangle(out_img, (10,20),(110,240),(0,255,0),2)
#	points = np.array([[8,220],[200,30],[410,30],[640,220]])
#	cv2.polylines(out_img, [points]	,True,(0,255,0),2)
#	cv2.putText(out_img,'test test', org=(100,300), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1., color=(0,0,255))
	###############################################
	#out_img = np.dstack((warpedbin,warpedbin,warpedbin))*255
#	out_img = warpedimg
#	points = np.array([[200,220],[200,0],[410,0],[410,220]])
#	cv2.polylines(out_img, [points]	,True,(0,255,0),2)
	###############################################

#	rectx1 = 300
#	recty1 = 280
#	rectx2 = 340
#	recty2 = 320
#	points = np.array([[rectx1,recty1],[rectx2,recty1],[rectx2,recty2],[rectx1,recty2]])
#	out_img = np.dstack((img_bin,img_bin,img_bin))*255
#	rectwhite = imagefunctions.rect_is_all_white(img_bin,rectx1,recty1,rectx2,recty2)
#	if rectwhite is True:
#		cv2.polylines(out_img, [points]	,True,(0,255,0),2)
#	else:
#		cv2.polylines(out_img, [points]	,True,(255,0,0),2)
	
	ss, xshift = imagefunctions.find_white_patch(img_bin)
	nrows=480 #img_bin.shape[0] #480
	ncols=640 #img_bin.shape[1] #640
	rectx1 = int(ncols/2-ss/2+xshift)
	recty1 = nrows-ss
	rectx2 = int(ncols/2+ss/2+xshift)
	recty2 = nrows
	points = np.array([[rectx1,recty1],[rectx2,recty1],[rectx2,recty2],[rectx1,recty2]])
	out_img = np.dstack((img_bin,img_bin,img_bin))*255
	cv2.polylines(out_img, [points]	,True,(0,255,0),2)
			
	# show the frame
	cv2.imshow("Frame", out_img) #out_img) #img_bin)
	key = cv2.waitKey(1) & 0xFF	

	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


