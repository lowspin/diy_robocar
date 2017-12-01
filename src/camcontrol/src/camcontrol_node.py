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
	# Extract pixels for detection
	###############################################
	img_bin = imagefunctions.extractpixels(image)
	###############################################

	###############################################
	# Presentation Overlays 
	###############################################
	out_img = np.dstack((img_bin,img_bin,img_bin))*255
	#cv2.rectangle(out_img, (10,20),(110,240),(0,255,0),2)
	points = np.array([[8,220],[200,30],[410,30],[640,220]])
	cv2.polylines(out_img, [points]	,True,(0,255,0),2)
	cv2.putText(out_img,'test test', org=(100,300), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1., color=(0,0,255))
	###############################################
	
	# show the frame
	cv2.imshow("Frame", out_img) #img_bin)
	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


