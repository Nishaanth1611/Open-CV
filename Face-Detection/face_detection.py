import cv2
from random import randrange

trained_data = cv2.CascadeClassifier("data.xml")
"""
You can use the image for detection
img = cv2.imread('image.png')

You can also use the video for detection
webcam=cv2.VideoCapture("video.mp4")
"""

#Here we are using our webcam
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read,frame= webcam.read()
    
    grey_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates = trained_data.detectMultiScale(grey_img)
    
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)
    
    cv2.imshow('Camera Face Detector',frame)
    
    key=cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()

print("Code Completed")