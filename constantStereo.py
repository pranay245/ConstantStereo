# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:25:51 2022

@author: Pranay
"""

import cv2
import pygame as pg
pg.init()
sound0 = pg.mixer.Sound('audio.wav')
channel0 = pg.mixer.Channel(0)




cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 250)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)
channel0.play(sound0)

while True:
    
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,1.3,5)

    if(len(detections) > 0):
       (x,y,w,h) = detections[0]
       frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
       lv=0
       rv=0
       if x>125:
           lv=0.5+(((x-125)/125)*0.5)
           rv=0.5
       elif x<125:
           lv=0.5
           rv=0.5+(((125-x)/125)*0.5)
       elif x==125:
           lv=0.5
           rv=0.5
           
 
       channel0.set_volume(rv, lv)
       print(x,y,w,h)


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
