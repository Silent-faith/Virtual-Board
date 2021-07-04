# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 12:57:24 2021

@author: abhis
"""


import cv2 
import time 
import os 
import TrackHands as TH 
import numpy as np 

CurrentTime=0
PreviousTime =0

header_img = "Images"
header_img_list = os.listdir(header_img)
overlay_image =[]


for i in header_img_list:
    image = cv2.imread(f'{header_img}/{i}')
    overlay_image.append(image)

Capture = cv2.VideoCapture(0)
Capture.set(3,1280)
Capture.set(4,720)
Capture.set(cv2.CAP_PROP_FPS, 60)


BrushBoard = overlay_image[0]
DrawColor = (255,200,100)
Board = np.zeros((900,1280,3), np.uint8)
detector= TH.HandDetector(min_detection_confidence=.85)

x = 0
y = 0
xp, yp = 0,0
while True : 
    ret, frame = Capture.read()    
    frame = cv2.resize(frame, (1280, 900))
    #print(frame.shape)
    frame[0:125,0:1280] = BrushBoard

    
    landmark_list = detector.findPosition(frame)
    
    if(len(landmark_list)!=0):   # hand detected 
        
        x1, y1 =(landmark_list[8][1:]) #index fingure 
        x2, y2 = landmark_list[12][1:] #middle fingure
    
        my_fingers = detector.fingerStatus()
        
        if (my_fingers[1]and my_fingers[2]): # both middle and index fingure are straight 
            xp, yp = 0,0
            if (y1<125):
                if(200<x1<340):
                    BrushBoard = overlay_image[0] 
                    DrawColor = (255,0,0)
                elif (340<x1<500):
                    BrushBoard = overlay_image[1]
                    DrawColor = (47,225,245)
                elif (500<x1<640):
                    BrushBoard = overlay_image[2]
                    DrawColor = (197,47,245)
                elif (640<x1<780):
                    BrushBoard = overlay_image[3]
                    DrawColor = (53,245,47)
                elif (1100<x1<1280):
                    BrushBoard = overlay_image[4]
                    DrawColor = (0,0,0)
                    
            cv2.putText(frame, 'Brush Selecting', (900,680), fontFace=cv2.FONT_HERSHEY_COMPLEX, color= (0,255,255), thickness=2, fontScale=1)
            cv2.line(frame, (x1,y1), (x2,y2), color = DrawColor, thickness=3)

        elif (my_fingers[1] and not my_fingers[2]): #only the index fingure is straight 
                     
            cv2.putText(frame, "Writing Mode", (900,680), fontFace= cv2.FONT_HERSHEY_COMPLEX, color= (255,255,0), thickness=2, fontScale=1)
            cv2.circle(frame, (x1,y1),15, DrawColor, thickness=-1)

            if xp ==0 and yp ==0:
                xp = x1 
                yp = y1
            
            if DrawColor == (0,0,0):
                cv2.line(Board, (xp,yp),(x1,y1),color= DrawColor, thickness = 50)

            else:
                cv2.line(Board, (xp,yp),(x1,y1),color= DrawColor, thickness = 10)
            
            xp , yp = x1, y1
        else : 
            xp = 0 
            xp = 0 
        
    BoardGray = cv2.cvtColor(Board, cv2.COLOR_BGR2GRAY)
    _, BoardLines = cv2.threshold(BoardGray, 50, 255, cv2.THRESH_BINARY_INV)
    BoardLines = cv2.cvtColor(BoardLines, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, BoardLines)
    frame =cv2.bitwise_or(frame, Board)
    CurrentTime = time.time()
    fps = 1/(CurrentTime- PreviousTime)
    PreviousTime = CurrentTime
    
    frame = frame[:720, :1280]
    cv2.putText(frame, 'Client FPS:' + str(int(fps)), (10,670), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)


    cv2.imshow('Board', frame)
    cv2.waitKey(1)