#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:50:33 2020

@author: svosburg
For Design Project Team10
"""

import cv2
import numpy as np
import json
from os import listdir
from os.path import isfile, join

#Declare globals
DP10_SCALE = 1.0
DP10_SIZE = (640,360)
DP10_USER_FSM = 0 # 0 = SelectP1, 1 = SelectP2, 3 = Watch rest of vid with only the two players
DP10_FONT = cv2.FONT_HERSHEY_SIMPLEX
DP10_CLIP_PATH = "../Clips/"
DP10_JSON_PATH = "../json/"

#Create callback function
def clbk_mouseclick(event,x,y,flags,param):
    global mouseX,mouseY,DP10_USER_FSM, clipObj,DP10_SCALE,frameNum
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        print("frameNum,x,y,scale=%d,%d,%d,%f"%(frameNum,x,y,DP10_SCALE))
        if(DP10_USER_FSM == 0):
            clipObj.addP1(frameNum,x/DP10_SCALE,y/DP10_SCALE)
        elif(DP10_USER_FSM == 1):
            clipObj.addP2(frameNum,x/DP10_SCALE,y/DP10_SCALE)
        DP10_USER_FSM+=1
  
#Define serializable class that will be used to save position of players involved in penalty      
class Clip():
    def __init__(self):
        self.p1 = {'frameNum':0,'pos':[0,0]}
        self.p2 = {'frameNum':0,'pos':[0,0]}
    
    def addP1(self,frameNum,x,y):
        self.p1['frameNum'] = frameNum
        self.p1['pos'] = [x,y]
        
    def addP2(self,frameNum,x,y):
        self.p2['frameNum'] = frameNum
        self.p2['pos'] = [x,y]
        
    def clear(self):
        self.p1 = {'frameNum':0,'pos':[0,0]}
        self.p2 = {'frameNum':0,'pos':[0,0]}
        
if __name__ == "__main__":
    #Set callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',clbk_mouseclick)
    
    #Create output json
    outpath = DP10_CLIP_PATH + "playersInvolved.json"
    out = {}
    
    #Control Vars
    terminate = False
    clipObj = None
    
    #Infinite loop. Ask User to click on p1 (perp) & p2 (victim)    
    clipNames = onlyfiles = [f for f in listdir(DP10_CLIP_PATH) if (isfile(join(DP10_CLIP_PATH, f)) and ".mp4" in f)]
    clipNameIdx = 0
    while clipNameIdx <len(clipNames):
        clipName = clipNames[clipNameIdx]
        cap = cv2.VideoCapture(DP10_CLIP_PATH + clipName)
        clipObj = Clip()
        frameNum = 0;
        DP10_USER_FSM = 0
        frameSave = None
        lastFrame = False
        paused = False
        while(cap.isOpened()):
            #Fetch frame
            ret, frame = cap.read()
            if(frame is not None):
                frameSave = frame
            
            if not ret:
                if(DP10_USER_FSM < 2):
                    frame = frameSave
                    lastFrame = True
                else:
                    break
            
            #Resize frame by DP10_SCALE for speed purposes
            #Keep aspect Ratio constant
            DP10_SCALE = DP10_SIZE[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
            frame = cv2.resize(frame,(0,0),fx=DP10_SCALE,fy=DP10_SCALE)
            
            #Print Task to user
            if(DP10_USER_FSM == 0):
                # Seleting the perpetrator
                cv2.putText(frame, 'SelectP1 (Perp)', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 1):
                # Selecting the victim
                cv2.putText(frame, 'SelectP2 (Victim)', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 2):
                cv2.putText(frame, 'Validate', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                DP10_USER_FSM = 2
            
            #Display frameNum
            cv2.putText(frame, 'Frame#:%d'%frameNum, (0,60), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
            #Show image to user    
            cv2.imshow('frame',frame)
            
            #Get keyboard input
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'): #Print mouse position
                print mouseX,mouseY
            elif k == ord('n'): #Quit current video. go to Next video
                 break
            elif k == ord('r'): #restart this video
                clipNameIdx -= 1
                clipObj.clear()
                break
            elif k == ord(' '): #Pause video
                paused = not paused
            elif k == ord('q'): #terminate program
                cap.release()
                cv2.destroyAllWindows()
                terminate = True
            
            #control statments
            if(paused):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum-1)
                continue
            if(terminate):
                break
            
            #Update Iters
            if(not lastFrame):
                frameNum += 1
        
        #Close peacefully
        cap.release()
        if(not (terminate and DP10_USER_FSM == 0)):
            out[clipName] = clipObj.__dict__
        clipNameIdx += 1
        if(terminate):
            break
       
    #Save to file
    with open(outpath,'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    
    #Terminate program peacefully
    cv2.destroyAllWindows()
    
    
    
