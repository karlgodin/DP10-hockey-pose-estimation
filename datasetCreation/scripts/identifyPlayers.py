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
import os
from os import listdir, walk
from os.path import isfile, join

#Declare globals
DP10_SCALE = 1.0
DP10_SIZE = (640,360)
DP10_USER_FSM = 0 # 0 = SelectP1, 1 = SelectP2, 3 = Watch rest of vid with only the two players
DP10_FONT = cv2.FONT_HERSHEY_SIMPLEX
DP10_CLIP_PATH = "../Clips/"
DP10_TRIM_PATH = "../Clips/Trimmed/"
DP10_JSON_PATH = "../json/"

#Create callback function
def clbk_mouseclick(event,x,y,flags,param):
    global mouseX,mouseY,DP10_USER_FSM, clipObj,DP10_SCALE,frameNum
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y
        print("frameNum,x,y,scale=%d,%d,%d,%f"%(frameNum,x,y,DP10_SCALE))
        if(DP10_USER_FSM == 0):
            clipObj.startFrame = frameNum
        elif(DP10_USER_FSM == 1):
            clipObj.addP1(frameNum,x/DP10_SCALE,y/DP10_SCALE)
        elif(DP10_USER_FSM == 2):
            clipObj.addP2(frameNum,x/DP10_SCALE,y/DP10_SCALE)
        elif(DP10_USER_FSM == 3):
            clipObj.endFrame = frameNum
        DP10_USER_FSM+=1
  
#Define serializable class that will be used to save position of players involved in penalty      
class Clip():
    def __init__(self):
        self.p1 = {'frameNum':0,'pos':[0,0]}
        self.p2 = {'frameNum':0,'pos':[0,0]}
        self.startFrame = None
        self.endFrame = None
    
    def addP1(self,frameNum,x,y):
        self.p1['frameNum'] = frameNum
        self.p1['pos'] = [x,y]
        
    def addP2(self,frameNum,x,y):
        self.p2['frameNum'] = frameNum
        self.p2['pos'] = [x,y]
        
    def clear(self):
        self.p1 = {'frameNum':0,'pos':[0,0]}
        self.p2 = {'frameNum':0,'pos':[0,0]}
        self.startFrame = None
        self.endFrame = None
        
if __name__ == "__main__":
    #Set callback
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',clbk_mouseclick)
    
    #Create output json
    outpath = DP10_CLIP_PATH + "playersInvolved.json"
    outpathTrimmed = DP10_TRIM_PATH + "playersInvolved.json"
    
    #Create out dictionary
    if(isfile(outpath)):
        with open(outpath,'r') as f:
            out = json.load(f)
    else:
        out = {}
    
    #Control Vars
    terminate = False
    clipObj = None
    
    #Infinite loop. Ask User to click on p1 (perp) & p2 (victim) 
    clipNames = []
    for subdir,dirs,files in walk(DP10_CLIP_PATH):
        for f in files:
            filename = subdir+"/"+f
            if (isfile(filename) and ".mp4" in filename and "Trimmed" not in filename):
                clipNames.append(filename)
                
    #clipNames = [f for f in listdir(DP10_CLIP_PATH) if (isfile(join(DP10_CLIP_PATH, f)) and ".mp4" in f)]
    clipNameIdx = 0
    speed = 20
    while clipNameIdx <len(clipNames):
        clipName = clipNames[clipNameIdx]
        
        #check to see if file has already been labelled
        filename = "/".join(clipName.split('/')[2:])
        if filename in out:
           clipNameIdx += 1
           continue
        
        cap = cv2.VideoCapture(DP10_CLIP_PATH + clipName)
        clipObj = Clip()
        frameNum = 0;
        DP10_USER_FSM = 0
        frameSave = None
        lastFrame = False
        paused = False
	restart = False
        print "/".join(clipName.split('/')[2:])
        while(cap.isOpened()):
            jumpNumOfFrames = 0            
            #Fetch frame
            ret, frame = cap.read()
            if(frame is not None):
                frameSave = frame
            
            if not ret:
                if(DP10_USER_FSM < 4):
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
                # Seleting the first Frame
                cv2.putText(frame, 'Select Start Frame', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 1):
                # Seleting the perpetrator
                cv2.putText(frame, 'SelectP1 (Perp)', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 2):
                # Selecting the victim
                cv2.putText(frame, 'SelectP2 (Victim)', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 3):
                # Seleting the first Frame
                cv2.putText(frame, 'Select End Frame', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            elif(DP10_USER_FSM == 4):
                cv2.putText(frame, 'Validate', (0,30), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                DP10_USER_FSM = 4
            
            #Display frameNum
            cv2.putText(frame, 'Frame#:%d'%frameNum, (0,60), DP10_FONT, 1, (0, 255, 0), 1, cv2.LINE_AA)
            
            #Show image to user    
            cv2.imshow('frame',frame)
            
            #Get keyboard inputq
            k = cv2.waitKey(speed) & 0xFF
            if k != 255:
                print k
            if k == 27:
                break
            elif k == 52: #4 (left arrow) on number pad Back up
                if(frameNum-speed > 0):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum-speed)
                    frameNum -= speed
                    print "Jumped to frame %d"%(frameNum)
            elif k == 54: #6 (right arrow) on number pad fast forward
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum+speed)
                frameNum += speed
                print "Jumped to frame %d"%(frameNum)
            elif k == 56: #8 (up arrow) on number pad Speed up
                speed -= 5
                if(speed < 0):
                    speed = 1
                print 'speed: %d'%speed
            elif k == 50: #8 (down arrow) on number pad slow down
                speed += 5
                print 'speed: %d'%speed
            elif k == ord('a'): #Print mouse position
                print mouseX,mouseY
            elif k == ord('n'): #Quit current video. go to Next video
                 break
            elif k == ord('r'): #restart this video
                clipNameIdx -= 1
                clipObj.clear()
                restart = True
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
        if(not (terminate and DP10_USER_FSM == 0) and not restart):
            filename = "/".join(clipName.split('/')[2:])
            out[filename] = clipObj.__dict__
        clipNameIdx += 1
        if(terminate):
            break
        
    #check to see if any object with no labelling needs to be deleted
    filenamesToDelete = []
    for filename in out:
        startFrame = out[filename]['startFrame']
        endFrame = out[filename]['endFrame']
        if(startFrame is None or endFrame is None):
            filenamesToDelete.append(filename)
            continue
    for filename in filenamesToDelete:
	print("Deleting %s"%filename)
        del out[filename]
    
    #Save to file
    with open(outpath,'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
        
    #Create Trimmed videos
    if(isfile(outpathTrimmed)):
        with open(outpathTrimmed,'r') as f:
            outTrimmed = json.load(f)
    else:
        outTrimmed = {}
    outTrimmed.update(out)
    
    #Trim videos and update the out file
    print "Saving Trimmed Clips..."
    filenamesToDelete = []
    for filename in outTrimmed:
        startFrame = outTrimmed[filename]['startFrame']
        endFrame = outTrimmed[filename]['endFrame']
        
        if(startFrame is None or endFrame is None):
            filenamesToDelete.append(filename)
            continue
        
        #update element in dictionary
        outTrimmed[filename]['startFrame'] = 0
        outTrimmed[filename]['endFrame'] = endFrame - startFrame
        outTrimmed[filename]['p1']['frameNum'] -= startFrame
        outTrimmed[filename]['p2']['frameNum'] -= startFrame
        
        #trim video and save to file
        #Load Video
        cap = cv2.VideoCapture(DP10_CLIP_PATH + filename)
    
        #Get FrameRate and size
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        
        #Create output
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if(not os.path.exists(os.path.dirname(DP10_TRIM_PATH + filename))):
            os.makedirs(os.path.dirname(DP10_TRIM_PATH + filename))
        out_trimmedVid = cv2.VideoWriter(DP10_TRIM_PATH + filename,fourcc, fps, (width,height))
        
        #set frame 
        cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
        
        #Copy over frame from cap to out
        for _ in range(endFrame - startFrame):
            ret, frame = cap.read()
            if not ret:
                break
            out_trimmedVid.write(frame)
        
        #Release videos
        cap.release()
        out_trimmedVid.release()
        
    for filename in filenamesToDelete:
	print("Deleting entry for: %s"%filename)
        del outTrimmed[filename]
    #create outTrimmed.json
    with open(outpathTrimmed,'w') as f:
        json.dump(outTrimmed, f, ensure_ascii=False, indent=4)
    
    #Terminate program peacefully
    cv2.destroyAllWindows()
    
    
    
