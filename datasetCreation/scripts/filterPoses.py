#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 22:10:07 2020

@author: svosburg
"""

import os
import json
import cv2
import numpy as np

DP10_OPENPOSE_RESULTS_FOLDER = "../openposeResults/"
DP10_TRIMMED_FOLDER = '../Clips/Trimmed/'


if __name__ == '__main__':
    testName = "PHYT/_44.mp4"
    testOpenpose = DP10_OPENPOSE_RESULTS_FOLDER + 'PHYT/_44/'
    testTrimmed = DP10_TRIMMED_FOLDER + testName
    
    
    #Open playerInvolved.JSON file
    with open(DP10_TRIMMED_FOLDER + "playersInvolved.json", 'r') as f:
        clipInfo = json.load(f)

    cap = cv2.VideoCapture(testTrimmed)
    descriptor = cv2.ORB_create()
    kps_save, features_save = None, None
    p1,p2 = None, None
    homographyMatrixList = []
    terminate = False
    frameNum = -1
    frameSave = None
    while (cap.isOpened()):
        ret, frame = cap.read()
        frameNum += 1
        
        if(frame is None):
            terminate = True
            break
        
        #~~~~~~~~~~~~~~~~~~~~
        #Processing on frames
        #~~~~~~~~~~~~~~~~~~~~
        
        #1. Track using SIFT keypoints
        (kps, features) = descriptor.detectAndCompute(frame, None)
        
        # check if not first frame
        if(kps_save is not None and features_save is not None):
            
            #2. Feature Matching
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(features,features_save)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            matches = matches[:10]
                    
            #3. Find Homography Matrix
            src_pts = np.float32([ kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kps_save[m.trainIdx].pt for m in matches  ]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            
            #normalize H
            
            
            homographyMatrixList.append(H)
            
            #Transform Player Points
            if(p1 is not None):
                if(frameSave is not None):
                    cv2.imshow('test',cv2.drawMatches(frame,kps,frameSave,kps_save,matches[:100],frame,flags=2))
                p1save = p1
                p1 = np.array([p1[0],p1[1],1])
                p1 = p1[np.newaxis]
                print(p1.shape, H)
                #p1 = cv2.perspectiveTransform(np.asarray([a]),H)
                p1 = np.dot(H,np.transpose(p1))
            if(p2 is not None):
                pass
                #p2 = cv2.perspectiveTransform([[p2]],H)
                
        #4. Fetch Player Position if we dont have it
        if(p1 is None):
            if(clipInfo[testName]['p1']['frameNum'] == frameNum):
                p1 = np.float32(clipInfo[testName]['p1']['pos'])
        if(p2 is None):
            if(clipInfo[testName]['p2']['frameNum'] == frameNum):
                p2 = np.float32(clipInfo[testName]['p2']['pos'])
                
        #5. Draw players on Screen
        if(p1 is not None):
            frameSave = frame
            frame = cv2.circle(frame, (p1[0],p1[1]), 5, color = (255, 0, 0), thickness = 10)
        if(p2 is not None):
            frame = cv2.circle(frame, (p2[0],p2[1]), 5, color = (255, 255, 0), thickness = 10)
                
        
        
        #=============
        #Display Image
        #=============
        cv2.imshow('frame',frame)
        
        #================
        #Await User Input
        #================
        while(True):
            k = cv2.waitKey(20) & 0xFF
            if(k == ord(' ')):
                break
            elif(k == ord('q')):
                terminate = True
                break
        
        #=========================
        # Save kps and features
        #=========================
        kps_save, features_save = kps, features
        
        #========================================
        #Check if needing Termination of Program
        #========================================
        if(terminate or not ret):
            break

    #Terminate program peacefully
    cv2.destroyAllWindows()
     
