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
import copy

DP10_OPENPOSE_RESULTS_FOLDER = "../openposeResults/"
DP10_TRIMMED_FOLDER = '../Clips/Trimmed/'
DP10_CLIPS_FOLDER = '../Clips/'
DP10_POSE_THRESHOLD_COEFF = 2
DP10_INTERPOLATION = 'linear'
distance_between_players = []
notInvolvedPlayers = []
notInvolvedPlayersToAdd = []
playersToDisplayDebug = []
DP10_CLOSEST_JOINTS = [ #Body 25
[17,18,15,16,1,2,5,3,6,8,4,7,9,12,10,13,11,14,24,21,23,19,22,20],#0
[2,5,0,17,18,15,16,3,6,8,4,7,9,12,10,13,11,14,24,21,23,19,22,20],#1
[1,3,5,4,0,6,9,8,17,15,16,18,7,12,10,13,11,14,24,21,23,19,22,20],#2
[2,4,9,1,8,0,10,17,15,16,18,5,6,7,12,13,11,14,24,21,23,19,22,20],#3
[3,2,9,1,8,0,10,17,15,16,18,5,6,7,12,13,11,14,24,21,23,19,22,20],#4
[1,6,2,0,18,7,12,8,16,15,17,3,4,9,13,10,14,11,21,24,20,23,19,22],#5
[5,7,12,8,9,1,2,3,4,0,18,16,15,17,13,10,14,11,21,24,20,23,19,22],#6
[6,12,8,5,9,1,4,3,2,13,10,14,11,21,24,20,23,19,22,0,18,16,15,17],#7
[12,9,7,4,13,10,6,3,5,1,2,14,11,21,24,20,23,22,19,0,18,17,16,15],#8
[8,10,12,1,4,3,2,7,6,5,13,11,14,24,21,23,19,22,20,0,18,17,16,15],#9
[9,11,24,23,22,8,12,13,14,21,19,20,4,7,3,6,2,5,1,0,17,18,15,16],#10
[24,23,22,10,20,19,21,14,13,9,12,8,1,4,7,3,6,2,5,0,17,18,15,16],#11
[24,23,22,10,20,19,21,14,13,9,12,8,1,4,7,3,6,2,5,0,17,18,15,16],#12
[14,21,10,20,12,11,24,22,23,12,8,9,1,7,6,5,4,3,2,0,18,16,15,17],#13
[21,19,20,13,23,22,24,11,10,12,9,8,7,6,5,4,3,2,1,0,18,16,15,17],#14
[17,0,16,18,1,2,5,3,6,4,7,9,8,12,10,13,11,14,24,21,23,19,22,20],#15
[18,0,15,17,1,2,5,3,6,4,7,9,8,12,10,13,11,14,24,21,23,19,22,20],#16
[15,0,16,18,1,2,5,3,6,4,7,9,8,12,10,13,11,14,24,21,23,19,22,20],#17
[16,0,15,17,1,2,5,3,6,4,7,9,8,12,10,13,11,14,24,21,23,19,22,20],#18
[20,21,14,13,12,23,22,11,24,10,9,8,1,7,4,6,3,5,2,0,18,16,15,17],#19
[19,21,14,13,12,23,22,11,24,10,9,8,1,7,4,6,3,5,2,0,18,16,15,17],#20
[14,19,20,13,12,23,22,11,24,10,9,8,1,7,4,6,3,5,2,0,18,16,15,17],#21
[23,11,24,10,9,19,20,21,14,13,12,8,1,7,4,6,3,5,2,0,18,16,15,17],#22
[22,11,24,10,9,19,20,21,14,13,12,8,1,7,4,6,3,5,2,0,18,16,15,17],#23
[11,22,23,10,9,19,20,21,14,13,12,8,1,7,4,6,3,5,2,0,18,16,15,17],#24
]

#=================
#HELPER FUNCTIONS
#=================
class FrameSave():
    def __init__(self,frame,frameNum,p1,pose1,p2,pose2,notInvolvedPlayers,pdd):
        self.frame = frame
        self.frameNum = frameNum
        self.p1 = p1
        self.pose1 = pose1
        self.p2 = p2
        self.pose2 = pose2
        self.nip = notInvolvedPlayers
        self.pdd = pdd
    
def getThreshold():
    return DP10_POSE_THRESHOLD_COEFF* max(distance_between_players[-5:])
 
def weightGenerator(n,firstPnt, secondPnt):
    if(DP10_INTERPOLATION is 'linear' or DP10_INTERPOLATION is 'polynomial'):
        s = n+1
        for i in range(1,s):
            yield (1 - i/s,i/s)
            
    elif(DP10_INTERPOLATION is 'piecewise'):
        for _ in range(n//2):
            yield(1,0)
        for _ in range(n - n//2):
            yield(0,1)
            
    elif (DP10_INTERPOLATION is 'square'):
        s = n+1
        for i in range(1,s):
            val = (i/s)**2
            yield (1 - (val),val)
    elif(DP10_INTERPOLATION is 'exp'):
        s = n+1
        for i in range(1,s):
            val = np.exp(-i)
            yield(val,1-val)
    elif(DP10_INTERPOLATION is 'sinusodial'):
        s = n+1
        for i in range(1,s):
            val = np.cos(0.5*np.pi*i/s)
            yield(val,1-val)
    elif(DP10_INTERPOLATION is 'pascal2'):
        s = n+1
        for i in range(1,s):
            val = 1 - 2*i/s + (i/s)**2
            yield(val,1-val)
    elif(DP10_INTERPOLATION is 'specialSpice'):
        s = n+1
        for i in range(1,s):
            val = 1 - 4*i/s + 6*(i/s)**2 - 4*(i/s)**3 + (i/s)**4
            yield(val,1-val)
    elif(DP10_INTERPOLATION is 'splice'):
        #Need three data points to fit splice
        pass
    else:
        while True:
            yield (0,1)
 
def findPoseElucianDistance(pose1,pose2):
    """
    Sums the distance between each joint
    
    Parameters
    ----------
    pose1: First pose from openPose
    pose2: Second pose from openPose
    
    Returns
    -------
    Float corresponding to the sum of distances between all joints
    """
    pass

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def displayPlayer(frame, pose,color = (255,0,0)):
    """
    Print pose on player
    
    Parameters
    ----------
    frame: image to print player on
    pose: array of joint locations
    color: tuple of (B,G,R)
    
    Returns
    -------
    void
    """
    halfColor = (color[0] // 2,color[1] // 2,color[2] // 2)
    for (x,y,c) in chunker(pose,3):
        if(c is 0):
            tempColor = halfColor
        else:
            tempColor = color
        frame = cv2.circle(frame, (int(x),int(y)), 1, color = tempColor, thickness = 10)
        
    return frame
    
def findClosestPoseFromPoint(p, poseArr):
    """
    Returns the pose of the closest player
    
    Parameters
    ----------
    p = (x,y) that user clicked on screen.
    poseArr = Array of player pose estimation
    
    Returns
    -------
    Has the pose Changed?, Array of pose joint location,poseArr
    """
    poseSave = None
    closestDistance = None
    idx = None
    for i,poseEstimation in enumerate(poseArr):
        for (x,y,c) in chunker(poseEstimation['pose_keypoints_2d'],3):
            """
                A Pose is formated like [x1,y1,c1,x2,y2,c2,...] 
                    where x is Xpos, y is Ypos and C is confidence score
                
                A confidence score of 0 means there is no estimation available for that joint                
            """
            
            if(c is not 0):
                if(closestDistance is None):
                    #The first joint is inherently the closest one
                    poseSave = poseEstimation
                    closestDistance = np.linalg.norm( [ p[0] - x, p[1] - y] )
                    idx = i
                else:
                    #Compare with saved closestDistance and update members if closest
                    tempDistance = np.linalg.norm( [ p[0] - x, p[1] - y] )
                    
                    if(tempDistance < closestDistance):
                        #Closer, save distance and pose
                        poseSave = poseEstimation
                        closestDistance = tempDistance
                        idx = i
                        
    
    if(poseSave is None):
        return False,poseSave, poseArr
    del poseArr[idx]
    return True,poseSave['pose_keypoints_2d'],poseArr    
 
def findClosestPoseFromPose(pose,distance, poseArr):
    """
    Returns the pose of the closest player
    
    Parameters
    ----------
    pose = pose of player 
    distance = threshold distance
    poseArr = Array of player pose estimation
    
    Returns
    -------
    Has the pose Changed?, Array of pose joint location,poseArr
    """
    poseSave = None
    idx = None
    errSave = None
    for i,poseEstimation in enumerate(poseArr):
        
        numOfValidJoint = 0
        err = 0
        for ((x1,y1,c1),(x2,y2,c2)) in zip(chunker(poseEstimation['pose_keypoints_2d'],3),chunker(pose,3)):
            """
                A Pose is formated like [x1,y1,c1,x2,y2,c2,...] 
                    where x is Xpos, y is Ypos and C is confidence score
                
                A confidence score of 0 means there is no estimation available for that joint                
            """
            
            if(c1 is not 0 and x2 is not 0 and y2 is not 0):
                dist = np.linalg.norm([x1-x2,y1-y2])
                
                
                numOfValidJoint += 1
                err += dist
                
        #Find Error per joint
        if(numOfValidJoint == 0 ):
            return False,pose, poseArr
        err /= numOfValidJoint
        
        #Compared with save values
        if(errSave is None):
            poseSave = poseEstimation
            errSave = err
            idx = i
        else:
            if(err < errSave):
                poseSave = poseEstimation
                errSave = err 
                idx = i
                
    if(errSave is None):
        return False,pose, poseArr
    if(errSave > getThreshold()):
        return False,pose, poseArr
    
    #Check if closest pose in under threshold. If above, replace with previous pose
    #Pose will be replaced.
    for i,((x1,y1,c1),(x2,y2,c2)) in enumerate(zip(chunker(pose,3),chunker(poseSave['pose_keypoints_2d'],3))):
        if(c2 is not 0):
            pose[3*i] = x2
            pose[3*i + 1] = y2
            pose[3*i + 2] = c2
        else:
            pose[3*i + 2] = c2
    del poseArr[idx]
    return True,pose,poseArr
    
    
def distancePoses(pose1,pose2):
    """
    Finds distance between poses
    
    Parameters
    ----------
    pose1: Pose of the first player
    pose2: pose of the second player
    
    Returns
    -------
    Average norm between the joints
    """

    err = 0
    numOfValidJoint = 0
    for ((x1,y1,c1),(x2,y2,c2)) in zip(chunker(pose1,3),chunker(pose2,3)):
        """
            A Pose is formated like [x1,y1,c1,x2,y2,c2,...] 
                where x is Xpos, y is Ypos and C is confidence score
            
            A confidence score of 0 means there is no estimation available for that joint                
        """
        
        if(c1 is not 0 and c2 is not 0):
            numOfValidJoint += 1
            err += np.linalg.norm([x1-x2,y1-y2])
            
    #Find Error per joint
    if(numOfValidJoint == 0):
        return np.Inf
    err /= numOfValidJoint
    return err
  

if __name__ == '__main__':
    for dir in os.listdir(DP10_OPENPOSE_RESULTS_FOLDER):
        for d in os.listdir(DP10_OPENPOSE_RESULTS_FOLDER + dir):
            testName = "%s/%s.mp4"%(dir,d)
            #testName = 'PHYT/_44.mp4'
            clipName = testName.split('.')[0].split('/')[1]
            testOpenpose = DP10_OPENPOSE_RESULTS_FOLDER + testName.split('.')[0] + '/output_jsn/'
            testTrimmed = DP10_TRIMMED_FOLDER + testName
            
            #Open playerInvolved.JSON file
            with open(DP10_TRIMMED_FOLDER + "playersInvolved.json", 'r') as f:
                clipInfo = json.load(f)

            cap = cv2.VideoCapture(testTrimmed)
            clipSave = []
            p1,p2 = None, None
            p1Changed,p2Changed = None, None
            pose1, pose2 = None, None
            openposeKeypoint = None
            terminate = False
            frameNum = -1
            frameSave = None
            nextVid = False
            while (cap.isOpened()):
                ret, frame = cap.read()
                frameNum += 1
                
                if(frame is None):
                    break
                
                #~~~~~~~~~~~~~~~~~~~~
                #Processing on frames
                #~~~~~~~~~~~~~~~~~~~~
                #1. Fetch openpose jsn file
                keypointFileName = clipName + "_" + str(frameNum).zfill(12) + '_keypoints.json'
                with open(testOpenpose + keypointFileName, 'r') as f:
                    openposeKeypoint = json.load(f)
                
                #Pre 2. Remove Non-Involved Players
                if(p1 is None and p2 is None):
                    notInvolvedPlayers = []
                    notInvolvedPlayersToAdd = []
                temp1 = len(openposeKeypoint['people'])
                indexToAdd = []
                for niPlayer in notInvolvedPlayers:
                    distList = [distancePoses(pose['pose_keypoints_2d'], niPlayer['pose_keypoints_2d']) for pose in openposeKeypoint['people']]
                    if(len(distList) > 0):
                        dist = min(distList)
                        i = distList.index(dist)
                        if(dist < 50):
                            indexToAdd.append(i) 
                        else:
                            playersToDisplayDebug.append(niPlayer['pose_keypoints_2d'])
                            notInvolvedPlayersToAdd.append(niPlayer)
                            
                #Get unique number
                indexToAdd = sorted(list(set(indexToAdd)), reverse=True)
                for idx in indexToAdd:
                    notInvolvedPlayersToAdd.append(openposeKeypoint['people'][idx])
                    del openposeKeypoint['people'][idx]
                
                #2. Fetch Player Position
                if(p1 is None):
                    if(clipInfo[testName]['p1']['frameNum'] == frameNum):
                        p1 = np.int32(clipInfo[testName]['p1']['pos'])
                        
                        #Find the pose of the player that best matches the labeled point
                        p1Changed,pose1,openposeKeypoint['people'] = findClosestPoseFromPoint(p1,openposeKeypoint['people'])
                else:
                        #Find the pose of the player that best matches the labeled point
                        p1Changed,pose1,openposeKeypoint['people'] = findClosestPoseFromPose(pose1,getThreshold(),openposeKeypoint['people'])
                        
                        #Update p1 variable with neck
                        p1 = np.int32([pose1[3], pose1[4]]) if pose1 is not None else p1
                
                if(p2 is None):
                    if(clipInfo[testName]['p2']['frameNum'] == frameNum):
                        p2 = np.int32(clipInfo[testName]['p2']['pos'])
                        
                        #Find the pose of the player that best matches the labeled point
                        p2Changed,pose2,openposeKeypoint['people'] = findClosestPoseFromPoint(p2,openposeKeypoint['people']) 
                else:
                        #Find the pose of the player that best matches the labeled point
                        p2Changed,pose2,openposeKeypoint['people'] = findClosestPoseFromPose(pose2,getThreshold(),openposeKeypoint['people'])
                        
                        #Update p1 variable with neck
                        p2 = np.int32([pose2[3], pose2[4]]) if pose2 is not None else p2
                
                #5. Update threshold with distance from both players
                if(p1Changed and p2Changed):
                    d = distancePoses(pose1,pose2)
                    if(d is not None):
                        distance_between_players.append(d)
                elif(p1Changed and not p2Changed):
                    hasChanged, tempPose2,openposeKeypoint['people'] = findClosestPoseFromPose(pose1,max(distance_between_players[-5:]),openposeKeypoint['people'])
                    if(hasChanged ):
                        pose2 = tempPose2
                    else:
                        #All c to 0.
                        for i,(x,y,c) in enumerate(chunker(pose2,3)):
                            pose2[3*i+2] = 0
                elif(not p1Changed and p2Changed):
                    hasChanged, tempPose1,openposeKeypoint['people'] = findClosestPoseFromPose(pose2,max(distance_between_players[-5:]),openposeKeypoint['people'])
                    if(hasChanged):
                        pose1 = tempPose1
                    else:
                        #All c to 0.
                        for i,(x,y,c) in enumerate(chunker(pose1,3)):
                            pose1[3*i+2] = 0
                
                #==================================
                #Keep Track of Non Involved Players
                #==================================
                notInvolvedPlayers = notInvolvedPlayersToAdd
                notInvolvedPlayersToAdd = openposeKeypoint['people']
                for pose in openposeKeypoint['people']:
                    notInvolvedPlayers.append(pose)
                
                #==========================
                #Save poses for each frame
                #==========================
                tempPose1 = [0]*75 if pose1 is None else pose1.copy()
                tempPose2 = [0]*75 if pose2 is None else pose2.copy()
                clipSave.append(FrameSave(frame,frameNum,p1,tempPose1,p2,tempPose2,notInvolvedPlayers.copy(),playersToDisplayDebug.copy()))
                playersToDisplayDebug = []
                
            #==================
            #Iterpolate joints
            #==================
            #1. Iterate thru all joints
            numOfJoints = 25 #BODY25
            for idx in range(numOfJoints):
                #2. Iterate thru all frames and find poses that needs interpolation
                p1HasValidJoint,p2HasValidJoint = False, False
                p1NeedsInter, p2NeedsInter = [],[]
                p1JointSave, p2JointSave = None, None
                p1LastFrameValid,p2LastFrameValid = False, False
                for framesave in clipSave:
                    pose1, pose2 = framesave.pose1, framesave.pose2
                    
                    #4. Check if joint is visible
                    jointP1_C, jointP2_C = pose1[3*idx+2], pose2[3*idx+2]
                    if(jointP1_C != 0 and len(p1NeedsInter) == 0):
                        p1HasValidJoint = True
                        p1JointSave = (pose1[3*idx],pose1[3*idx+1])
                    if(jointP2_C != 0 and len(p2NeedsInter) == 0):
                        p2HasValidJoint = True
                        p2JointSave = (pose2[3*idx],pose2[3*idx+1])
                    
                    #pre5. If last frame, all joints must have a location
                    if(framesave.frameNum == len(clipSave)-1 and jointP1_C == 0):
                        #Pick the closest valid joint
                        for closeIdx in DP10_CLOSEST_JOINTS[idx]:
                            if(pose1[3*closeIdx + 2] != 0):
                                pose1[3*idx], pose1[3*idx+1] = pose1[3*closeIdx],pose1[3*closeIdx + 1]
                                p1LastFrameValid = True
                                break
                    if(framesave.frameNum == len(clipSave)-1 and jointP2_C == 0):
                        #Pick the closest valid joint
                        for closeIdx in DP10_CLOSEST_JOINTS[idx]:
                            if(pose2[3*closeIdx + 2] != 0):
                                pose2[3*idx], pose2[3*idx+1] = pose2[3*closeIdx],pose2[3*closeIdx + 1]
                                p2LastFrameValid = True
                                break
                    
                    #5. If not visible and there is a previous framesave that contains joint, add to list
                    if(p1HasValidJoint and jointP1_C == 0):
                        p1NeedsInter.append(pose1)
                    if(p2HasValidJoint and jointP2_C == 0):
                        p2NeedsInter.append(pose2)
                    
                    #6. If we are at a frame where there is a joint but there werent previously, interpolate
                    if(len(p1NeedsInter) > 0 and p1JointSave is not None and (jointP1_C != 0 or p1LastFrameValid)):
                        #7. Interpolate with previous valid joint and this valid joint
                        x,y = pose1[3*idx],pose1[3*idx+1]
                        gen = weightGenerator(len(p1NeedsInter),p1JointSave,(x,y))
                        for jointNeedsInter in p1NeedsInter:
                            w0, w1 = next(gen)
                            jointNeedsInter[3*idx]   = w0*p1JointSave[0] + w1 * x
                            jointNeedsInter[3*idx+1] = w0*p1JointSave[1] + w1 * y
                        p1NeedsInter, p1JointSave = [],(x,y)
                        
                    if(len(p2NeedsInter) > 0 and p2JointSave is not None and (jointP2_C != 0 or p2LastFrameValid)):
                        #7. Interpolate with previous valid joint and this valid joint
                        x,y = pose2[3*idx],pose2[3*idx+1]
                        gen = weightGenerator(len(p2NeedsInter),p2JointSave,(x,y))
                        for jointNeedsInter in p2NeedsInter:
                            w0, w1 = next(gen)
                            jointNeedsInter[3*idx]   = w0*p2JointSave[0] + w1 * x
                            jointNeedsInter[3*idx+1] = w0*p2JointSave[1] + w1 * y
                        p2NeedsInter, p2JointSave = [],(x,y)
                    
                
            #======================================
            #View Results
            #======================================
            idx = 0
            while idx < len(clipSave):
                framesave = clipSave[idx]
                frame = framesave.frame
                idx = framesave.frameNum
                p1 = framesave.p1
                pose1 = framesave.pose1
                p2 = framesave.p2
                pose2 = framesave.pose2
                NIP = framesave.nip
                pdd = framesave.pdd
                
                #======================
                #Draw players on Screen
                #======================
                if(p1 is not None):
                    frame = cv2.circle(frame, (p1[0],p1[1]), 5, color = (255, 0, 255), thickness = 10)
                if(p2 is not None):
                    frame = cv2.circle(frame, (p2[0],p2[1]), 5, color = (255, 255, 0), thickness = 10)
                if(pose1 is not None):
                    frame = displayPlayer(frame, pose1,color = (255,0,0))
                if(pose2 is not None):
                    frame = displayPlayer(frame, pose2,color = (0,255,0))
                for notInvolvedPlayer in NIP:
                    frame = displayPlayer(frame,notInvolvedPlayer['pose_keypoints_2d'],color=(0,0,255))
                for p in pdd:
                    frame = displayPlayer(frame,p,color=(0,255,255))
                
                #=============
                #Display Image
                #=============
                cv2.imshow('frame',frame)
                
                #================
                #Await User Input
                #================
                while(True):
                    k = cv2.waitKey(20) & 0xFF
                    if(k == ord('.')):
                        break
                    elif(k == ord(',')):
                        frameNum -= 2
                        if(idx > 0):
                            idx -= 2
                        break
                    elif(k == ord('n')):
                        nextVid = True
                        break
                    elif(k == ord('q')):
                        terminate = True
                        break
                        
                #========================================
                #Check if needing Termination of Program
                #========================================
                if(terminate or nextVid):
                    nextVid = False
                    break
                    
                idx += 1
                
            #==================
            #Save Video and jsn
            #==================
            
            
            if(terminate):
                break
        if(terminate):
            break    

    #Terminate program peacefully
    cv2.destroyAllWindows()
     
