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
import pandas as pd
import shutil
import copy

DP10_OPENPOSE_RESULTS_FOLDER = "../openposeResults/"
DP10_OUTPUT_FOLDER = "../FilteredPoses/"
DP10_TRIMMED_FOLDER = '../Clips/Trimmed/'
DP10_CLIPS_FOLDER = '../Clips/'
DP10_POSE_THRESHOLD_COEFF = 2
DP10_TTL_POSES = 10
DP10_THRESHOLD = 250
DP10_MIN_NUM_OF_JOINTS = 10
DP10_DO_INTERPOLATION = True
DP10_DISPLAY_NON_INVOLVED = True
DP10_DISPLAY_BOUNDING_BOX = True
DP10_DISPLAY_LOST_PLAYERS = False
DP10_INTERPOLATION_STYLE = 'linear'
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
colorList = [(255,0,0),(0,255,0),(0,0,255)]
#=================
#HELPER FUNCTIONS
#=================
    
def getThreshold(height,width):
    return DP10_THRESHOLD* (height/1280 ) * width / 760
 
def weightGenerator(n,firstPnt, secondPnt):
    if(DP10_INTERPOLATION_STYLE is 'linear' or DP10_INTERPOLATION_STYLE is 'polynomial'):
        s = n+1
        for i in range(1,s):
            yield (1 - i/s,i/s)
            
    elif(DP10_INTERPOLATION_STYLE is 'piecewise'):
        for _ in range(n//2):
            yield(1,0)
        for _ in range(n - n//2):
            yield(0,1)
            
    elif (DP10_INTERPOLATION_STYLE is 'square'):
        s = n+1
        for i in range(1,s):
            val = (i/s)**2
            yield (1 - (val),val)
    elif(DP10_INTERPOLATION_STYLE is 'exp'):
        s = n+1
        for i in range(1,s):
            val = np.exp(-i)
            yield(val,1-val)
    elif(DP10_INTERPOLATION_STYLE is 'sinusodial'):
        s = n+1
        for i in range(1,s):
            val = np.cos(0.5*np.pi*i/s)
            yield(val,1-val)
    elif(DP10_INTERPOLATION_STYLE is 'pascal2'):
        s = n+1
        for i in range(1,s):
            val = 1 - 2*i/s + (i/s)**2
            yield(val,1-val)
    elif(DP10_INTERPOLATION_STYLE is 'specialSpice'):
        s = n+1
        for i in range(1,s):
            val = 1 - 4*i/s + 6*(i/s)**2 - 4*(i/s)**3 + (i/s)**4
            yield(val,1-val)
    elif(DP10_INTERPOLATION_STYLE is 'splice'):
        #Need three data points to fit splice
        pass
    else:
        while True:
            yield (0,1)
 
def chunker(seq, size):
    if(seq is not None):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    else:
        return []

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
    
def findIndexOfClosestPoseFromPoint(poses, point):
    distanceList = []
    for pose in poses:
        numOfJoints = 0
        d = None
        for (x,y,c) in chunker(pose,3):
            if(c != 0):
                tempD = (x-point[0])**2 + (y-point[1])**2
                if(d is None):
                    d = tempD
                elif (tempD < d):
                    d = tempD
        distanceList.append(d)
    return distanceList.index(min(distanceList))

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
    
def getDistanceMatrixPoseandPose(prevPoses, currentPoses):
    """
        Returns a nxm Pandas Dataframe that contains distances between
        every poses from previous frame and every pose in current frame
    """
    df_distances = pd.DataFrame(columns = range(len(currentPoses)))
    for prevPose in prevPoses:
        #For every pose in previous Frame, get distance to every pose in current frame
        distanceList = []
        for currPose in currentPoses:
            distanceList.append(distancePoses(prevPose,currPose))
        
        #Build pandas Series of distances
        serDistanceOfPrev = pd.Series(distanceList)
        
        #Append to distance matrix
        df_distances = df_distances.append(serDistanceOfPrev, ignore_index=True)
    
    #The distance matrix has been built.
    return df_distances
    
def getDistanceMatrixPoseandPoint(poses, point):
    """
        Returns a nxm Pandas Series that contains distances between
        every poses and point
    """
    distanceList = []
    for pose in poses: 
        tempDist = []
        hasAValidJoint = False
        for (x,y,c) in chunker(pose,3):
            if(c != 0 ):
                hasAValidJoint = True
                dist = np.linalg.norm([x-point[0],y-point[1]])
                tempDist.append(dist)
        if(hasAValidJoint):
            distanceList.append(min(tempDist))
    return pd.Series(distanceList)

def findIdxesOfMinInDataFrame(df):
    """
        Returns (rowIdx,colIdx) of min in DataFrame
    """
    #Find all mins
    minIdx = (0,0)
    minVal = np.inf
    for colName in df.columns:
        rowName = df.loc[:][colName].idxmin()
        tempVal = df.loc[rowName][colName]
        
        #Kepp track of minimum seen up to now
        if(tempVal < minVal):
            minIdx = (rowName,colName)
            minVal = tempVal
    
    #(row,col) of minimum value in df
    return minIdx, minVal
    
def matchPlayers(df):
    """
        Tracks players between two frames by finding the minimum distance between
        two poses and matching between them.
        
        Returns list of tuple of matching index, where:
        [(prevIdx1, currentIdx1),(prevIdx2, currentIdx2),...]
        
        and PrevIdx of lost players
    """
    listOfMatches = []
    while(len(df.index) > 0 and len(df.columns) > 0):
        #Find row and col of minimum values
        idxes,minVal = findIdxesOfMinInDataFrame(df)
        
        #Terminate if no matches under threshold
        if(minVal > getThreshold(clipHeight,clipWidth)):
            break
        
        #Save indexes
        listOfMatches.append(idxes)
        
        #Remove row and column of minimum value
        df.drop(labels = idxes[0], axis = 0, inplace = True)    #Drop row
        df.drop(labels = idxes[1], axis = 1, inplace = True)    #Drop col
    
    #Build list of lost players
    lostPlayers = df.index
    newPlayers = df.columns
    
    #Return
    return listOfMatches,lostPlayers,newPlayers

def hasAtLeastKJoints(pose, k):
    """
        Counts number of Joints.
        Returns true if num Of Joints >= k
    """
    count = 0
    for (x,y,c) in chunker(pose,3):
        if(c!=0):
            count+=1
    return True if count>=k else False
    
def save(clipName,frameList):
    """
        Save filtered poses to File
    """
    #Get the Last Frame with Valid Data
    lastFrameNum = -1
    for frame in frameList:
        for pIdx in frame.pi:
            for x,y,c in chunker(frame.openpose[pIdx],3):
                if(c != 0 ):
                    lastFrameNum = frame.frameNum
    
    #Create/Clear folders
    rootPath = DP10_OUTPUT_FOLDER+clipName+'/'
    jsnPath = rootPath + 'filtered_jsn/'
    vidPath = rootPath + 'filtered_vid/'
    if(os.path.exists(rootPath)):   #Remove what already exists
        shutil.rmtree(rootPath)
    os.makedirs(os.path.dirname(rootPath))
    os.makedirs(os.path.dirname(jsnPath))
    os.makedirs(os.path.dirname(vidPath))
    
    #Create Video and output JSON
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vidPath + '%s.mp4'%clipName.split('/')[1],fourcc,frameList[0].fps,frameList[0].res)
    outJson = []
    for i in range(lastFrameNum + 1):
        frame = frameList[i]
        tempDict = {}
        tempDict['frameNum'] = frame.frameNum
        for pIdx, playerName in zip(frame.pi,['perp','victim']):
            tempDict[playerName] = frame.openpose[pIdx]
        out.write(frame.frame)
        outJson.append(tempDict)
    
    #Save to File
    out.release()
    with open(jsnPath + '%s.json'%clipName.split('/')[1], 'w', encoding='utf-8') as f:
        json.dump(outJson, f, ensure_ascii=False, indent=4)
    
    
class frameSave():
    def __init__(self,frameNum,frame,openpose,points,fps,height,width):
        self.frameNum = frameNum
        self.frame = frame
        self.openpose = openpose
        self.pi = [None,None]
        self.lostPlayersPoses = []
        self.lastPIPose = []
        self.points = points
        self.fps = int(np.round(fps))
        self.res = (int(np.round(width)),int(np.round(height)))
    
if __name__ == '__main__':
    #1. Fetch all clippaths
    filePaths = []
    for dir in os.listdir(DP10_OPENPOSE_RESULTS_FOLDER):
        for clipID in os.listdir(DP10_OPENPOSE_RESULTS_FOLDER + dir):
            filePaths.append('%s/%s'%(dir,clipID))
    
    #2. Fetch Players involved
    with open(DP10_TRIMMED_FOLDER + 'playersInvolved.json','r') as f:
        playersInvolved = json.load(f)
    
    #3. Read in video and filter poses
    clipIdx = 0
    while clipIdx < len(filePaths):
        clipName = filePaths[clipIdx]
        clipPath = DP10_TRIMMED_FOLDER + clipName + '.mp4'
        print(clipIdx, clipName)
        
        #Read in Video
        frameList = []
        frameNum = 0
        cap = cv2.VideoCapture(clipPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        clipWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        clipHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        while(cap.isOpened()):
            #Read in Frame
            ret, frame = cap.read()
            if(not ret):
                break
            
            #Read in JSON file
            with open(DP10_OPENPOSE_RESULTS_FOLDER + clipName+'/output_jsn/'+clipName.split('/')[1]+'_'+str(frameNum).zfill(12) + '_keypoints.json') as f:
                openPoseResults = json.load(f)
            
            #Create list of poses
            poses = []
            for pose in openPoseResults['people']:
                if(hasAtLeastKJoints(pose['pose_keypoints_2d'],DP10_MIN_NUM_OF_JOINTS)):
                    poses.append(pose['pose_keypoints_2d'])
            
            #Read in points
            points = playersInvolved[clipName + '.mp4']
            
            #Save member
            frameList.append(frameSave(frameNum,frame,poses,points,fps,clipHeight,clipWidth))
            
            
            #Update Clip Members
            frameNum += 1
        
        #==========================
        #PERFORM FILTERING OF POSES
        #==========================
        #Entry point, identify labeled players 
        for frameObj in frameList:
            frameIdx = frameObj.frameNum
            points = frameObj.points
            frameObj.points = copy.deepcopy(points)
            
            #Build dataframe with columns as pose idx and index as labelled player
            dist_df = pd.DataFrame(columns = range(len(frameObj.openpose)))
            for p in ['p1','p2']:
                if(points[p]['frameNum'] == frameIdx):
                    tempSer = getDistanceMatrixPoseandPoint(frameObj.openpose,frameObj.points[p]['pos'])
                    tempSer.name = p
                    dist_df = dist_df.append(tempSer)
            
            toLabel = list(dist_df.index)
            for _ in range(len(dist_df)):
                if(dist_df.empty):
                    for p in toLabel:
                        points[p]['frameNum'] += 1
                    break
                #Fetch which player is closer to which label
                minIdx, minDist = findIdxesOfMinInDataFrame(dist_df)
                dist_df.drop(minIdx[0],axis=0,inplace=True)
                dist_df.drop(minIdx[1],axis=1,inplace=True)
                
                if( not (minDist < 0.5*getThreshold(clipHeight,clipWidth))):
                    points[minIdx[0]]['frameNum'] += 1
                else:
                    frameObj.pi[['p1','p2'].index(minIdx[0])] = minIdx[1]
                del toLabel[toLabel.index(minIdx[0])]
            
        #First Pass Through
        hasAPoint = False
        prevFrameObj = frameList[0]
        for frameObj in frameList[1:]:
            #0.5 Update Lost poses index
            frameObj.lostPlayersPoses = [(pose,ttl - 1,piIdx) for pose,ttl,piIdx in prevFrameObj.lostPlayersPoses if (ttl > 0 or piIdx is not None)]
        
            #Check if you have players involved in previous frames
            if(len([temp for temp in prevFrameObj.pi if temp is not None]) > 0):
                hasAPoint = True
            if(hasAPoint):
                #Need To Interpolate players.
                #To do so, will use a variant of Dancing Links algorithm by Prof. Donald Knuth
                
                #1. Build nxm Matrix of distances between poses. 
                #       where   n = number of poses in previous frame.
                #               m = number of poses in this frame.
                dl_df = getDistanceMatrixPoseandPose(prevFrameObj.openpose, frameObj.openpose)
                
                #2. While there are distance values under THRESHOLD, 
                #   remove row and column that contains min value as
                #   it is a match :)
                #   Convention for matches: (prevPlayer, currentPlayer)
                matches,prevlostIndexes,newPlayersIndex = matchPlayers(dl_df.copy())
                
                #3. Match lostPlayer poses and newPlayers
                lostPoses = [pose for pose,_,_ in frameObj.lostPlayersPoses]
                newPlayers = [(i,pose) for i,pose in enumerate(frameObj.openpose) if i in newPlayersIndex]
                newPlayerInvolved = [None,None]
                if(len(lostPoses) > 0 and len(newPlayers) > 0):
                    lostAndNew_df = getDistanceMatrixPoseandPose(lostPoses,[pose for _,pose in newPlayers])
                    
                    #Find matches between lost players and new players
                    LostAndNew_matches,_,_ = matchPlayers(lostAndNew_df.copy())
                     
                    #Remove lost player because we have found him again :)
                    LostAndNew_matches = sorted(LostAndNew_matches,key = lambda x:x[0],reverse = True)
                    for match in LostAndNew_matches:
                        #If a newly found player is an involved player,
                        #Then add it to the list for the current Frame!
                        if(frameObj.lostPlayersPoses[match[0]][2] is not None):
                            newPlayerInvolved[frameObj.lostPlayersPoses[match[0]][2]] = newPlayers[match[1]][0]
                        
                        del frameObj.lostPlayersPoses[match[0]]
                
                
                #4. Find the index of the players involved and 
                #   update the players involved of current frame.
                for idx,prevPI in enumerate(prevFrameObj.pi):
                    #Search thru list of matches.
                    isThereAMatch = False
                    for match in matches:
                        #If match, then add playerInvolved in current Frame
                        if(match[0] == prevPI):
                            frameObj.pi[idx] = match[1]
                            isThereAMatch = True
                            break
                        
                    if(not isThereAMatch):
                        #We have lost track of player.
                        #Generally, can be due to 2 scenarios
                        #   1. OpenPose could not estimate any joint of the player
                        #   2. The Threshold used is too low. 
                        
                        #Trying to solve case 1.
                        #   From observations: We generally get a pose within a few frames of losing it.  
                        #   Will append pose of lost player in prevFrame to current Frame.
                        if(newPlayerInvolved[idx] is not None):
                            frameObj.pi[idx] = newPlayerInvolved[idx]
                            isThereAMatch = True
                        
                
                #Update Lost Poses
                for idx in prevlostIndexes:
                    piIdx = None
                    if idx in prevFrameObj.pi and idx is not None:
                        piIdx = prevFrameObj.pi.index(idx)
                    frameObj.lostPlayersPoses.append((prevFrameObj.openpose[idx],DP10_TTL_POSES,piIdx))
                        
            prevFrameObj = frameObj
            
        #=============
        #INTERPOLATION
        #=============
        if(DP10_DO_INTERPOLATION):
            for jointIdx in range(25):
                for piIdx,p in enumerate(['p1','p2']):
                    hasValidLeft = False
                    p1 = None
                    toInterpolate = []
                    for frameObj in frameList:
                        poseIdx = frameObj.pi[piIdx]
                        
                        if(poseIdx is None):
                            frameObj.pi[piIdx] = len(frameObj.openpose)
                            frameObj.openpose.append([0]*75) #BODY25
                            poseIdx = frameObj.pi[piIdx]
                            
                        
                        x = frameObj.openpose[poseIdx][3*jointIdx + 0]
                        y = frameObj.openpose[poseIdx][3*jointIdx + 1]
                        c = frameObj.openpose[poseIdx][3*jointIdx + 2]
                        
                        #Must have a previous value to interpolate
                        if(c > 0 and not hasValidLeft):
                            hasValidLeft = True
                        if(not hasValidLeft):
                            continue
                        
                        
                        if(c == 0):
                            toInterpolate.append(frameObj)
                            #Remove the lost pose 
                            for i,(_,_,p) in enumerate(frameObj.lostPlayersPoses):
                                if(p == piIdx):
                                    del frameObj.lostPlayersPoses[i]
                                    break
                                
                        elif(len(toInterpolate) > 0):
                            p2 = (x,y)
                            #Interpolate here
                            gen = weightGenerator(len(toInterpolate),p1,p2)
                            for frameToInterpolate in toInterpolate:
                                w0, w1 = next(gen)
                                tempPoseIdx = frameToInterpolate.pi[piIdx]
                                frameToInterpolate.openpose[tempPoseIdx][3*jointIdx + 0] = w0 * p1[0] + w1 * p2[0]
                                frameToInterpolate.openpose[tempPoseIdx][3*jointIdx + 1] = w0 * p1[1] + w1 * p2[1]                            
                            
                            p1 = (x,y)
                            toInterpolate = []
                        else:
                            p1 = (x,y)
                    
                    #We are at the end of the clip, but there are still some joints to 
                    # interpolate
                    if(len(toInterpolate) > 0):
                        for frameToInterpolate in toInterpolate:
                            tempPoseIdx = frameToInterpolate.pi[piIdx]
                            frameToInterpolate.openpose[tempPoseIdx][3*jointIdx + 0] = p1[0]
                            frameToInterpolate.openpose[tempPoseIdx][3*jointIdx + 1] = p1[1]                       
        #============
        #VIEW RESULTS
        #============
        frameIdx = 0
        nextVid = False
        terminate = False
        while frameIdx < len(frameList):
            frameObj = frameList[frameIdx]
        
            frameNum = frameObj.frameNum
            frame = frameObj.frame.copy()
            openpose = frameObj.openpose
            points = frameObj.points
            posesInvolved = frameObj.pi
            lp = frameObj.lostPlayersPoses
            
            #===============
            #Draw on Frame
            #===============
            #Draw joints of players
            for i,pose in enumerate(openpose):
                if(i == posesInvolved[0]):
                    color = (0,0,255)
                elif(i == posesInvolved[1]):
                    color = (0,255,0)
                elif(i not in posesInvolved):
                    if(DP10_DISPLAY_NON_INVOLVED):
                        color = (255,0,0)
                    else:
                        continue
                else:
                    continue
                #Display bounding box
                if(i in posesInvolved and DP10_DISPLAY_BOUNDING_BOX):
                    xVals = [int(x) for x,_,c in chunker(pose,3) if x is not 0]
                    yVals = [int(y) for _,y,c in chunker(pose,3) if y is not 0]
                    if(len(xVals) == 0 ):
                        continue
                    frame = cv2.rectangle(frame, (min(xVals) - 1 ,min(yVals) - 1 ), (max(xVals) + 1,max(yVals) + 1), color=color, thickness = 2 )
                
                for (x,y,c) in chunker(pose,3):
                    if(c == 0):
                        if(not DP10_DO_INTERPOLATION):
                            continue
                        tempColor = (color[0]//2,color[1]//2,color[2]//2)
                    else:
                        tempColor = color
                    frame = cv2.circle(frame, (int(x),int(y)), 1, color = tempColor, thickness = 10)
            
            #LOST PLAYERS
            if(DP10_DISPLAY_LOST_PLAYERS):
                for pose,_,_ in lp:
                    for (x,y,c) in chunker(pose,3):
                        frame = cv2.circle(frame, (int(x),int(y)), 1, color = (255,255,0), thickness = 10)
            
            #Draw points
            for p,color in zip(['p1','p2'],[(255,0,255),(0,255,255)]):
                if(frameNum == points[p]['frameNum']):
                    x = points[p]['pos'][0]
                    y = points[p]['pos'][1]
                    frame = cv2.circle(frame, (int(x),int(y)), 1, color = color, thickness = 10)
            
            #Draw Frame Number
            frame = cv2.putText(frame,"Frame#: %d"%frameNum, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,127))
            #Draw Legend
            frame = cv2.putText(frame,"Perp", (230,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            frame = cv2.putText(frame,"Victim", (320,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            
            #=============
            #Display Frame
            #=============
            windowName = '%s'%clipName
            cv2.imshow(windowName,frame)
            
            #================
            #Await User Input
            #================
            while(True):
                k = cv2.waitKey(20) & 0xFF
                if(k == ord('.')):
                    if(frameIdx == len(frameList) - 1):
                        frameIdx = -1
                    break
                elif(k == ord(',')):
                    if(frameIdx == 0):
                        frameIdx += len(frameList)-2
                    else:
                        frameIdx -= 2                            
                    break
                elif(k == ord('n')):
                    nextVid = True
                    break
                elif(k == ord('s')):
                    save(clipName,frameList)
                    nextVid = True
                    break
                elif(k == ord('q')):
                    terminate = True
                    break
            
            if(nextVid or terminate):
                break
            
            #Update members
            frameIdx += 1
        
        if(terminate):
            break
        
        #Close current window
        cv2.destroyWindow(windowName)
        
        #Update Members
        clipIdx += 1
        
            
    #Terminate program peacefully
    cv2.destroyAllWindows()
     
