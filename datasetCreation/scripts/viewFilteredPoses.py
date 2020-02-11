import os 
import cv2
import json

DP10_FILTERED_POSES = '../FilteredPoses/'
DP10_OPENPOSE_RESULTS = '../openposeResults/'

def chunker(seq, size):
    if(seq is not None):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    else:
        return []

if __name__ == '__main__':
    #1. Fetch all clippaths
    filePaths = []
    for dir in os.listdir(DP10_FILTERED_POSES):
        for clipID in os.listdir(DP10_FILTERED_POSES + dir):
            filePaths.append('%s/%s'%(dir,clipID))
    
    terminate = False
    clipIdx = 0
    while(clipIdx < len(filePaths)):
        clipName = filePaths[clipIdx]
        phytType = clipName.split('/')[0]
        clipID = clipName.split('/')[1]
        print(clipIdx, clipName)
        
        #Fetch openpose video and filtered poses video
        openPosePath = DP10_OPENPOSE_RESULTS + clipName + '/output_vid/%s.mp4'%clipID
        capOP = cv2.VideoCapture(openPosePath)
        filteredPath = DP10_FILTERED_POSES + clipName + '/filtered_vid/%s.mp4'%clipID
        capF = cv2.VideoCapture(filteredPath)
        
        #Build List of OpenPose Videos
        frameListOpenPose = []
        while(capOP.isOpened()):
            #Read in Frame
            ret, frame = capOP.read()
            if(not ret):
                break
            frameListOpenPose.append(frame)
        
        #Build List of OpenPose Videos
        frameListFiltered = []
        while(capF.isOpened()):
            #Read in Frame
            ret, frame = capF.read()
            if(not ret):
                break
            frameListFiltered.append(frame)
            
        #Fetch JSON about pose
        jsnPath = DP10_FILTERED_POSES + clipName + '/filtered_jsn/%s.json'%clipID
        with open(jsnPath,'r') as f:
            JSNFilt = json.load(f)
        
        nextVid = False
        frameIdx = 0
        while(1):
            frameOP = frameListOpenPose[frameIdx].copy()
            frameF = frameListFiltered[frameIdx].copy()
            
            #Display information on filtered poses
            perpPose = JSNFilt[frameIdx]['perp']
            victimPose = JSNFilt[frameIdx]['victim']
            colorPerp = (0,0,255)
            colorVictim = (0,255,0)
            
            #Display poses with coloring
            for pose,color in zip([perpPose,victimPose],[colorPerp,colorVictim]):
                for x,y,c in chunker(pose,3):
                    xVals = [int(x) for x,_,c in chunker(pose,3) if x is not 0 and c is not 0]
                    yVals = [int(y) for _,y,c in chunker(pose,3) if y is not 0 and c is not 0]
                    if(len(xVals) != 0 ):
                        frameF = cv2.rectangle(frameF, (min(xVals) - 1 ,min(yVals) - 1 ), (max(xVals) + 1,max(yVals) + 1), color=color, thickness = 2 )
                    
                    if(c == 0):
                        tempColor = (int(color[0]/2),int(color[1]/2),int(color[2]/2))
                    else:
                        tempColor = color
                    frameF = cv2.circle(frameF, (int(x),int(y)), 1, color = tempColor, thickness = 10)
            
            #Draw Frame Number
            shape = frameOP.shape
            if(shape[1] > 1000):
                frameF = cv2.rectangle(frameF, (10,10),(450,60), color=(0,0,0), thickness = -1 )
                frameOP = cv2.rectangle(frameOP, (10,10),(240,60), color=(0,0,0), thickness = -1 )
            frameF = cv2.putText(frameF,"Frame#: %d"%frameIdx, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            frameOP = cv2.putText(frameOP,"Frame#: %d"%frameIdx, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            
            #Draw Legend
            frameF = cv2.putText(frameF,"Perp", (240,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
            frameF = cv2.putText(frameF,"Victim", (320,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            
            if(shape[1] > 1000):
                resizedOP = cv2.resize(frameOP, (int(shape[1]/2),int(shape[0]/2)))
                resizedF = cv2.resize(frameF, (int(shape[1]/2),int(shape[0]/2)))
            else:
                resizedOP = frameOP
                resizedF = frameF
            cv2.imshow('openpose',resizedOP)
            cv2.imshow('filtered',resizedF)
            
            
            while(True):
                k = cv2.waitKey(20) & 0xFF
                if(k == ord('.')):
                    if(frameIdx == len(frameListOpenPose) - 1):
                        frameIdx = -1
                    break
                elif(k == ord(',')):
                    if(frameIdx == 0):
                        frameIdx += len(frameListOpenPose)-2
                    else:
                        frameIdx -= 2                            
                    break
                elif(k == ord('n')):
                    nextVid = True
                    break
                elif(k == ord('q')):
                    terminate = True
                    break
            
            if(terminate or nextVid):
                nextVid = False
                break
            frameIdx += 1
            frameIdx = frameIdx % len(frameListFiltered)
            
        if(terminate):
            break
        clipIdx += 1