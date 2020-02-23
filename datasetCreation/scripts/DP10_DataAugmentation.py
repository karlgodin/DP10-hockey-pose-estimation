import numpy as np


def DP10_getDistanceVector2Poses(data,p1,j1,p2,j2):
    """
        Calculates the Distance Vector between the joint of p1 and the joint of p2.
        Returns a list of euclidean distances
        
        Parameters
        ----------
        data: Dict of filtered poses for each frame
        p1: str. 'perp' or 'victim'
        j1: number between 0-24. Follows OpenPose output convention for BODY25
        p2: str. 'perp' or 'victim'
        j2: number between 0-24. Follows OpenPose output convention for BODY25
        
        Returns
        -------
        List of euclidean distances
    """
    #Argument Check
    for p in [p1,p2]:
        if(p is not 'perp' and p is not 'victim'):
            raise Exception()
    for j in [j1,j2]:
        if(j<0 or j > 24):
            raise Exception()
    
    #Iterate through all sorted frames
    data_sorted = sorted(data,key = lambda x: x['frameNum'])
    out = []
    for frame in data_sorted:
        (x1,y1) = frame[p1][3*j1:3*j1+2]
        (x2,y2) = frame[p2][3*j2:3*j2+2]
        
        #Calculate euclidean distance
        dist = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
        out.append(dist)
        
    return out
   
def DP10_getMotionVector2Poses(data,p1,j1,p2,j2):
    """
        Calculates the Motion Vector between the joint of p1 and the joint of p2.
        Returns a list of euclidean distances
        
        Parameters
        ----------
        data: Dict of filtered poses for each frame
        p1: str. 'perp' or 'victim'
        j1: number between 0-24. Follows OpenPose output convention for BODY25
        p2: str. 'perp' or 'victim'
        j2: number between 0-24. Follows OpenPose output convention for BODY25
        
        Returns
        -------
        List of euclidean distances
    """
    #Argument Check
    for p in [p1,p2]:
        if(p is not 'perp' and p is not 'victim'):
            raise Exception()
    for j in [j1,j2]:
        if(j<0 or j > 24):
            raise Exception()
    
    #Iterate through all sorted frames
    data_sorted = sorted(data,key = lambda x: x['frameNum'])
    out = []
    for frame1,frame2 in zip(data_sorted[:-1],data_sorted[1:]):
        (x1,y1) = frame1[p1][3*j1:3*j1+2]
        (x2,y2) = frame2[p2][3*j2:3*j2+2]
        
        #Calculate euclidean distance
        motion = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
        out.append(motion)
        
    return out

if __name__ == '__main__':
    import os
    import json
    DP10_FILTERED_POSES = '../FilteredPoses/'
    
    #1. Fetch all clippaths
    filePaths = []
    for dir in os.listdir(DP10_FILTERED_POSES):
        for clipID in os.listdir(DP10_FILTERED_POSES + dir):
            filePaths.append('%s/%s'%(dir,clipID))
    
    #2. To test, take only the first Result
    clipPath = filePaths[0]
    PHYT_Type = clipPath.split('/')[0]
    clipID = clipPath.split('/')[1]
    print('Testing with: %s'%clipPath)
    
    #3. Open JSON file
    with open(DP10_FILTERED_POSES+clipPath+'/filtered_jsn/%s.json'%clipID) as f:
        posesList = json.load(f)
    
    distVector = DP10_getDistanceVector2Poses(posesList,'perp',2,'victim',3)
    print('Distance Vector:\n', distVector)
    
    motionVector = DP10_getMotionVector2Poses(posesList,'perp',2,'victim',3)
    print('Motion Vector:\n',motionVector)