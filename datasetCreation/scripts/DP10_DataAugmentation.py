from numpy import sqrt

def chunker(seq, size):
    if(seq is not None):
        return [seq[pos:pos + size] for pos in range(0, len(seq), size)]
    else:
        return []

def DP10_getDistanceVector2Poses(p1,p2):
    """
        Calculates the Distance Vector between the joint of p1 and the joint of p2.
        Returns a list of euclidean distances
        
        Parameters
        ----------
        p1: List of x,y,x for a joint
        p2: List of x,y,c for a joint
        
        Returns
        -------
        List of euclidean distances
    """    
    #Argument Check
    for j in [len(p1),len(p2)]:
        if((j-1)%3 != 0):
            raise Exception()
    
    #Iterate through all sorted frames
    out = []
    chunkedP1 = chunker(p1[:-1],3)
    chunkedP2 = chunker(p2[:-1],3)
    for (x1,y1,c1),(x2,y2,c2) in zip(chunkedP1,chunkedP2):
        #Calculate euclidean distance
        dist = sqrt( (x1-x2)**2 + (y1-y2)**2 )
        out.append(dist)
        
    return out
   
def DP10_getMotionVector2Poses(p1,p2):
    """
        Calculates the Distance Vector between the joint of p1 and the joint of p2.
        Returns a list of euclidean distances
        
        Parameters
        ----------
        p1: List of x,y,x for a joint
        p2: List of x,y,c for a joint
        
        Returns
        -------
        List of euclidean distances
    """    
    #Argument Check
    for j in [len(p1),len(p2)]:
        if((j-1)%3 != 0):
            raise Exception()
    
    #Iterate through all sorted frames
    out = []
    chunkedP1 = chunker(p1[:-1],3)[:-1]
    chunkedP2 = chunker(p2[:-1],3)[1:]
    for (x1,y1,c1),(x2,y2,c2) in zip(chunkedP1,chunkedP2):
        #Calculate euclidean distance
        dist = sqrt( (x1-x2)**2 + (y1-y2)**2 )
        out.append(dist)
        
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
    
    #3. Create Test Data
    numOfFrames = 93
    p1,p2 = [], []
    for i in range(numOfFrames):
        p1.append((5+i*10))
        p1.append((10+i*20))
        p1.append((0.1*i)%1)
        
        p2.append((2+i))
        p2.append((4+i))
        p2.append((0.2*i)%1)
    p1.append(2.0)
    p2.append(3.0)
    
    distVector = DP10_getDistanceVector2Poses(p1,p2)
    print('Distance Vector:\n', distVector)
    print('Length: ',len(distVector))
    
    motionVector = DP10_getMotionVector2Poses(p1,p2)
    print('Motion Vector:\n',motionVector)
    print('Length: ',len(motionVector))