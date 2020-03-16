import os
import json

DP10_PHYT_FOLDER = "../json/" 
DP10_FILTERED_POSES = '../FilteredPoses/'

if __name__ == "__main__":
    #1. Fetch all clippaths
    filePaths = []
    for dir in os.listdir(DP10_FILTERED_POSES):
        for clipID in os.listdir(DP10_FILTERED_POSES + dir):
            filePaths.append('%s/%s'%(dir,clipID))
    
    #2. Fetch Phyt Dataset
    phytDataset = {} 
    for filename in os.listdir(DP10_PHYT_FOLDER):
        if('PHYT' in filename):
            with open(DP10_PHYT_FOLDER + filename) as f:
                phytDataset[filename.split('.')[0]] = json.load(f)
    
    #3. Create output for each video.
    count = []
    for filePath in filePaths:
        PHYT_Name = filePath.split("/")[0]
        clipID = int(filePath.split("/")[1].split("_")[1])
        
        #4. Find output label
        for video in phytDataset[PHYT_Name]['video_description']:
            foundClip = False
            for penalty in phytDataset[PHYT_Name]['video_description'][video]['penalties']:
                if(penalty['clip_ID'] == clipID):
                    label = penalty['label']
                    label = (phytDataset[PHYT_Name]['labels'][str(label)])
                    foundClip = True
                    break
            if(foundClip):
                break
        
        #5. Write output file
        outjsn = {}
        outjsn['Definitions'] = {'penaltyOnly':['Tripping','Cross-Checking','No Penalty'],'penaltyAndPlayerRole':['%perp','%victim']}
        if(label == 'No Penalty'):
            outjsn['penaltyOnly'] = [0,0,1]
            outjsn['penaltyAndPlayerRole'] = [0.5,0.5]
        elif (label == 'Tripping'):
            outjsn['penaltyOnly'] = [1,0,0]
            outjsn['penaltyAndPlayerRole'] = [1.0,0.0]
        elif (label == 'Cross-Checking'):
            outjsn['penaltyOnly'] = [0,1,0]
            outjsn['penaltyAndPlayerRole'] = [1.0,0.0]
        
        with open(DP10_FILTERED_POSES + filePath+ '/filtered_jsn/label.out','w') as f:
            json.dump(outjsn, f, ensure_ascii=False, indent=4)
        