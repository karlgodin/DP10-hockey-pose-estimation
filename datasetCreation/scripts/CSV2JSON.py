import glob
import json

#Constants
labels = {
    0: "Tripping",
    1: "Cross-Checking",
    2: "No Penalty"
}
shotTypes = {
    0:"Short",
    1:"Medium",
    2:"Long"
}

class c_penalty:
    def __init__(self,ID,start,end,label,camera,flags):
        self.clip_ID = ID
        self.start_time = start
        self.end_time = end
        self.label = label
        self.camera_shot = camera
        self.flags = flags.__dict__

class c_flags:
    def __init__(self,location,fighting,twoPlayers,caption):
        self.middle_rink = True if location else False
        self.has_fighting = True if fighting else False
        self.two_players = True if twoPlayers else False
        self.has_caption = True if caption else False
class c_video:
    def __init__(self,url,resolution):
        self.url = url
        self.video_resolution = resolution
        self.penalties = []

if __name__ == "__main__":
    files = glob.glob("../csv/*.*")
    #Read in CSV
    
    videoIdx = 0
    clipIdx = 0
    videoList = {}
    videoInfo = {}
    numberOfPenaltyClipByType = {i : 0 for i in range(len(labels))}
    for filename in files:
        #CSV file
        if filename.endswith(".csv"):
            fin = open(filename,'r')
            for line in fin:
                content = line.split(',')
                URL = content[0].split('=')[1].split('&')[0]
                resolution = int(content[1])
                start = int(content[2])
                end = int(content[3])
                label = int(content[4])
                camera = int(content[5])
                location = int(content[6])
                fighting = int(content[7])
                twoPlayers = int(content[8])
                caption = int(content[9])
                                
                flags = c_flags(location,fighting,twoPlayers,caption)
                penalty = c_penalty(clipIdx,start,end,label,camera,flags)
                clipIdx = clipIdx + 1
                
                #Check if Key is not in dictionary
                if (URL not in videoList):
                    video = c_video(URL,resolution)
                    videoName = "video_" + '%04d' % videoIdx
                    videoList[URL] = videoName
                    
                    #Create new video info entry
                    videoInfo[videoName]= video.__dict__
                    
                    videoIdx = videoIdx + 1
                else:
                    videoName = videoList[URL]
                
                   
                videoInfo[videoName]['penalties'].append(penalty.__dict__)
        
                #Update Statistics
                numberOfPenaltyClipByType[label] +=1 
                
        #Not CSV file
        else:
            pass
    
    #Build Dataset Description
    datasetDescription = {}
    datasetDescription['NumOfVideos'] = videoIdx 
    datasetDescription['NumOfPenaltyClips'] = clipIdx
    temp_NumberOfPenaltyClipByType = { labels[i] : numberOfPenaltyClipByType[i] for i in range(len(labels))}
    datasetDescription['NumOfPenaltyClipsByType'] = temp_NumberOfPenaltyClipByType
    
    out = {}
    out['dataset_description'] = datasetDescription
    out["video_ID"] = videoList
    out["labels"] = labels
    out["camera_shot_types"] = shotTypes
    out["video_description"] = videoInfo
    fout = open("../json/PHYT.json",'w')  
    json.dump(out,fout,indent=4, sort_keys=True)