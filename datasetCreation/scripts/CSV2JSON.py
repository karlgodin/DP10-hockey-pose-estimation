import glob
import json
import YTVideoInfo as ytvi
import argparse
import os

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
        self.flags = flags

class c_flags:
    def __init__(self,location,fighting,twoPlayers,caption,isSeparable):
        self.middle_rink = True if location else False
        self.has_fighting = True if fighting else False
        self.two_players = True if twoPlayers else False
        self.has_caption = True if caption else False
        self.isSeperable = (True if isSeparable else False) if isSeparable != 2 else None

class c_video:
    def __init__(self,resolution):
        self.video_resolution = resolution
        self.penalties = []

class c_list:
    def __init__(self):
        self.items = []
        self.count = 0
        
    def addToList(self,item):
        self.items.append(item)
        self.count = self.count + 1 

class c_statistics:
    def __init__(self):
        self.twoPlayers = 0
        self.seperable = 0
        self.HD = 0
        self.middleOfRink = 0
        self.twoPlayersOrSeperable = 0
        self.twoPlayersOrSeperableAndHDAndMiddle = 0
        self.count = 0
        
    def update1(self, clipID, resolution,penalty):
        tempFlags = penalty.flags
        
        #Check Two Players
        if(tempFlags.two_players):
            self.twoPlayers += 1
            
        #Check Seperable
        if(tempFlags.isSeperable):
            self.seperable += 1
            
        #Check HD
        if(resolution > 480):
            self.HD += 1
            
        if(tempFlags.middle_rink):
            self.middleOfRink += 1
            
        if(tempFlags.two_players or tempFlags.isSeperable):
            self.twoPlayersOrSeperable += 1
            
        if((tempFlags.two_players or tempFlags.isSeperable) and resolution > 480 and tempFlags.middle_rink):
            self.twoPlayersOrSeperableAndHDAndMiddle += 1
        
        self.count += 1
        
class c_datasetStatistics(c_statistics):
    def __init__(self):
        super().__init__()
        self.YoutubeVideos = c_list()
        
    def update1(self, URL, clipID, resolution,penalty):
        super().update1(clipID, resolution,penalty)
        
        #Check if Key is not in dictionary
        if (URL not in self.YoutubeVideos.items):
            #Create new video info entry
            self.YoutubeVideos.addToList(URL)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV to PHYT JSON.')
    parser.add_argument('--c', metavar='Channels', nargs='+',
                        help='List of comma-seperated channel names to produce different PHYT JSON')
    args = parser.parse_args()
    args = vars(args)    
    # ChannelName = ytvi.getChannelName("tFfJHfKNSv4")
    # print(ChannelName in args['c'])
    # exit(1)
    
    files = glob.glob("../csv/*.*")
    
    OutputFiles = ["../json/PHYT.json"]
    
        
    file_path = "../json/channels.json"
    if(os.path.exists(file_path)):
        f = open(file_path,'r')
        try:
            URL2CHANNELS = json.load(f)
        except:
            URL2CHANNELS = {}
    
        f.close()
    else:
        URL2CHANNELS = {}
    
    #Seperate file names by channel    
    for filename in files:
        if filename.endswith(".csv"):
            fin = open(filename,'r')
            for line in fin:
                content = line.split(',')
                URL = content[0].split('=')[1].split('&')[0]
                if URL not in URL2CHANNELS:
                    URL2CHANNELS[URL] = ytvi.getChannelName(URL)
                    print("Found channel for " + URL + ": " + URL2CHANNELS[URL])
                    
    with open(file_path,'w') as f:
        json.dump(URL2CHANNELS,f,indent=4, sort_keys=False)
    
    if(args['c'] is not None):
        args['c'].append("")
    else:
        args['c'] = [""]
    
    for YoutubeChannel in args['c']:
        videoIdx = 0
        clipIdx = 0
        videoInfo = {}
        numberOfPenaltyClipByType = {i : 0 for i in range(len(labels))}
        statsByType = {i : c_statistics() for i in range(len(labels))}
        statsWholeDataSet = c_datasetStatistics()

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
                    
                    if(YoutubeChannel is "" and URL2CHANNELS[URL] in args['c']):
                        continue
                    if(YoutubeChannel is not "" and YoutubeChannel != URL2CHANNELS[URL] ):
                        continue

                    if len(content) == 10 : # for PHYT2.0
                        isSeparable = 2     #unspecified
                    else :
                        isSeparable = int(content[10])
                                    
                    flags = c_flags(location,fighting,twoPlayers,caption,isSeparable)

                    penalty = c_penalty(clipIdx,start,end,label,camera,flags)

                    clipIdx = clipIdx + 1
                    
                    #Check if Key is not in dictionary
                    if (URL not in videoInfo):
                        video = c_video(resolution)
                        
                        #Create new video info entry
                        videoInfo[URL]= video.__dict__
                        
                        videoIdx = videoIdx + 1
                    else:
                        videoName = videoInfo[URL]
                    
                     
                    #Update Statistics
                    statsByType[label].update1(clipIdx, resolution,penalty)
                    statsWholeDataSet.update1(URL,clipIdx,resolution,penalty)
                    
                    #Convert flags to dict to be JSON serializable
                    penalty.flags = penalty.flags.__dict__
                    videoInfo[URL]['penalties'].append(penalty.__dict__)
                    
            #Not CSV file
            else:
                pass
        
        #Build Dataset Description
        datasetDescription = {}
        datasetDescription['NumOfVideos'] = videoIdx 
        datasetDescription['NumOfPenaltyClips'] = clipIdx
        datasetDescription["statisticsByLabel"] = { labels[i] : statsByType[i].__dict__ for i in range(len(labels))}
        statsWholeDataSet.YoutubeVideos = statsWholeDataSet.YoutubeVideos.__dict__
        datasetDescription["statisticsDataset"] = statsWholeDataSet.__dict__
        
        out = {}
        out['dataset_description'] = datasetDescription
        out["labels"] = labels
        out["camera_shot_types"] = shotTypes
        out["video_description"] = videoInfo
        
        foutName = "../json/PHYT" + ("_" + YoutubeChannel.replace(" ", "_") if YoutubeChannel is not "" else "") + ".json"
        fout = open(foutName ,'w')  
        json.dump(out,fout,indent=4, sort_keys=False)