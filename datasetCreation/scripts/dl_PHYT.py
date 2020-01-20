from __future__ import unicode_literals
import json
import sys
import youtube_dl #pip install youtube_dl
import os
from sys import platform as _platform

def checkOS():
    if _platform == "linux" or _platform == "linux2":
       return "LINUX"
    elif _platform == "darwin":
       return "MACOS"
    elif _platform == "win32" or _platform == "win64":
       return "WINDOWS"
    else:
        raise("OS not supported")

if __name__ == "__main__" :
    if(len(sys.argv) > 2):
        print("Usage: python dlVideosPHYT.py <filepath to PHYT.json>")
        exit(1)
    elif(len(sys.argv) == 2):
        json_filepath = sys.argv[1]
    else:
        json_filepath = "../json/PHYT.json"
        
    #Check OS
    OS = checkOS()
    
    #Fetch JSON data
    json_file = open(json_filepath)
    data = json.load(json_file)
    
    #Fetch number of videos for stats display
    numOfVids = data['dataset_description']['statisticsDataset']['YoutubeVideos']['count']
    
    for i,keys in enumerate(data['video_description'].keys()):
        url = 'https://www.youtube.com/watch?v=' + keys
        print("Downloading video #" + str(i+1) + " of " + str(numOfVids))
        if(OS == "WINDOWS"):
            os.system('youtube-dl.exe ' + url +' -o ../Videos/%(id)s.%(ext)s -f mp4')
        elif(OS == "LINUX"):
            command = 'youtube-dl ' + url +' -o ../Videos/%(id)s.%(ext)s -f mp4'
            print(command)
            #os.system('youtube-dl' + url +' -o ../Videos/%(id)s.%(ext)s -f mp4')
        
        