from __future__ import unicode_literals
import json
import sys
import youtube_dl
import os



if __name__ == "__main__" :
    if(len(sys.argv) > 2):
        print("Usage: python dlVideosPHYT.py <filepath to PHYT.json>")
        exit(1)
    elif(len(sys.argv) == 2):
        json_filepath = sys.argv[1]
    else:
        json_filepath = "../json/PHYT.json"
        
    
    #Fetch JSON data
    json_file = open(json_filepath)
    data = json.load(json_file)
    
    #Fetch number of videos for stats display
    numOfVids = data['dataset_description']['NumOfVideos']
    
    for i,keys in enumerate(data['video_ID'].keys()):
        url = 'https://www.youtube.com/watch?v=' + keys
        print("Downloading video #" + str(i+1) + " of " + str(numOfVids))
        os.system('youtube-dl.exe ' + url +' -o ../Videos/%(id)s.%(ext)s -f mp4')
    
    
    
    