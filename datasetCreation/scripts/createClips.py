#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 18:42:12 2020

@author: svosburg
"""

import cv2
import json
import os.path

if __name__ == "__main__":
    VideoPath = "../Videos/"
    ClipPath = "../Clips/"
    PHYTfiles = ['PHYT.json', 'PHYT_NHL.json']    

    for PHYTfile in PHYTfiles:
	
	    #Load PHYT dataset
	    with open("../json/" + PHYTfile) as json_file:
		phyt = json.load(json_file)
	    
	    #Loop through all penalties
	    for videoID in phyt['video_description']:
		for penalty in phyt['video_description'][videoID]['penalties']:
			if(os.path.isfile(VideoPath + videoID + ".mp4")):
			    clip_ID = penalty['clip_ID']
			    start_time = penalty['start_time']
			    end_time = penalty['end_time']
                
			    #Load Video
			    cap = cv2.VideoCapture(VideoPath + videoID + ".mp4")
			    
			    #Get FrameRate and size
			    fps = cap.get(cv2.CAP_PROP_FPS)
			    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
			    
			    #Get start and end Frames number
			    startFrame = int(start_time * fps)
			    endFrame = int(end_time * fps)
			    
			    #Create output
			    fourcc = cv2.VideoWriter_fourcc(*'XVID')
			    print(ClipPath+PHYTfile.split('.')[0]+'/_'+str(clip_ID)+".mp4")
			    out = cv2.VideoWriter(ClipPath+PHYTfile.split('.')[0]+'/_'+str(clip_ID)+".mp4",fourcc, fps, (width,height))
			    
			    #set frame 
			    cap.set(cv2.CAP_PROP_POS_FRAMES,startFrame)
			    
			    #Copy over frame from cap to out
			    print("%d %d %d"%(clip_ID,start_time,end_time))
			    for _ in range(endFrame - startFrame):
				ret, frame = cap.read()
				if not ret:
				    break
				out.write(frame)
			    
			    #Release videos
			    cap.release()
			    out.release()
		    
    #Close peacefully
    cv2.destroyAllWindows()
