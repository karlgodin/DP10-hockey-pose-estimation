import sys

f = open("youtubePenaltiesList.csv",'a')

#Check number of arguments is 3
if(len(sys.argv) != 5):
    print(len(sys.argv))
    print("Wrong number of Args. Must be <Youtube URL> <start min:sec> <end min:sec> <Trip/CC/none>")
    exit(1)
    
URL = sys.argv[1]
startTime = sys.argv[2].split(':')
endTime = sys.argv[3].split(':')
penalty = sys.argv[4]

#Process Start and End Times
flag = 0
if(len(startTime) != 2):
    print("Start Time must be in format min:sec")
    flag = 1
if(len(endTime) != 2):
    print("End Time must be in format min:sec")
    flag = 1
if(flag == 1):
    exit(1)
startTime = 60*int(startTime[0]) + int(startTime[1])
endTime = 60*int(endTime[0]) + int(endTime[1])

#Write to CSV
f.write(URL +","+str(startTime)+","+str(endTime)+","+penalty"\n")