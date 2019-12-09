import sys

def main():

    f = open("..\csv\youtubePenaltiesList.csv",'a')

    #Check number of arguments is 3
    if(len(sys.argv) != 11):
        print(len(sys.argv))
        print("Wrong number of Args. Must be "
              "<Youtube URL> "
              "<video resolution> "
              "<start min:sec> "
              "<end min:sec> "
              "<label: 0=Trip 1=CC 2=none> "
              "<camera shot type 0=short 1=med 2=long> "
              "<Middle rink 0=Middle 1=Board> "
              "<has fighting 0=false 1=true> "
              "<only two players 0=false 1=true> "
              "<has caption 0=false 1=true"
              "is separable 0=false 1=true")
        exit(1)

    URL = sys.argv[1]
    res = sys.argv[2]
    startTime = sys.argv[3].split(':')
    endTime = sys.argv[4].split(':')
    label = sys.argv[5]
    shot_type = sys.argv[6]
    middle_rink = sys.argv[7]
    has_fighting = sys.argv[8]
    two_players = sys.argv[9]
    has_caption = sys.argv[10]
    is_separable = sys.argv[11]

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
    f.write(
        URL +","+
        res + "," +
        str(startTime)+","+
        str(endTime)+","+
        label + "," +
        shot_type + "," +
        middle_rink + "," +
        has_fighting + "," +
        two_players + "," +
        has_caption  + "," +
        is_separable +
        "\n")

if __name__ == "__main__":
    main()