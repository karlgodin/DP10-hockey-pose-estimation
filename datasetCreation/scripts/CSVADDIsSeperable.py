import glob 
import webbrowser

TYPES = {
    0: "Tripping",
    1: "CrossChecking",
    2: "NoPenalty"
}

flagWanted = "isSeperable"

files = glob.glob("../csv/*.csv")

for filename in files:
    with open(filename,'r') as f:
        file_Contents = f.read()
        
    entries = file_Contents.split('\n')
    
    #Edit filename
    newFileName = filename.split(".csv")[0]+"_Test.shawn"
    
    with open(newFileName,'a') as f: 
        for entry in entries:
            f.write(entry)
            isSeperable = 0
            values = entry.split(',')
            
            if(len(values) == 10):
                url = values[0].split("&")[0]+"&t="+values[2]+"s"
                webbrowser.open(url,new=0)
                print(url + " is an example of " + TYPES[int(values[4])])
                
                
                isSeperableInput = input("Is it seperable? y/n")
                while(1):
                    if(isSeperableInput == 'y' or isSeperableInput == 'Y'):
                        isSeperable = 1
                        break
                    elif(isSeperableInput == 'n' or isSeperableInput == 'N'):
                        isSeperable = 0
                        break
                    else:
                        isSeperableInput = input("Not a valid input. Is it seperable? y/n")
                        
                print(isSeperable)
                f.write("," + str(isSeperable))
        
            f.write('\n')