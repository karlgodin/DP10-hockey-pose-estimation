import torch.utils.data
import os
import csv
import numpy as np
import json


from sklearn.model_selection import KFold

import torch
import random

kFoldGenerator = None
KFoldLength = 1

target_frame_amount = 15

def generatorKFold(sizeOfElem,k=KFoldLength):
    indexes = [i for i in range(sizeOfElem)]
    random.shuffle(indexes)
    if(k == 1):
        yield indexes, []
    else:
        kf = KFold(n_splits=KFoldLength)
        for train_index, test_index in kf.split(indexes):
            train = [indexes[i] for i in train_index]
            test = [indexes[i] for i in test_index]
            yield train, test


def parse_PHYT_clip(file_name: str):
    with open(file_name) as json_file:
        data = json.load(json_file)

        perp_frames = []
        victim_frames = []

        for frame in data:
            perp = frame["perp"]
            victim = frame["victim"]

            # TODO: add boolean to keep or remove confidence parameter
            # del perp[2::3]
            # del victim[2::3]

            perp_frames.append(perp)
            victim_frames.append(victim)

        perp_frames = np.array(perp_frames, dtype=np.float32)
        victim_frames = np.array(victim_frames, dtype=np.float32)


        num_frames = perp_frames.shape[0]

        # Take 15 middle frames if at least target frame amount frames
        if num_frames >= target_frame_amount:
            perp_frames = perp_frames[int(num_frames / 2) - int(target_frame_amount / 2):, :]
            perp_frames = perp_frames[:target_frame_amount, :]

            victim_frames = victim_frames[int(num_frames / 2) - int(target_frame_amount / 2):, :]
            victim_frames = victim_frames[:target_frame_amount, :]
        else:  # Add zeros for final frames if less than 15 frames
            perp_frames = np.vstack(
                (perp_frames, np.zeros((target_frame_amount - num_frames, 45), dtype=np.float32)))
            victim_frames = np.vstack(
                (victim_frames, np.zeros((target_frame_amount - num_frames, 45), dtype=np.float32)))

        # Transpose dataset
        perp_frames = get_joints(perp_frames, 25)
        victim_frames = get_joints(victim_frames, 25)

        return perp_frames, victim_frames


def parse_SBU_clip(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        frame_list = list(csv_reader)
        frame_list = np.array(frame_list, dtype=np.float32)
        num_frames = frame_list.shape[0]
        perp_frames = frame_list[:,1:46]
        victim_frames = frame_list[:,46:]

        #Take 15 middle frames if at least 15 frames
        if num_frames >= target_frame_amount:
            perp_frames = perp_frames[int(num_frames/2)- int(target_frame_amount/2):,:]
            perp_frames = perp_frames[:target_frame_amount,:]

            victim_frames = victim_frames[int(num_frames / 2) - int(target_frame_amount/2):, :]
            victim_frames = victim_frames[:target_frame_amount, :]
        else: # Add zeros for final frames if less than 15 frames
            perp_frames = np.vstack((perp_frames, np.zeros((target_frame_amount-num_frames, 45), dtype=np.float32)))
            victim_frames = np.vstack((victim_frames, np.zeros((target_frame_amount-num_frames, 45), dtype=np.float32)))


        perp_frames = get_joints(perp_frames, 15)
        victim_frames = get_joints(victim_frames, 15)

        return perp_frames, victim_frames


"""
Returns a 2D numpy array formatted according to the Interaction Recognition paper
Each row contains temporal information for one joint 

Input array:
[[ x_00, y_00, c_00, ... , x_n0, y_n0, c_n0 ],
 [ x_01, y_01, c_01, ... , x_n1, y_n1, c_n1 ],
 [                   ...                    ],
 [ x_0t, y_0t, c_0t, ... , x_nt, y_nt, c_nt ]]

Output array:
[[ x_00, y_00, c_00, ... , x_0t, y_0t, c_0t, 0 ],
 [ x_10, y_10, c_10, ... , x_1t, y_1t, c_1t, 1 ],
 [                   ...                       ],
 [ x_n0, y_n0, c_n0, ... , x_nt, y_nt, c_nt, 24 ]]
 """
def get_joints(player_frames : np.ndarray, nb_joints):
    # Obtain an array for each data point type
    player_x = player_frames[:, 0::3]
    player_y = player_frames[:, 1::3]
    player_c = player_frames[:, 2::3]

    # Create an empty array with appropriate size
    player_frames = np.empty((player_frames.shape[0] * 3, nb_joints), dtype=player_frames.dtype)

    # Merge data points into a single 2D array
    player_frames[0::3] = player_x
    player_frames[1::3] = player_y
    player_frames[2::3] = player_c

    # Add body part information to each row and transpose the array such that each row
    # contains x_n0,y_n0,c_n0, ... x_nt, y_nt, c_nt where n is the row number and t is the frame number
    player_frames = np.append(player_frames, np.float32(np.reshape(np.arange(nb_joints), (1, nb_joints))), axis=0)

    return np.transpose(player_frames)


class PHYTDataset(torch.utils.data.Dataset):
    def __init__(self, clips_folder: str, hparams,transform=None, cuda=False):
        self.cuda = cuda
        self.hparams = hparams

        inputFiles, outputFiles = [], []
        for (dirpath, dirnames, filenames) in os.walk(clips_folder):
            for file in filenames:
                if(file.endswith('.json')):
                    inputFiles.append(os.path.join(dirpath, file))
                    
                if(file == 'label.out'):
                    outputFiles.append(os.path.join(dirpath, file))
                    
        self.clips = []
        self.y = []

        
        for input_name,output_name in zip(inputFiles,outputFiles):
            perp, victim = parse_PHYT_clip(input_name)            
            for (p1,p2) in [[perp,victim]]:
                self.clips.append(np.concatenate((p1, p2)))
                
                #Get label
                with open(output_name, 'r') as f:
                    outputlabel = json.load(f)
                self.y.append(torch.tensor(outputlabel['penaltyOnly'], dtype=torch.float32))
        
        global kFoldGenerator
        if(kFoldGenerator is None):
            kFoldGenerator = generatorKFold(len(self.clips), k=hparams.kfold)
        train_index, test_index = next(kFoldGenerator)
        self.valclips = [self.clips[i] for i in test_index]
        self.clips = [self.clips[i] for i in train_index]
        self.valy = [self.y[i] for i in test_index]
        self.y = [self.y[i] for i in train_index]
        
        
        for clipList, labelList in zip([self.clips,self.valclips],[self.y,self.valy]):
            #Data Augmentation. Changing Perp and Victim Place
            if(self.hparams.changeOrder):
                for clipIdx in range(len(clipList)):
                    clips = clipList[clipIdx]
                    label = labelList[clipIdx]
                    perp = clips[:int(clips.shape[0]/2),:]
                    victim = clips[int(clips.shape[0]/2):,:]
                    
                    clipList.append(np.concatenate((victim, perp)))
                    labelList.append(label)
            
            if(self.hparams.randomJointOrder != 0):
                for clipIdx in range(len(clipList)):
                    clips = clipList[clipIdx]
                    label = labelList[clipIdx]
                    p1 = clips[:int(clips.shape[0]/2),:]
                    p2 = clips[int(clips.shape[0]/2):,:]
                    
                    for _ in range(self.hparams.randomJointOrder):
                        np.random.shuffle(p1)
                        np.random.shuffle(p2)
                        
                        clipList.append(np.concatenate((p1, p2)))
                        labelList.append(label)
                        

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], self.y[index]


class SBUDataset(torch.utils.data.Dataset):
    def __init__(self, SBU_data_folder: str, hparams, cuda=False, ):
        self.cuda = cuda

        files = []
        for (dirpath, dirnames, filenames) in os.walk(SBU_data_folder):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".txt")]

        self.clips = []
        self.y = []

        for file_name in files:
            temp = np.concatenate(parse_SBU_clip(file_name))
            self.clips.append(temp)
            arr = [0] * 8
            arr[int(file_name.split('\\')[1].split('_')[1]) - 1] = 1
            self.y.append(torch.tensor(arr, dtype=torch.float32))

        global kFoldGenerator

        if (kFoldGenerator is None):
            kFoldGenerator = generatorKFold(len(self.clips), k=hparams.kfold)
        train_index, test_index = next(kFoldGenerator)
        self.valclips = [self.clips[i] for i in test_index]
        self.clips = [self.clips[i] for i in train_index]
        self.valy = [self.y[i] for i in test_index]
        self.y = [self.y[i] for i in train_index]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], self.y[index]


if __name__ == "__main__":
    dataset = SBUDataset('SBUDataset/')
