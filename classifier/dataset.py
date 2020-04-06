import torch.utils.data
import os
import csv
import numpy as np
import json
import pathlib
import itertools


from sklearn.model_selection import KFold

import torch
import random

kFoldGenerator = None
KFoldLength = 1

target_frame_amount = 15


# have the pairs of joints together to feed in the network
def get_combinations_inter(perps: torch.FloatTensor, victims: torch.FloatTensor):
    num_distances = int((perps.size()[2] - 1) / 3)

    outputs = torch.empty(perps.size()[0], perps.size()[1] * perps.size()[1],
                          perps.size()[2] * 2 + 2 * num_distances - 1)

    for count, perp in enumerate(perps):
        victim = victims[count]
        sizeOfPerp = perp.size()
        nb_position_input = sizeOfPerp[1]
        array_body_index = np.arange(sizeOfPerp[0])
        nb_frames = int((nb_position_input - 1) / 3)

        values = itertools.product(array_body_index, repeat=2)
        nb_players = 2
        # TODO transform this into good comment that describes size of data: sizeOfData = nb_frames * 3 * nb_players + nb_frames + (nb_frames - 1) + nb_players
        output = []
        for x in values:
            joint1 = perp[x[0]]
            joint2 = victim[x[1]]

            x_1 = joint1[::3]
            x_1 = x_1[:-1]
            x_2 = joint2[::3]
            x_2 = x_2[:-1]

            y_1 = joint1[1::3]
            y_2 = joint2[1::3]

            distances = torch.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

            motions = torch.sqrt((x_1[:-1] - x_2[1:]) ** 2 + (y_1[:-1] - y_2[1:]) ** 2)

            # put on the same row for the matrix: joint1, joint2, distances, motions
            iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
            output.append(iteration1)

        output = torch.stack(output, dim=0)
        outputs[count] = output
    return outputs


def get_combinations_intra(p1: torch.FloatTensor):
    num_distances = int((p1.size()[2] - 1) / 3)

    outputs = torch.empty(p1.size()[0], int((p1.size()[1] * p1.size()[1] - p1.size()[1])/2),
                          p1.size()[2] * 2 + 2 * num_distances - 1)

    for count, player1 in enumerate(p1):
        sizeOfPlayer1 = player1.size()
        nb_position_input = sizeOfPlayer1[1]
        array_body_index = np.arange(sizeOfPlayer1[0])
        nb_frames = int((nb_position_input - 1) / 3)

        values = itertools.combinations(array_body_index, 2)
        nb_players = 2
        # TODO transform this into good comment that describes size of data: sizeOfData = nb_frames * 3 * nb_players + nb_frames + (nb_frames - 1) + nb_players
        output = []
        for x in values:
            joint1 = player1[x[0]]
            joint2 = player1[x[1]]

            x_1 = joint1[::3]
            x_1 = x_1[:-1]
            x_2 = joint2[::3]
            x_2 = x_2[:-1]

            y_1 = joint1[1::3]
            y_2 = joint2[1::3]

            distances = torch.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

            motions = torch.sqrt((x_1[:-1] - x_2[1:]) ** 2 + (y_1[:-1] - y_2[1:]) ** 2)

            # put on the same row for the matrix: joint1, joint2, distances, motions
            iteration1 = torch.cat([joint1, joint2, distances, motions], dim=0)
            output.append(iteration1)

        output = torch.stack(output, dim=0)
        outputs[count] = output
    return outputs


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
    def __init__(self, clips_folder: str, hparams,TwoClasses = True,transform=None, cuda=False):
        self.cuda = cuda
        self.hparams = hparams
        self.isInter = self.hparams.inter and not self.hparams.intra
        self.isIntra = not self.hparams.inter and self.hparams.intra
        with open(str(pathlib.Path(__file__).parent.absolute())+'/sets.json') as f:
            sets = json.load(f)
        
        inputFiles, outputFiles = [], []
        for (dirpath, dirnames, filenames) in os.walk(clips_folder):
            for file in filenames:
                if(file.endswith('.json')):
                    inputFiles.append(os.path.join(dirpath, file))
                    outputFiles.append(dirpath + '/label.out')
                    
        self.clips = []
        self.y = []
        tempLabels = []
        
        for input_name,output_name in zip(inputFiles,outputFiles):
            perp, victim = parse_PHYT_clip(input_name)            
            for (p1,p2) in [[perp,victim]]:
                self.clips.append(np.concatenate((p1, p2)))
                
                #Get label
                with open(output_name, 'r') as f:
                    outputlabel = json.load(f)
                outputLabel = outputlabel['penaltyOnly']
                tempLabels.append(outputLabel)
                
                #Go from 3 classes to 2 classes
                if(TwoClasses):
                    if(outputLabel[0] == 1 or outputLabel[1] == 1):
                        outputLabel = [1,0]
                    else:
                        outputLabel = [0,1]
                self.y.append(torch.tensor(outputLabel, dtype=torch.float32))
        
        #Display count of data
        disp = [0 for i in range(len(tempLabels[0]))]
        for y in tempLabels:
            disp[list(map(lambda x: int(x),y)).index(1)] += 1
        for name, num in zip(outputlabel['Definitions']['penaltyOnly'],disp):
            print('For label %s'%name)
            print('\t... %d clips.'%num)
        print('Total number of clips:',sum(disp))
        
        #Seperate test from train
        self.testclips = [self.clips[i] for i in sets['test']]
        self.testy = [self.y[i] for i in sets['test']]
        self.clips = [self.clips[i] for i in sets['train']]
        self.y = [self.y[i] for i in sets['train']]
        
        global kFoldGenerator
        if(kFoldGenerator is None):
            kFoldGenerator = generatorKFold(len(self.clips), k=hparams.kfold)
        train_index, test_index = next(kFoldGenerator)
        self.valclips = [self.clips[i] for i in test_index]
        self.clips = [self.clips[i] for i in train_index]
        self.valy = [self.y[i] for i in test_index]
        self.y = [self.y[i] for i in train_index]
        
        #Print size of datasets
        print('Size of Training Set:',len(self.clips))
        print('Size of Validation Set:',len(self.valclips))
        print('Size of Tests Set:',len(self.testclips))
        
        for clipList, labelList in zip([self.clips,self.valclips,self.testclips],[self.y,self.valy,self.testy]):
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

        self.numOfFrames = (len(self.clips[0][0])-1)//3
        self.numOfJoints = len(self.clips[0])//2

        #Apply get combinations
        for clipList, labelList in zip([self.clips,self.valclips,self.testclips],[self.y,self.valy,self.testy]):
          for clipIdx in range(len(clipList)):
              clips = clipList[clipIdx]
              if(self.hparams.full_gpu):
                labelList[clipIdx] = labelList[clipIdx].cuda()
              p1 = torch.FloatTensor(clips[:int(clips.shape[0]/2),:]).unsqueeze(0)
              p2 = torch.FloatTensor(clips[int(clips.shape[0]/2):,:]).unsqueeze(0)
              
              if(self.isInter):
                input_data_clip_combinations = get_combinations_inter(p1, p2)

                if(self.hparams.full_gpu):
                    input_data_clip_combinations = input_data_clip_combinations.cuda()
                clipList[clipIdx] = input_data_clip_combinations.squeeze(0)
              
              if(self.isIntra):
                input_data_clip_combinations_P1 = get_combinations_intra(p1)
                input_data_clip_combinations_P2 = get_combinations_intra(p2)
                
                if(self.hparams.full_gpu):
                    input_data_clip_combinations_P1 = input_data_clip_combinations_P1.cuda()
                    input_data_clip_combinations_P2 = input_data_clip_combinations_P2.cuda()
                clipList[clipIdx] = (input_data_clip_combinations_P1.squeeze(0),input_data_clip_combinations_P2.squeeze(0))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], self.y[index]


class SBUDataset(torch.utils.data.Dataset):
    def __init__(self, SBU_data_folder: str, hparams, cuda=False, ):
        self.cuda = cuda
        self.hparams = hparams
        self.isInter = self.hparams.inter and not self.hparams.intra
        self.isIntra = not self.hparams.inter and self.hparams.intra

        files = []
        for (dirpath, dirnames, filenames) in os.walk(SBU_data_folder):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".txt")]

        self.clips = []
        self.y = []

        for file_name in files:
            temp = np.concatenate(parse_SBU_clip(file_name))
            self.clips.append(temp)
            arr = [0] * 8
            arr[int(file_name.split('_')[-4]) - 1] = 1
            self.y.append(torch.tensor(arr, dtype=torch.float32))

        global kFoldGenerator

        if (kFoldGenerator is None):
            kFoldGenerator = generatorKFold(len(self.clips), k=hparams.kfold)
        train_index, test_index = next(kFoldGenerator)
        self.valclips = [self.clips[i] for i in test_index]
        self.clips = [self.clips[i] for i in train_index]
        self.valy = [self.y[i] for i in test_index]
        self.y = [self.y[i] for i in train_index]

        self.numOfFrames = (len(self.clips[0][0])-1)//3
        self.numOfJoints = len(self.clips[0])//2

        #Apply get combinations
        for clipList, labelList in zip([self.clips,self.valclips],[self.y,self.valy]):
          for clipIdx in range(len(clipList)):
              clips = clipList[clipIdx]
              if(self.hparams.full_gpu):
                labelList[clipIdx] = labelList[clipIdx].cuda()
              p1 = torch.FloatTensor(clips[:int(clips.shape[0]/2),:]).unsqueeze(0)
              p2 = torch.FloatTensor(clips[int(clips.shape[0]/2):,:]).unsqueeze(0)
              
              if(self.isInter):
                input_data_clip_combinations = get_combinations_inter(p1, p2)

                if(self.hparams.full_gpu):
                    input_data_clip_combinations = input_data_clip_combinations.cuda()
                clipList[clipIdx] = input_data_clip_combinations.squeeze(0)
              
              if(self.isIntra):
                input_data_clip_combinations_P1 = get_combinations_intra(p1)
                input_data_clip_combinations_P2 = get_combinations_intra(p2)
                
                if(self.hparams.full_gpu):
                    input_data_clip_combinations_P1 = input_data_clip_combinations_P1.cuda()
                    input_data_clip_combinations_P2 = input_data_clip_combinations_P2.cuda()
                clipList[clipIdx] = (input_data_clip_combinations_P1.squeeze(0),input_data_clip_combinations_P2.squeeze(0))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], self.y[index]


if __name__ == "__main__":
    dataset = SBUDataset('SBUDataset/')
