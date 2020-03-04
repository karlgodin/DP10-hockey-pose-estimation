import torch.utils.data
import os
from classifier.joints import parse_clip
import csv
import numpy as np
import json

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

        # Transpose dataset
        perp_frames = get_joints(np.array(perp_frames, dtype=np.float32), 25)
        victim_frames = get_joints(np.array(victim_frames, dtype=np.float32), 25)

        return perp_frames, victim_frames


def parse_SBU_clip(file_name):
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        frame_list = list(csv_reader)
        frame_list = np.array(frame_list, dtype=np.float32)
        perp_frames = frame_list[:,1:46]
        victim_frames = frame_list[:,46:]

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
    def __init__(self, clips_folder: str, transform=None, cuda=False):
        self.cuda = cuda

        files = []
        for (dirpath, dirnames, filenames) in os.walk(clips_folder):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".json")]

        self.clips = []

        for file_name in files:
            perp, victim = parse_PHYT_clip(file_name)

            self.clips.append({'perp': perp, 'victim': victim})

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]


class SBUDataset(torch.utils.data.Dataset):
    def __init__(self, SBU_data_folder: str, cuda=False):
        self.cuda = cuda

        files = []
        for (dirpath, dirnames, filenames) in os.walk(SBU_data_folder):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".txt")]

        self.clips = []
        self.y = []

        for file_name in files:
            self.clips.append(parse_SBU_clip(file_name))
            arr = [0] * 8
            arr[int(file_name.split('_')[1]) - 1] = 1
            self.y.append(file_name)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index], self.y[index]


if __name__ == "__main__":
    dataset = SBUDataset('SBUDataset/')
