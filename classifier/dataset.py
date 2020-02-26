import torch.utils.data
import os
from classifier.joints import parse_clip

class ClipsDataset(torch.utils.data.Dataset):
    def __init__(self, clips_folder: str, transform=None, cuda=False):
        self.cuda = cuda


        files = []
        for (dirpath, dirnames, filenames) in os.walk(clips_folder):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".json")]


        self.clips = []

        for file_name in files:
            perp, victim = parse_clip(file_name)

            self.clips.append({'perp': perp, 'victim': victim})

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        return self.clips[index]
"""
        # Load X data
        with open(x_data_filename, 'rb') as f:
            if self.cuda:
                self.x = pickle.load(f).cuda()
            else:
                self.x = pickle.load(f)

        # self.x = x_data
        self.length = len(self.x)

        # Load Y data
        if y_data_filename is None:
            # If no Y file supported, assume test and fill with zeroes
            if self.cuda:
                self.y = torch.cuda.LongTensor(self.length)
            else:
                self.y = torch.zeros(self.length)
        else:
            # Load Y data
            with open(y_data_filename, 'rb') as f:
                if self.cuda:
                    self.y = pickle.load(f).cuda()
                else:
                    self.y = pickle.load(f)

        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.transform:
            sample = (self.transform(self.x[index]), self.y[index])
        else:
            sample = (self.x[index], self.y[index])
        return sample"""
