import torch
import torch.utils.data as data
from utils.utils import load_config
import os
import numpy as np
import pickle
import sys

config = load_config('configs/configs.yaml')

class Dataset_3DCNN(data.Dataset):
    "PyTorch dataset that stores the video names and labels and returns VIT features for each video"
    
    def __init__(self, folders, labels):
        "Initialization"
        self.labels = labels
        self.folders = folders
        self.load_saved_features(feature_extractor=config["UNIMODAL_FEATURE_EXTRACTOR"])

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_text(self,selected_folder):
        return torch.tensor(self.inputDataFeatures[selected_folder])
        
    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]
        try:
            # Load data
            X_text = self.read_text(folder)
            y = torch.LongTensor([self.labels[index]])         # (labels) LongTensor are for int64 instead of FloatTensor
        except:
            with open("Exceptions.txt","a") as f:
                f.write("{}\n".format(folder))
            return None
        return X_text, y
    
    def load_saved_features(self, feature_extractor = "VIT"):  
        "Function loads saved features of video extracted out of a pre-trained model" 
        self.inputDataFeatures = {} 
        if feature_extractor == "VIT":
            allVidList = [name[:-6] for name in os.listdir(config["VIT_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
            for i in allVidList:
                with open(f"{config['VIT_FOLDER']}/{i}_vit.p", 'rb') as fp:
                    self.inputDataFeatures[i] = np.array(pickle.load(fp))
        elif feature_extractor == "BERT":
            with open(config["PICKLE_FOLDER"]+'all_rawBERTembedding.p','rb') as fp:
                self.inputDataFeatures = pickle.load(fp)
        elif feature_extractor == 'MFCC':
            allAudList = []
            for name in os.listdir(config['MFCC_FOLDER']):
                if name.endswith('_mfcc.p'):
                    if 'non_hate' in name:
                        basename = name[:-16]
                        fullname = basename+'_non_hate_mfcc.p'
                    elif 'hate' in name:
                        basename = name[:-12]
                        fullname = basename+'_hate_mfcc.p'
                    else:
                        continue
                    allAudList.append([basename, fullname])
            for basename, fullname in allAudList:
                filepath = f"{config['MFCC_FOLDER']}{fullname}"
                with open(filepath, 'rb') as fp:
                    self.inputDataFeatures[basename] = np.array(pickle.load(fp))
        else:
            print("Feature Extractor not defined.. Exiting")
            sys.exit(1)

    
def collate_fn(batch):
    """Function to remove missing datapoints from the batch(for the last batch)"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)