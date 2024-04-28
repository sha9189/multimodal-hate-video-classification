import os
import torch
import pickle
import torch.utils.data as data
from utils.utils import load_config
import pickle
import numpy as np
import sys


config = load_config('configs/configs.yaml')


class MultiModalData(data.Dataset):
    def __init__(self, folders, labels):
        self.labels = labels
        self.folders = folders
        self.load_saved_features(feature_extractor=config["UNIMODAL_FEATURE_EXTRACTOR"])

    def __len__(self):
        return len(self.folders)

    def read_text(self,selected_folder):
        return (torch.tensor(self.textData[selected_folder]), torch.tensor(self.vidData[selected_folder]), torch.tensor(self.audData[selected_folder]))

    def __getitem__(self, index):
        folder = self.folders[index]
        try:
            X_text, X_vid, X_audio = self.read_text(folder)
            y = torch.LongTensor([self.labels[index]])
        except:
            with open("Exceptions.txt", "a") as f:
                f.write(f"Missing data for folder {folder}:\n")
            return None  # Skip this sample or handle differently
        return X_text, X_vid, X_audio, y
    
    
    def load_saved_features(self, feature_extractor = "ALL"):
        self.textData, self.vidData, self.audData = {}, {}, {}
        if feature_extractor == "ALL":
            # BERT
            with open(config["PICKLE_FOLDER"]+'all_rawBERTembedding.p','rb') as fp:
                self.textData = pickle.load(fp)
            #VIT
            allVidList = [name[:-6] for name in os.listdir(config["VIT_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
            for i in allVidList:
                with open(f"{config['VIT_FOLDER']}/{i}_vit.p", 'rb') as fp:
                    self.vidData[i] = np.array(pickle.load(fp))
            # MFCC
            # allAudList = [name[:-7] for name in os.listdir(config["MFCC_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
            for i in allVidList:
                with open(f"{config['MFCC_FOLDER']}/{i}_mfcc.p", 'rb') as fp:
                    self.audData[i] = np.array(pickle.load(fp))
        else:
            print("Feature Extractor not defined.. Exiting")
            sys.exit(1)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch)







# # BERT features
# with open(config["PICKLE_FOLDER"]+'all_rawBERTembedding.p','rb') as fp:
#     textData = pickle.load(fp)
# """--------------------------------------------------------------------------------------------------------------------------------------"""

# # MFCC features
# mfcc_folder = config['MFCC_FOLDER']
# mfcc_feature_files = [f for f in os.listdir(mfcc_folder) if f.endswith('_mfcc.p') and not f.startswith('.')]
# audData = {}
# for file_name in mfcc_feature_files:
#     if 'non_hate' in file_name:
#         basename = file_name[:-16]
#     elif 'hate' in file_name:
#         basename = file_name[:-12]
#     filepath = os.path.join(mfcc_folder, file_name)
#     with open(filepath, 'rb') as fp:
#         audData[basename] = np.array(pickle.load(fp))


# """--------------------------------------------------------------------------------------------------------------------------------------"""
# # ViT features
# vit_folder = config['VIT_FOLDER']
# vit_feature_files = [f for f in os.listdir(vit_folder) if f.endswith('_vit.p') and not f.startswith('.')]
# vidData = {}
# for file_name in vit_feature_files:
#     basename = file_name[:-7]  # adjusted to exclude '_vit.p'
#     filepath = os.path.join(vit_folder, file_name)
#     with open(filepath, 'rb') as fp:
#         vidData[basename] = np.array(pickle.load(fp))



