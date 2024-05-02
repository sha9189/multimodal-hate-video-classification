import torch
from torch.utils.data import Dataset
from data_preprocessing.multimodal_helpers import read_data_for_modality
from utils.utils import load_config

config = load_config('configs/configs.yaml')

class MultimodalDataset(Dataset):
    """
    Class for MultiModal Model
    """
    def __init__(self, folders, labels, modalities):
        """
        Inputs:
            folders: list of video names
            labels: list of labels
            modalities: list of modalities to use, example: ["AUD", "TEXT", "VIDEO"]
        """
        self.folders = folders
        self.labels = labels 
        self.load_modalities_data(modalities)       

    def load_modalities_data(self, modalities):
        """Function loads the saved features for each modality and saves it in a dict
            Outputs:
                self.modalities_data: dict with key as modality(AUD/VID/TEXT) 
                    and value as (dict w key as video name and value as saved feature for that video 
                        and modality given in config file)
        """
        self.modalities_data = {}
        for modality in modalities:
            self.modalities_data[modality] = read_data_for_modality(modality)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """For the given index(video name), get X and y as output
        Outputs:
            X: dict with key as modality(AUD/VID/TEXT) and value is 
                the feature for that modality
            y: label for that video
        """
        folder = self.folders[idx]
        sample = {}

        for modality in self.modalities_data.keys():
            sample[modality + '_features'] = self.get_features_for_modality(folder, modality)
        y = torch.LongTensor([self.labels[idx]])

        return sample, y

    def get_features_for_modality(self, folder, modality):
        """Return features for one video and one modality out of AUD/VID/TEXT"""
        return torch.tensor(self.modalities_data[modality][folder])   