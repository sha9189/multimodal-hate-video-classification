from utils.utils import load_config
import numpy as np
import pickle
import os

config = load_config('configs/configs.yaml')


def read_audio_data():
    """Load all features for the defined AUDIO_FEATURE_EXTRACTOR
    Output:
        modality_data: dict with key as video name and value as 
            the feature for that video extracted from model 
            defined in config.AUDIO_FEATURE_EXTRACTOR 
    """
    modality_data = {}
    if config["AUDIO_FEATURE_EXTRACTOR"] == "MFCC":
        allAudList = [name[:-7] for name in os.listdir(config["MFCC_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
        for i in allAudList:
            with open(f"{config['MFCC_FOLDER']}/{i}_mfcc.p", 'rb') as fp:
                modality_data[i] = np.array(pickle.load(fp))
    return modality_data


def read_text_data():
    """Load all features for the defined TEXT_FEATURE_EXTRACTOR
    Output:
        modality_data: dict with key as video name and value as 
            the feature for that video extracted from model 
            defined in config.TEXT_FEATURE_EXTRACTOR 
    
    """
    modality_data = {}
    if config["TEXT_FEATURE_EXTRACTOR"] == "BERT":
        with open(config["PICKLE_FOLDER"]+'all_rawBERTembedding.p','rb') as fp:
            modality_data = pickle.load(fp)
    return modality_data


def read_video_data():
    """Load all features for the defined VIDEO_FEATURE_EXTRACTOR
    Output:
        modality_data: dict with key as video name and value as 
            the feature for that video extracted from model 
            defined in config.VIDEO_FEATURE_EXTRACTOR 
    
    """
    modality_data = {}
    allVidList = [name[:-6] for name in os.listdir(config["VIT_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
    for i in allVidList:
        with open(f"{config['VIT_FOLDER']}/{i}_vit.p", 'rb') as fp:
            modality_data[i] = np.array(pickle.load(fp))
    return modality_data


def read_data_for_modality(modality):
    """Function loads the features for the given modality
    Input: 
        modality: AUD/VID/TEXT
    Output:
        modality_data: dict with key as the video name and value as the 
            features for that video
    """
    if modality == 'AUD':
        modality_data = read_audio_data()
    elif modality == 'TEXT':
        modality_data = read_text_data()
    elif modality == 'VID':
        modality_data = read_video_data()
    return modality_data


