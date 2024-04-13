# %%
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pickle
from sklearn.metrics import *
import numpy as np
import torch.nn as nn
from utils.utils import fix_the_random
from models.vision_models import LSTM
from utils.utils import load_config
from training.evaluation import evalMetric
from training.train_vision import train, validation
from datasets.custom_datasets import Dataset_3DCNN, collate_fn


# %%

config = load_config('configs/config.yaml')

fix_the_random(2021)







# %%
# ROOT_FOLDER = "../"
ROOT_FOLDER = config["ROOT_FOLDER"]
DATASET_FOLDER = config["DATASET_FOLDER"]
VIT_FOLDER = config["VIT_FOLDER"]


# %%



# input_size = 768
# sequence_length = 100
# hidden_size = 128
# num_layers = 2


# training parameters
k = 2            # number of target category
epochs = config["EPOCHS"]
batch_size = config["BATCH_SIZE"]
learning_rate = config["LEARNING_RATE"]
log_interval = config["LOG_INTERVAL"]
num_workers = config["NUM_WORKERS"]
pin_memory = config["PIN_MEMORY"]


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {
    'batch_size': batch_size, 
    'shuffle': True, 
    'num_workers': num_workers, 
    'pin_memory': pin_memory} if use_cuda else {'batch_size': batch_size, 'shuffle': True}
valParams = {
    'batch_size': batch_size, 
    'shuffle': False, 
    'num_workers': num_workers, 
    'pin_memory': pin_memory} if use_cuda else {'batch_size': batch_size, 'shuffle': False}


with open(DATASET_FOLDER+'allFoldDetails.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)


allF = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']


finalOutputAccrossFold ={}

for fold in allF:
    # train, test split
    train_list, train_label= allDataAnnotation[fold]['train']
    val_list, val_label  =  allDataAnnotation[fold]['val']
    test_list, test_label  =  allDataAnnotation[fold]['test']


    train_set, valid_set , test_set = Dataset_3DCNN(train_list, train_label), Dataset_3DCNN(val_list, val_label), Dataset_3DCNN(test_list, test_label)
    train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **params)
    test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **valParams)
    valid_loader = data.DataLoader(valid_set, collate_fn = collate_fn, **valParams)

    comb = LSTM().to(device)

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        comb = nn.DataParallel(comb)

    optimizer = torch.optim.Adam(comb.parameters(), lr=learning_rate)   # optimize all cnn parameters

    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    validFinalValue = None
    testFinalValue = None
    finalScoreAcc =0
    prediction  = None

    # start training
    for epoch in range(epochs):
        # train, test model
        train_losses, train_scores = train(log_interval, comb, device, train_loader, optimizer, epoch)
        test_loss, test_scores, veTest_pred = validation(comb, device, optimizer, test_loader)
        test_loss1, test_scores1, veValid_pred = validation(comb, device, optimizer, valid_loader)
        if (test_scores1['mF1Score']>finalScoreAcc):
            finalScoreAcc = test_scores1['mF1Score']
            validFinalValue = test_scores1
            testFinalValue = test_scores
            prediction = {'test_list': test_list , 'test_label': test_label, 'test_pred': veTest_pred}

        # save results
        epoch_train_losses.append(train_losses)
        epoch_train_scores.append(list(x['accuracy'] for x in train_scores))
        epoch_test_losses.append(test_loss)
        epoch_test_scores.append(test_scores['accuracy'])


        # save all train test results
        A = np.array(epoch_train_losses)
        B = np.array(epoch_train_scores)
        C = np.array(epoch_test_losses)
        D = np.array(epoch_test_scores)
    finalOutputAccrossFold[fold] = {'validation':validFinalValue, 'test': testFinalValue, 'test_prediction': prediction}
        

with open('foldWiseRes_lstmVision.p', 'wb') as fp:
    pickle.dump(finalOutputAccrossFold,fp)
        


# %%
# allValueDict ={}
# for fold in allF:
#     for val in finalOutputAccrossFold[fold]['test']:
#         try:
#             allValueDict[val].append(finalOutputAccrossFold[fold]['test'][val])
#         except:
#             allValueDict[val]=[finalOutputAccrossFold[fold]['test'][val]]



# import numpy as np
# for i in allValueDict:
#     print(f"{i} : Mean {np.mean(allValueDict[i])}  STD: {np.std(allValueDict[i])}")



# %%



