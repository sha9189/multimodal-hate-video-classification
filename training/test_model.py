from utils.utils import load_config, fix_the_random
from data_preprocessing.custom_datasets import Dataset_3DCNN, collate_fn
import torch
import torch.utils.data as data
import torch.nn as nn
from training.train_vision import train, validation
from training.evaluation import best_performance_using_kfold_results, aggregate_performance_by_epoch
import numpy as np
from tqdm import tqdm

config = load_config('configs/configs.yaml')
fix_the_random(2021)

def test_model(
        model_cls,
        dataset_cls,
        epochs:int,
        optimizer_name,
        allDataAnnotation: dict
):
    """Function to run k-fold CV on the model and return its performance metrics
    Inputs:
        model_cls: model class to test
        dataloaders: tuple of (train_dataloader, val_dataloader, test_dataloader) --remove
        epochs: epochs
        loss_fn: loss_fn
        optimizer_name: name of optimizer like 'Adam'
        allDataAnnotation: dict of fold_num and corresponding train, val, test indexes
    Outputs:
        performance_by_epoch: average model performance by epoch 
        best_mean_metrics: avg over scores of best model in each fold on test set
        best_std_metrics: std of scores of best model in each fold on test set
    """

    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    trainParams = {
        'batch_size': config["BATCH_SIZE"], 
        'shuffle': True, 
        'num_workers': config["NUM_WORKERS"], 
        'pin_memory': config["PIN_MEMORY"]} if use_cuda else {'batch_size': config["BATCH_SIZE"], 'shuffle': True}
    valParams = {
        'batch_size': config["BATCH_SIZE"], 
        'shuffle': False, 
        'num_workers': config["NUM_WORKERS"], 
        'pin_memory': config["PIN_MEMORY"]} if use_cuda else {'batch_size': config["BATCH_SIZE"], 'shuffle': False}
    testParams = valParams
    
    finalOutputAccrossFold = {}

    for fold in tqdm(allDataAnnotation.keys()):
    # train, test split
        train_list, train_label= allDataAnnotation[fold]['train']
        val_list, val_label  =  allDataAnnotation[fold]['val']
        test_list, test_label  =  allDataAnnotation[fold]['test']

        train_set, valid_set , test_set = dataset_cls(train_list, train_label), dataset_cls(val_list, val_label), dataset_cls(test_list, test_label)
        train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **trainParams)
        test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **valParams)
        valid_loader = data.DataLoader(valid_set, collate_fn = collate_fn, **testParams)
        
        model = model_cls().to(device)

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])   # optimize all cnn parameters
        train_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.41, 0.59])).to(device)
        test_criterion = nn.CrossEntropyLoss(reduction='sum')

        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []
        epoch_val_losses = []
        epoch_val_scores = []

        validFinalValue = None
        testFinalValue = None
        finalScoreAcc = 0
        prediction  = None

        # start training
        for epoch in range(epochs):
            # train, test model
            train_loss, train_scores = train(
                config["LOG_INTERVAL"], 
                model, 
                device, 
                train_loader, 
                train_criterion,
                optimizer, 
                epoch
            )
            val_loss, val_scores, veValid_pred = validation(
                model, device, test_criterion, valid_loader, dataset_name="Val"
            )
            test_loss, test_scores, veTest_pred = validation(
                model, device, test_criterion, test_loader, dataset_name="Test"
            )
            if (val_scores['mF1Score']>finalScoreAcc):
                finalScoreAcc = val_scores['mF1Score']
                validFinalValue = val_scores
                testFinalValue = test_scores
                prediction = {'test_list': test_list , 'test_label': test_label, 'test_pred': veTest_pred} #dict of test video, label and prediction

            # save results
            epoch_train_losses.append(train_loss)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(test_loss)
            epoch_test_scores.append(test_scores)
            epoch_val_losses.append(val_loss)
            epoch_val_scores.append(val_scores)

        finalOutputAccrossFold[fold] = {
            'epoch_train_losses': epoch_train_losses,
            'epoch_train_scores': epoch_train_scores,
            'epoch_test_losses': epoch_test_losses,
            'epoch_test_scores': epoch_test_scores,
            'epoch_val_losses': epoch_val_losses,
            'epoch_val_scores': epoch_val_scores,
            'validation_best':validFinalValue, 
            'test_best': testFinalValue, 
            'best_test_prediction': prediction}
    
    performance_by_epoch = aggregate_performance_by_epoch(finalOutputAccrossFold)
    best_mean_metrics, best_std_metrics = best_performance_using_kfold_results(finalOutputAccrossFold)
    return performance_by_epoch, best_mean_metrics, best_std_metrics

