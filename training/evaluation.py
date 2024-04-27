from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, precision_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import load_config
import os

config = load_config('configs/configs.yaml')

def evalMetric(y_true, y_pred):
    """Function returns 6 types of model evaluation metrics for classification - accuracy, 
    macro-F1 score, F1 score, AUC, Recall and Precision. 
    Inputs: 
        y_true: Actual target
        y_pred: predicted target
    Outputs:
        dict with keys - accuracy, mF1Score(F1-Macro), f1Score, auc, precision, recall
    """
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred))
    precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred))
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c,
           'precision': precisionScore, 'recall': recallScore})


def best_performance_using_kfold_results(finalOutputAccrossFold):
    """Function to take k-fold CV results and return average/std of results for best performing model in each fold
    """
    allValueDict = {}
    for fold in (finalOutputAccrossFold.keys()):
        for val in finalOutputAccrossFold[fold]['test_best']:
            try:
                allValueDict[val].append(finalOutputAccrossFold[fold]['test_best'][val])
            except:
                allValueDict[val]=[finalOutputAccrossFold[fold]['test_best'][val]]

    metrics_mean = {}
    metrics_std = {}
    # get avg/std of performance of best model from each fold
    for metric in allValueDict:
        metric_mean = np.mean(allValueDict[metric])
        metric_std = np.std(allValueDict[metric])
        metrics_mean[metric] = metric_mean
        metrics_std[metric] = metric_std
    return metrics_mean, metrics_std


def calculate_average_list(list_of_lists):
    """Given a list of equally-sized lists, get one list as output where each 
    element is the average of the value in each input list at the same position
    Example usage: You have one list of accuracy by epoch for each fold, hence, you
    would have k-lists of accuracies. Use this function to combine the values in each 
    fold to get the avg accuracy per epoch over all folds
    Input:
        list_of_lists: list of equally sized lists
    Output:
        averages: one list with value for each index as the average of all lists for the same index 
    """

    num_lists = len(list_of_lists)
    list_length = len(list_of_lists[0])  # Assuming all sublists are of equal length
    averages = [0] * list_length

    for i in range(list_length):
        total = sum(lst[i] for lst in list_of_lists)
        averages[i] = total / num_lists

    return averages


def calculate_average_dict(list_of_dicts):
    """Function that takes a list of dicts and returns one dict with the average value 
    for each key in all the input dicts.
    Example Usage: Given 
    """
    keys = list_of_dicts[0].keys()  # Assuming all dictionaries have the same keys
    averages = {}

    for key in keys:
        total = sum(d[key] for d in list_of_dicts)
        averages[key] = total / len(list_of_dicts)

    return averages


def aggregate_performance_by_epoch(finalOutputAccrossFold):
    """Function that takes k-fold CV results and returns avg performance over folds by epoch"""

    totalOutput = {}
    
    loss_sets = ['epoch_train_losses', 'epoch_val_losses', 'epoch_test_losses']
    score_sets = ['epoch_train_scores', 'epoch_test_scores', 'epoch_val_scores']
    
    fold1 = next(iter(finalOutputAccrossFold))
    fold1 = finalOutputAccrossFold.get(fold1)
    epochs = len(fold1[loss_sets[0]])
    
    # folds_count = len(finalOutputAccrossFold.keys())


    # metrics_list = ["accuracy", "mF1Score", "f1Score", "auc", "precision", "recall"]

    for loss_set in loss_sets:
        list_of_lists = []
        for fold in finalOutputAccrossFold.keys():
            list_of_lists.append(finalOutputAccrossFold[fold][loss_set])
        averages = calculate_average_list(list_of_lists)
        totalOutput[loss_set] = averages
    
    # create placeholders for storing values of metrics list
    for score_set in score_sets:
        totalOutput[score_set] = [] 
    
    for score_set in score_sets:
        for epoch in range(epochs):
            # process a list of all scores from k-folds for a given score set and epoch
            list_of_dicts = []
            for fold in finalOutputAccrossFold.keys():
                list_of_dicts.append(finalOutputAccrossFold[fold][score_set][epoch])
            averages = calculate_average_dict(list_of_dicts)
            averages['loss'] = totalOutput[score_set.replace("scores", "losses")][epoch]
            totalOutput[score_set].append(averages)
    
    # delete loss sets since score set includes the loss
    for loss_set in loss_sets:
        totalOutput.pop(loss_set)
    return totalOutput   


def get_metric_by_epoch(metric_name:str, score_set:str, performance_by_epoch:dict):
    """Function to extract metric value by epoch and return a list
    Inputs:
        metric_name: accuracy/mF1Score/f1Score/auc/precision/recall
        score_set: epoch_train_scores/epoch_val_scores/epoch_test_scores
        performance_by_epoch: performance_by_epoch variable returned by test_model
    Output:
        metric_by_epoch: list of model metric score by epoch 
    """
    metric_by_epoch = []
    for epoch_data in performance_by_epoch[score_set]:
        metric_by_epoch.append(epoch_data[metric_name])
    return metric_by_epoch


def plot_all_metrics(performance_by_epoch):
    """Function plots all the seven metrics being captured for analysis"""
    metrics = ["loss", "mF1Score", "accuracy", "f1Score", "auc", "precision", "recall"]
    score_sets = ["epoch_train_scores", "epoch_val_scores", "epoch_test_scores"]
    score_set_names = ["training set", "validation set", "test set"]

    epochs = range(1, len(performance_by_epoch[score_sets[0]])+1)

    fig, axs = plt.subplots(3, 3, figsize=(12, 8)) # WxH

    for i, metric in enumerate(metrics):
        train_metrics = get_metric_by_epoch(metric, score_sets[0], performance_by_epoch)
        val_metrics = get_metric_by_epoch(metric, score_sets[1], performance_by_epoch)
        test_metrics = get_metric_by_epoch(metric, score_sets[2], performance_by_epoch)

        max_val_metric_epoch = (np.argmax(val_metrics) + 1) if metric != "loss" else (np.argmin(val_metrics) + 1)

        # pick the subplot for the given metric
        row, col = int(i/3), int(i%3)
        axs[row, col].plot(epochs, train_metrics, 'b', label='Training ' + metric)
        axs[row, col].plot(epochs, val_metrics, 'r', label='Validation ' + metric)
        axs[row, col].plot(epochs, test_metrics, 'g', label='Test ' + metric)
        axs[row, col].set_title(f"{metric.capitalize()} vs Epoch")
        axs[row, col].set_xlabel('Epochs')
        axs[row, col].set_ylabel(metric)
        axs[row, col].legend()

        # Add vertical line at the epoch with highest validation metric value
        axs[row, col].axvline(x=max_val_metric_epoch, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()


def save_results(performance_by_epoch, name_of_entry):
    """Function that saves the best model's performance on test set to ./data/results.csv"""
    best_epoch_idx = np.argmax(get_metric_by_epoch("mF1Score", "epoch_val_scores", performance_by_epoch))
    test_metrics = performance_by_epoch["epoch_test_scores"][best_epoch_idx]
    test_metrics["LR"] = config["LEARNING_RATE"]
    test_metrics["Name"] = name_of_entry
    test_metrics = pd.DataFrame.from_dict(test_metrics, orient="index").T

    results_file_path = config["DATASET_FOLDER"]+"results.csv"
    if os.path.exists(results_file_path):
        results = pd.read_csv(results_file_path)
    else:
        columns = ['Name', 'LR', 'accuracy', 'mF1Score', 'f1Score', 'auc', 'precision', 'recall', 'loss']
        results = pd.DataFrame(columns=columns)
    
    results = pd.concat([results, test_metrics], axis=0).sort_values(by='mF1Score', ascending=False)
    results.to_csv(results_file_path, index=False)
    print("Results saved successfully!")
    