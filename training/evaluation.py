from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, precision_score
import numpy as np


def evalMetric(y_true, y_pred):
    """Function returns 8 types of model evaluation metrics for classification
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