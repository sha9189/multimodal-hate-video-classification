# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import load_config, fix_the_random
import pickle
import numpy as np
from models.combine_model import textModel, audioModel, videoModel, combinedModel
from data_preprocessing.multimodal_dataset import multiModelData

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []
    scores = []
    for batch_idx, (X_text, X_vid, X_aud, y) in enumerate(train_loader):
        X_text, X_vid, X_aud, y = (X_text.float()).to(device), (X_vid.float()).to(device), (X_aud.float()).to(device), y.to(device).view(-1,)
        #y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(X_text, X_vid, X_aud)
        print("Output shape:", output.shape)
        print("y shape:", y.shape)         
        loss = torch.nn.functional.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device))
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx + 1, len(train_loader), loss.item()))
        losses.append(loss.item())
    return losses

def validation(model, device, valid_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_text, X_vid, X_aud, y in valid_loader:
            X_text, X_vid, X_aud, y = X_text.to(device), X_vid.to(device), X_aud.to(device), y.to(device)
            output = model(X_text, X_vid, X_aud)
            test_loss += torch.nn.functional.cross_entropy(output, y, reduction='sum').item()
    return test_loss / len(valid_loader.dataset)

def test_model(model_cls, dataset_cls, epochs, optimizer_name, allDataAnnotation):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_cls().to(device)
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    results = {}
    for fold, data in allDataAnnotation.items():
        train_dataset = dataset_cls(data['train'][0], data['train'][1])
        val_dataset = dataset_cls(data['val'][0], data['val'][1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        for epoch in range(epochs):
            train(70, model, device, train_loader, optimizer, epoch)
            loss = validation(model, device, val_loader)
            print(f'Validation loss for fold {fold} at epoch {epoch}: {loss}')
        results[fold] = loss

    return results

