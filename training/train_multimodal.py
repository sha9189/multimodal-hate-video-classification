
import torch
import torch.nn.functional as F
from training.evaluation import evalMetric #, average_metrics
from utils.utils import load_config

config = load_config('configs/configs.yaml')

def train_multimodal(
        log_interval, 
        modalities,
        model, 
        device, 
        train_loader, 
        criterion, 
        optimizer, 
        epoch):
    """Function for training model through one epoch including model weight updates
    Inputs:
        log_interval: after how many batches to print the performance on the batch while training
        model: training model
        device: "cpu" or "gpu"
        train_loader: training dataloader
        optimizer: optimizer object
        epoch: current epoch number
    Outputs:
        loss: average loss
        scores: performance metrics dict
    """
    
    # set model as training mode
    model.train()

    losses = []
    scores = []
    all_y = []
    all_y_pred = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device 
        for modality in modalities:
            X[modality+'_features'] = X[modality+'_features'].to(device)
        y = y.to(device).view(-1, )
        N_count += X[modality+'_features'].size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        loss = criterion(output, y)
        losses.append(loss.item())

        y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

        # collect all y and y_pred in all batches
        all_y.extend(y)
        all_y_pred.extend(y_pred)

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            metrics = evalMetric(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
            # print('\nTrain Epoch: {epoch} [{N_count}/{total_count} ({percentage:.0f}%)]\tLoss: {loss:.6f}, Accu: {accuracy:.2f}%, MF1 Score: {mF1:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}'.format(
            #     epoch=epoch + 1, 
            #     N_count=N_count, 
            #     total_count=len(train_loader.dataset), 
            #     percentage=100. * (batch_idx + 1) / len(train_loader), 
            #     loss=loss.item(), 
            #     accuracy=100 * metrics['accuracy'], 
            #     mF1=metrics['mF1Score'], 
            #     f1=metrics['f1Score'], 
            #     auc=metrics['auc'], 
            #     precision=metrics['precision'], 
            #     recall=metrics['recall']))
            print('\nTrain Epoch: {epoch} [{N_count}/{total_count} ({percentage:.0f}%)]\tLoss: {loss:.6f}, Accu: {accuracy:.2f}%, MF1 Score: {mF1:.4f}'.format(
                epoch=epoch + 1, 
                N_count=N_count, 
                total_count=len(train_loader.dataset), 
                percentage=100. * (batch_idx + 1) / len(train_loader), 
                loss=loss.item(), 
                accuracy=100 * metrics['accuracy'], 
                mF1=metrics['mF1Score']))

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    scores = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # scores = average_metrics(scores)
    loss = sum(losses) / len(losses)

    return loss, scores



def validation_multimodal(
        modalities, 
        model, 
        device, 
        criterion, 
        test_loader, 
        dataset_name):
    """Function to evaluate model on hold-out set (test or validation set)
    Inputs:
        model: model to evaluate
        device: "cpu" or "gpu"
        optimizer: optimizer
        test_loader: test or validation dataloader 
    Outputs:
        test_loss: loss on the entire dataset
        metrics: dict of 8 performance metrics on the entire dataset
        y_pred = list of predicted values for the entire dataset
    """
    # set model as testing mode
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            for modality in modalities:
                X[modality+'_features'] = X[modality+'_features'].to(device)
            y = y.to(device).view(-1, )

            output = model(X)

            loss = criterion(output, y)
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    
    # print('{dataset_name} set: ({N_count:d} samples): Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, MF1 Score: {mF1:.4f}'.format(
    #             dataset_name=dataset_name,
    #             N_count=len(all_y), 
    #             test_loss=test_loss, 
    #             accuracy=100 * metrics['accuracy'], 
    #             mF1=metrics['mF1Score']))
  
    # # save Pytorch models of best record
    # torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pt'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pt'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy())
