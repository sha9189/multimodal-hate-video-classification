import torch
import torch.nn.functional as F
from training.evaluation import evalMetric

def train(log_interval, model, device, train_loader, optimizer, epoch):
    """Function for training model through one epoch including model weight updates
    Inputs:
        log_interval: after how many batches to print the performance on the batch while training
        model: training model
        device: "cpu" or "gpu"
        train_loader: training dataloader
        optimizer: optimizer object
        epoch: current epoch number
    Outputs:
        losses: list of losses, one value per batch
        scores: list of performance metrics dict, one value per batch
    """
    
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, (X_text, y) in enumerate(train_loader):
        # distribute data to device 
        X_text, y = (X_text.float()).to(device), y.to(device).view(-1, )
    
        N_count += X_text.size(0)

        optimizer.zero_grad()
        output = model(X_text)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device))
        #loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        metrics = evalMetric(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(metrics)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            # print(f'Train Epoch: {epoch + 1} [{N_count}/{len(train_loader.dataset)} ({(100. * (batch_idx + 1) / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}, Accu: {(100 * metrics['accuracy']):.2f}%, MF1 Score: {(metrics['mF1Score']):.4f}, F1 Score: {metrics['f1Score']:.4f}, Area Under Curve: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}, Recall Score: {metrics['recall']:.4f}')
            print(f'Train Epoch: {epoch + 1} [{N_count}/{len(train_loader.dataset)} ({(100. * (batch_idx + 1) / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}, Accu: {(100 * metrics["accuracy"]):.2f}%, MF1 Score: {(metrics["mF1Score"]):.4f}, F1 Score: {metrics["f1Score"]:.4f}, Area Under Curve: {metrics["auc"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall Score: {metrics["recall"]:.4f}')

    return losses, scores


def validation(model, device, optimizer, test_loader):
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
        for X_text, y in test_loader:
            # distribute data to device
            X_text, y = (X_text.float()).to(device), y.to(device).view(-1, )

            output = model(X_text)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # to compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    print("====================")
    # try:
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    # except:
    #   metrics = None

    # show information
    print('\nTest set: ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
                len(all_y), test_loss, 100 * metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))
  
    # # save Pytorch models of best record
    # torch.save(model.state_dict(), os.path.join(save_model_path, '3dcnn_epoch{}.pt'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, '3dcnn_optimizer_epoch{}.pt'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy())
