import torch

def get_labels_and_preds(device, model, dataloader):
    """
    Gets the true and predicted labels for a given dataloader.

    Returns
    -------
    labels: list
    preds: list
    """
    labels = []
    preds = []
    ## Get every prediction and label
    for X, Y in dataloader:
        X = X.to(device).float()
        with torch.no_grad():
            pred = model(X)
            pred = pred.softmax(-1).argmax(-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(Y.numpy())
    
    return labels, preds