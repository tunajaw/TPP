import numpy as np

from easy_tpp.utils.const import PredOutputIndex
from easy_tpp.utils.metrics import MetricsHelper


@MetricsHelper.register(name='rmse', direction=MetricsHelper.MINIMIZE, overwrite=False)
def rmse_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    pred = predictions[PredOutputIndex.TimePredIndex][seq_mask][:, 0].squeeze()
    label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.sqrt(np.mean((pred - label) ** 2))


@MetricsHelper.register(name='acc', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def acc_metric_function(predictions, labels, **kwargs):
    """Compute accuracy ratio metrics of the type predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: accuracy ratio of the type predictions.
    """
    seq_mask = kwargs.get('seq_mask')
    pred = predictions[PredOutputIndex.TypePredIndex][seq_mask][:, 0].squeeze()
    label = labels[PredOutputIndex.TypePredIndex][seq_mask]
    pred = np.reshape(pred, [-1])
    label = np.reshape(label, [-1])
    return np.mean(pred == label)

@MetricsHelper.register(name='top5', direction=MetricsHelper.MAXIMIZE, overwrite=False)
def top5_metric_function(predictions, labels, **kwargs):
    """Compute rmse metrics of the time predictions.

    Args:
        predictions (np.array): model predictions.
        labels (np.array): ground truth.

    Returns:
        float: average rmse of the time predictions.
    """

    seq_mask = kwargs.get('seq_mask')
    pred = predictions[PredOutputIndex.TimePredIndex][seq_mask]
    label = labels[PredOutputIndex.TimePredIndex][seq_mask]

    # pred = np.reshape(pred, [-1])
    # label = np.reshape(label, [-1])
    print(pred.shape, label.shape)
    k = 5
    
    # Step 1: Find the indices of the top k scores
    # We use argpartition to perform an efficient partial sort
    top_k_predictions = np.argpartition(pred, -k, axis=1)[:, -k:]
    
    # Step 2: Check if the true labels are in the top k predictions
    # Create an array of labels repeated to match top_k_predictions' shape
    label_matrix = np.tile(label[:, np.newaxis], (1, k))
    
    # Step 3: Calculate the matches by comparing top_k_predictions with the label matrix
    matches = np.any(top_k_predictions == label_matrix, axis=1)
    
    # Calculate the accuracy as the mean of matches
    accuracy = np.mean(matches)

    return accuracy
