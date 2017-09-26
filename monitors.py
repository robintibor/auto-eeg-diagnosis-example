import numpy as np
from sklearn.metrics import roc_auc_score

from braindecode.datautil.iterators import _compute_start_stop_block_inds


class CroppedDiagnosisMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        preds_per_trial = compute_preds_per_trial(
            all_preds, dataset, input_time_length=self.input_time_length,
            n_stride=self.n_preds_per_input)

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        out = {column_name: float(misclass)}
        y = dataset.y

        n_true_positive = np.sum((y == 1) & (pred_labels_per_trial == 1))
        n_positive = np.sum(y == 1)
        if n_positive > 0:
            sensitivity = n_true_positive / float(n_positive)
        else:
            sensitivity = np.nan
        column_name = "{:s}_sensitivity".format(setname)
        out.update({column_name: float(sensitivity)})

        n_true_negative = np.sum((y == 0) & (pred_labels_per_trial == 0))
        n_negative = np.sum(y == 0)
        if n_negative > 0:
            specificity = n_true_negative / float(n_negative)
        else:
            specificity = np.nan
        column_name = "{:s}_specificity".format(setname)
        out.update({column_name: float(specificity)})
        if (n_negative > 0) and (n_positive > 0):
            auc = roc_auc_score(y, mean_preds_per_trial[:,1])
        else:
            auc = np.nan
        column_name = "{:s}_auc".format(setname)
        out.update({column_name: float(auc)})
        return out

def compute_preds_per_trial(preds_per_batch, dataset, input_time_length,
                            n_stride):
    n_trials = len(dataset.X)
    i_pred_starts = [input_time_length -
                     n_stride] * n_trials
    i_pred_stops = [t.shape[1] for t in dataset.X]

    start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
        i_pred_starts,
        i_pred_stops, input_time_length, n_stride,
        False)

    n_rows_per_trial = [len(block_inds) for block_inds in
                        start_stop_block_inds_per_trial]

    all_preds_arr = np.concatenate(preds_per_batch, axis=0)
    i_row = 0
    preds_per_trial = []
    for n_rows in n_rows_per_trial:
        preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
        i_row += n_rows
    assert i_row == len(all_preds_arr)
    return preds_per_trial


class CroppedNonDenseTrialMisclassMonitor(object):
    """
    Compute trialwise misclasses from predictions for crops for non-dense predictions.

    Parameters
    ----------
    input_time_length: int
        Temporal length of one input to the model.
    """

    def __init__(self, input_time_length, n_preds_per_input):
        self.input_time_length = input_time_length
        self.n_preds_per_input = n_preds_per_input

    def monitor_epoch(self, ):
        return

    def monitor_set(self, setname, all_preds, all_losses,
                    all_batch_sizes, all_targets, dataset):
        """Assuming one hot encoding for now"""
        n_trials = len(dataset.X)
        i_pred_starts = [self.input_time_length -
                         self.n_preds_per_input] * n_trials
        i_pred_stops = [t.shape[1] for t in dataset.X]

        start_stop_block_inds_per_trial = _compute_start_stop_block_inds(
            i_pred_starts,
            i_pred_stops, self.input_time_length, self.n_preds_per_input,
            False)

        n_rows_per_trial = [len(block_inds) for block_inds in
                            start_stop_block_inds_per_trial]

        all_preds_arr = np.concatenate(all_preds, axis=0)
        i_row = 0
        preds_per_trial = []
        for n_rows in n_rows_per_trial:
            preds_per_trial.append(all_preds_arr[i_row:i_row + n_rows])
            i_row += n_rows

        mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
        mean_preds_per_trial = np.array(mean_preds_per_trial)

        pred_labels_per_trial = np.argmax(mean_preds_per_trial, axis=1)
        assert pred_labels_per_trial.shape == dataset.y.shape
        accuracy = np.mean(pred_labels_per_trial == dataset.y)
        misclass = 1 - accuracy
        column_name = "{:s}_misclass".format(setname)
        return {column_name: float(misclass)}