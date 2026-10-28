from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_eval_metrics(y_true, y_pred) -> dict:
    """Computes binary classification evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (e.g. "cat" or "dog")
    y_pred : array-like
        Predicted labels (e.g. "cat" or "dog")

    Returns
    -------
    dict
        Dictionary containing test_accuracy, test_precision, test_recall,
        and test_f1 keyed for direct use with mlflow.log_metrics
    """
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, average="binary", pos_label="dog"),
        "test_recall": recall_score(y_true, y_pred, average="binary", pos_label="dog"),
        "test_f1": f1_score(y_true, y_pred, average="binary", pos_label="dog"),
    }
