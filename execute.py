from dataset.sportsmans_height import Sportsmanheight
from model.simple_classifier import Classifier
import numpy as np
from config.cfg import cfg
import plotly.graph_objects as go


def get_data():
    dataset = Sportsmanheight()()
    predictions = Classifier()(dataset['height'])
    gt = dataset['class']
    return predictions, gt


def calculate_metrics(predictions, gt, threshold):
    predictions_bin = np.where(predictions >= threshold, 1, 0)
    TP = np.sum(np.logical_and(predictions_bin == 1, gt == 1))
    FP = np.sum(np.logical_and(predictions_bin == 1, gt == 0))
    TN = np.sum(np.logical_and(predictions_bin == 0, gt == 0))
    FN = np.sum(np.logical_and(predictions_bin == 0, gt == 1))
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * precision * recall / (precision + recall)
    return TP, FP, TN, FN, accuracy, recall, precision, f1_score


def calculate_metrics_for_all_thresholds(predictions, gt, thresholds):
    results = []
    for threshold in thresholds:
        results.append(calculate_metrics(predictions, gt, threshold))
    return np.array(results)


def plot_precision_recall_curve(metrics, thresholds, area_under_curve):
    precision = metrics[:, 6]
    recall = metrics[:, 5]
    accuracy = metrics[:, 4]
    f1_score = metrics[:, 7]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier'))
    fig.update_layout(
        title=f"Precision-Recall curve (AUC={area_under_curve:.2f})",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    fig.update_traces(hovertemplate="Threshold=%{text}<br>Precision=%{y:.2f}<br>Recall=%{x:.2f}<br>Accuracy=%{customdata[0]:.2f}<br>F1-score=%{customdata[1]:.2f}",
                      text=thresholds,
                      customdata=np.vstack((accuracy, f1_score)).T)
    fig.show()


def plot_roc_curve(metrics, thresholds, area_under_curve):
    fpr = metrics[:, 1] / (metrics[:, 1] + metrics[:, 3])
    tpr = metrics[:, 0] / (metrics[:, 0] + metrics[:, 2])
    accuracy = metrics[:, 4]
    f1_score = metrics[:, 7]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random classifier'))
    fig.update_layout(
        title=f"ROC curve (AUC={area_under_curve:.2f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    fig.update_traces(hovertemplate="Threshold=%{text}<br>TPR=%{y:.2f}<br>FPR=%{x:.2f}<br>Accuracy=%{customdata[0]:.2f}<br>F1-score=%{customdata[1]:.2f}",
                      text=thresholds,
                      customdata=np.vstack((accuracy, f1_score)).T)
    fig.show()

def evaluate_model():
    predictions, gt = get_data()

    # Normalize predictions
    predictions_norm = predictions / cfg.max_height

    # Calculate metrics for all thresholds
    thresholds = np.arange(0, 1.01, 0.01)
    metrics = calculate_metrics_for_all_thresholds(predictions_norm, gt, thresholds)

    # Calculate area under precision-recall curve
    precision = metrics[:, 6]
    recall = metrics[:, 5]
    area_under_curve = np.trapz(precision, recall)

    # Plot precision-recall curve
    plot_precision_recall_curve(metrics, thresholds, area_under_curve)

    # Calculate area under ROC curve
    fpr = metrics[:, 1] / (metrics[:, 1] + metrics[:, 3])
    tpr = metrics[:, 0] / (metrics[:, 0] + metrics[:, 2])
    area_under_curve = np.trapz(tpr, fpr)

    # Plot ROC curve
    plot_roc_curve(metrics, thresholds, area_under_curve)


if __name__ == "__main__":
    evaluate_model()