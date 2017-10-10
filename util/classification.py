import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conf import settings
from sklearn.metrics import accuracy_score, roc_curve, auc


def gini(vector):
    sorted_list = sorted(list(vector))
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(vector) / 2
    return (fair_area - area) / fair_area


def normalized_gini(y_estimate, y_real):
    return gini(y_estimate) / gini(y_real)


def classifier_procedure(model, datasets, submission):
    model.fit(datasets.get_train(), datasets.get_train(True))
    return ClassifierResults(model, datasets, submission)


class ClassifierResults(object):

    train_label = settings.ModelConf.labels.train
    test_label = settings.ModelConf.labels.test
    validate_label = settings.ModelConf.labels.validate

    result_type = "classification"

    def __init__(self, model, datasets, submission):
        self.model = model
        self.datasets = datasets
        self.roc_auc = {}
        self.submission_data = datasets.external(submission)
        self.train_data = self.datasets.get_train(False)
        self.test_data = self.datasets.get_test(False)
        self.validate_data = self.datasets.get_validate(False)

    def data(self, label):
        return self.train_data if label == self.train_label else (
            self.test_data if label == self.test_label else
            self.validate_data)

    def original_output(self, label, apply_inverse=False):
        return self.datasets.get_train(True, apply_inverse) if label == self.train_label else (
            self.datasets.get_test(True, apply_inverse) if label == self.test_label else
            self.datasets.get_validate(True, apply_inverse))

    def prediction(self, label, proba=True, apply_inverse=False):
        model_function = self.model.predict if not proba else self.model.predict_proba
        prediction_values = model_function(self.data(label))
        prediction_values = prediction_values if not proba else [i[-1] for i in prediction_values]
        if apply_inverse:
            return self.datasets.inverse_function[self.datasets.link](prediction_values)
        return prediction_values

    def get_submission_predictions(self, proba=True, apply_inverse=False):
        model_function = self.model.predict if not proba else self.model.predict_proba
        prediction_values = model_function(self.submission_data)
        prediction_values = prediction_values if not proba else [i[-1] for i in prediction_values]
        if apply_inverse:
            return self.datasets.inverse_function[self.datasets.link](prediction_values)
        return prediction_values

    def correct_values(self, label, proba=False, apply_inverse=False):
        original_output = self.original_output(label, apply_inverse)
        estimated_output = self.prediction(label, proba, apply_inverse)
        return sum(original_output == estimated_output) / len(estimated_output)

    def sklearn_accuracy_score(self, label, proba=True, apply_inverse=False):
        return accuracy_score(self.original_output(label, apply_inverse), self.prediction(label, proba, apply_inverse))

    def normalized_gini_score(self, label, proba=True, apply_inverse=False):
        return normalized_gini(self.prediction(label, proba, apply_inverse), self.original_output(label, apply_inverse))

    def sklearn_roc_curve(self, label, proba=True, apply_inverse=False):
        fpr, tpr, thresholds = roc_curve(self.original_output(label, apply_inverse),
                                         self.prediction(label, proba=proba, apply_inverse=apply_inverse))
        roc_auc = auc(fpr, tpr)
        self.roc_auc[label] = roc_auc
        return fpr, tpr, roc_auc

    def plot_roc_curve(self, label, proba=True, apply_inverse=False):
        fpr, tpr, roc_auc = self.sklearn_roc_curve(label, proba, apply_inverse)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=0.8)
        plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC fold %d (AUC = %0.2f)' % (1, roc_auc))
        plt.title("{} : Receiver operating characteristic".format(label.upper()))
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()

