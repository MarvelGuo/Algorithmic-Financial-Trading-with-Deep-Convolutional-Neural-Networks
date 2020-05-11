"""
Created on 2020/4/11 19:29
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""

from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def plot_history(history):
    plt.figure(figsize=(8,6),dpi=80)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['f1_metric'])
    plt.plot(history.history['val_f1_metric'])
    plt.title('Model Metrics')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc', 'f1', 'val_f1'], loc='upper left')

def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, axis=0, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))

def Confusion_Mat(y_test_classes, pred_classes):
    conf_mat = confusion_matrix(y_test_classes, pred_classes)
    conf_mat_pct = np.round(conf_mat / np.sum(conf_mat, axis=1).reshape(-1, 1), 2)
    print('Confusion Matrix:')
    print(conf_mat)

    plt.figure(figsize=(6, 6), dpi=80)
    ax = sns.heatmap(data=conf_mat_pct, annot=True, cmap='RdPu')
    ax.set_title('Confusion Matrix')
    ax.set(ylabel='True Label', xlabel='Predicted Label')
    ax.set_aspect("equal")
    plt.show()

    prec = []
    for i, row in enumerate(conf_mat):
        prec.append(np.round(row[i] / np.sum(row), 2))
        print("precision of class {} = {}".format(i, prec[i]))
    print("precision avg", round(sum(prec) / len(prec),2), '\n')
    return conf_mat

def test_analysis(model, x_test, y_test):
    pred = model.predict(x_test)
    pred_classes = np.argmax(pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    return pred_classes, y_test_classes

def cal_f1_scores(y_test_classes, pred_classes):
    f1_weighted = f1_score(y_test_classes, pred_classes, labels=None,
                           average='weighted', sample_weight=None)
    f1_macro = f1_score(y_test_classes, pred_classes, labels=None,
                                       average='macro', sample_weight=None)
    f1_micro = f1_score(y_test_classes, pred_classes, labels=None,
                                       average='micro', sample_weight=None)
    print("\nF1 score (weighted)", round(f1_weighted,2))
    print("F1 score (macro)", round(f1_macro,2))
    print("F1 score (micro)", round(f1_micro,2))  # weighted and micro preferred in case of imbalance

    print("cohen's Kappa", cohen_kappa_score(y_test_classes, pred_classes), '\n')

if __name__ == '__main__':
    pass