import os, time, timeit, json, torch, mat73, argparse
import numpy as np, pandas as pd
from scipy.signal import resample
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt, seaborn as sns
sns.set_style("ticks")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch import nn
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import ConfusionMatrixDisplay

from aeon.classification.deep_learning import InceptionTimeClassifier
from aeon.networks import InceptionNetwork
from aeon.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.distance_based import ProximityForest
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.interval_based import DrCIFClassifier
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

from sktime.transformations.panel.rocket import Rocket, MiniRocket, MultiRocket
from sktime.classification.feature_based import Catch22Classifier
from sktime.transformations.panel.catch22 import Catch22


def run_tabular_models(X_train, y_train, X_test, y_test, window_size):

    tabular_models = {
        'Logistic': LogisticRegression(random_state=0, n_jobs=-1, solver='newton-cg'),
        'RandomForest': RandomForestClassifier(n_estimators=150, random_state=0),
        'KNN': KNeighborsClassifier(),
        'Ridge': RidgeClassifier(random_state=0),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_features=0.1,
                                                    criterion="entropy", n_jobs=-1,
                                                    random_state =0)
    }

    metrics_df = pd.DataFrame(columns=['model', 'window_size', 'runtime', 'precision', 'recall', 'f1-score'])
    for model in tabular_models.items():
        metrics = predict_eval(model, window_size, X_in=(X_train, X_test),
                               y_in=(y_train, y_test))
        these_metrics = pd.DataFrame(data=metrics)
        metrics_df = pd.concat([metrics_df, these_metrics])
    
    return metrics_df


def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')

def predict_eval(model, win_size, X_in=None, y_in=None, starttime=None, adapt_threshold=False, verbose=1):
    target_names = ['ADL', 'Fall']
    model_name, clf = model
    has_proba = hasattr(clf, 'predict_proba')
    if X_in is not None:
        X_train, X_test = X_in
        y_train, y_test = y_in
    X_train = X_train[:, :win_size]
    X_test = X_test[:, :win_size]
    print("> {:<17} (win_size={})".format(model_name, win_size), end='\t')
    if starttime is None:
        starttime = timeit.default_timer()
    if verbose:
        print("Training", end='')
    clf.fit(X_train, y_train)
    if verbose:
        print("/Testing", end=' ')
    if has_proba:
        probs = clf.predict_proba(X_test)[:, 1]
        train_probs = clf.predict_proba(X_train)[:, 1]
    else:
        if adapt_threshold:
            adapt_threshold = False
            print("Setting adapt_threshold=False since chosen classifier has no predict_proba() method")
    if adapt_threshold:
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(y_train, to_labels(train_probs, t)) for t in thresholds]
        # get best threshold
        ix = np.argmax(scores)
        print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
        y_pred = to_labels(probs, thresholds[ix])
    else:
        y_pred = clf.predict(X_test)
    runtime = timeit.default_timer() - starttime
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    if verbose:
        print(f"in {np.round(runtime,2)} secs.")
    if verbose > 1:
        if has_proba:
            print(f'AUC: {np.round(roc_auc_score(y_test, probs), 2)}')
        else:
            print("Skipping AUC since chosen classifier has no predict_proba() method")
            print(classification_report(y_test, y_pred, target_names=target_names))
    if verbose > 2:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.grid(False)
        plt.show()
    return dict({'model': [model_name], 'window_size': win_size, 'runtime':[runtime],
                 'precision':[precision], 'recall':[recall], 'f1-score':[f1]})